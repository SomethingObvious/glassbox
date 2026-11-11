"""Granite 3.0 1B-A400M as a routing-only ONNX model for Figure 2: token ids in, every layer's router logits out.

The experts are stored in 8 bits and the routers stay in fp32. The 4-bit version picked the same 8 experts as the
full model at only 57% of token and layer pairs, against 94% for this one.
"""

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .route import MODEL

if TYPE_CHECKING:
    import numpy as np
    import torch

ROUTE = Path(__file__).resolve().parent.parent / "docs" / "data" / "route.json"


def _dense(self: "torch.nn.Module", layer_input: "torch.Tensor") -> "torch.Tensor":
    """Every expert on every token, weighted by the router's gates. Two 2-D matmuls so the quantizer can reach them,
    where the original splits tokens by a count only known at run time and so can't be traced."""
    import torch

    bsz, length, emb = layer_input.shape
    x = layer_input.reshape(-1, emb)
    logits = self.router.layer(x).float()
    self.seen = logits
    top, idx = logits.topk(self.router.top_k, dim=1)
    gates = torch.zeros_like(logits).scatter(1, idx, top.softmax(1)).to(x.dtype)
    experts, inner = self.input_linear.num_experts, self.hidden_size
    h = (x @ self.w_in).view(-1, experts, 2 * inner)
    h = self.activation(h[..., :inner]) * h[..., inner:]
    y = (h * gates[..., None]).reshape(-1, experts * inner) @ self.w_out
    return y.view(bsz, length, emb)


def _sets(logits: "np.ndarray") -> "np.ndarray":
    import numpy as np

    return np.sort(np.argsort(-logits, -1)[..., :8], -1)


def export(out: Path) -> None:
    """Writes routes.onnx and routes.onnx.data to `out`, then checks them against the routes in route.json."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").eval()
    for layer in model.model.layers:
        moe = layer.block_sparse_moe
        e, o, d = moe.input_linear.weight.shape
        moe.register_buffer("w_in", moe.input_linear.weight.detach().permute(2, 0, 1).reshape(d, e * o).contiguous())
        moe.register_buffer("w_out", moe.output_linear.weight.detach().permute(0, 2, 1).reshape(-1, d).contiguous())
        moe.forward = _dense.__get__(moe)

    class Routes(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = model.model

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # An explicit mask skips transformers' packed-sequence check, which uses an op ONNX lacks.
            self.body(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
            return torch.stack([layer.block_sparse_moe.seen for layer in self.body.layers])

    runs = json.loads(ROUTE.read_text(encoding="utf-8"))["runs"]
    with tempfile.TemporaryDirectory() as tmp:
        full_path = Path(tmp) / "routes.onnx"
        ids = tok(runs[0]["text"], return_tensors="pt").input_ids
        torch.onnx.export(
            Routes().eval(),
            (ids,),
            str(full_path),
            input_names=["input_ids"],
            output_names=["router_logits"],
            dynamic_axes={"input_ids": {1: "tokens"}, "router_logits": {1: "tokens"}},
            opset_version=21,
            dynamo=False,
            external_data=True,
        )
        full = onnx.load(str(full_path))
        shapes = {i.name: tuple(i.dims) for i in full.graph.initializer}
        routers = [n.name for n in full.graph.node if n.op_type == "MatMul" and shapes.get(n.input[1], (0,))[-1] == 32]
        quantizer = MatMulNBitsQuantizer(full, bits=8, block_size=32, is_symmetric=True, nodes_to_exclude=routers)
        quantizer.process()
        small = quantizer.model.model

    # The embedding table in fp16, cast back after the lookup.
    name = "body.embed_tokens.weight"
    table = next(i for i in small.graph.initializer if i.name == name)
    table.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(table).astype(np.float16), name))
    gather = next(n for n in small.graph.node if n.op_type == "Gather" and n.input[0] == name)
    gather.output[0], looked_up = gather.output[0] + "_half", gather.output[0]
    cast = helper.make_node("Cast", [gather.output[0]], [looked_up], to=TensorProto.FLOAT, name="embed_to_float")
    small.graph.node.insert(list(small.graph.node).index(gather) + 1, cast)
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(small, str(out / "routes.onnx"), save_as_external_data=True, location="routes.onnx.data")

    session = ort.InferenceSession(str(out / "routes.onnx"), providers=["CPUExecutionProvider"])
    same = total = 0
    for run in runs:
        ids = tok(run["text"], return_tensors="np").input_ids.astype(np.int64)
        got = _sets(session.run(None, {"input_ids": ids})[0])
        want = np.sort(np.array([layer["experts"] for layer in run["layers"]]), -1)
        same += int((got == want).all(-1).sum())
        total += got.shape[0] * got.shape[1]
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e9
    print(f"{out}: {size:.2f} GB, same 8 experts as route.json at {same / total:.1%} of token and layer pairs")
