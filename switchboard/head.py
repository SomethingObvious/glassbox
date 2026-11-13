"""Figure 1's live run as two ONNX graphs built on the 4-bit Qwen3-1.7B the page already ran.

Both keep their weights in the file Exempliss hosts, apart from a 4-bit copy of the LM head that goes in beside the
EAGLE head's. They are made for WebGPU graph capture, where a pass is replayed from recorded GPU commands. That
needs fixed shapes and every input on the GPU, so the attention mask is swapped for the two numbers attention reads
from it, and the KV cache is one fixed buffer per layer that a pass writes its new rows into. A guess the big model
turns down needs no cleanup, as the next pass writes over its row.

The big graph writes alone, one token a pass. The pass graph checks two guesses and runs the EAGLE-3 head right after
the big model on what it read, so it guesses after every row in the same call. It also works out which rows the
next pass reads, as the page starts that pass before the picks of this one are back on the CPU.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .draft import HEAD, MOST, TARGET, accepted

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import onnx
    import onnxruntime as ort

BIG = {
    "repo_id": "Exempliss/Qwen3-1.7B-ONNX-Split",
    "revision": "f9b9656836a4315a690d0270f885b0758923bd13",
}
LAYERS = (2, 14, 25)  # the same hidden states eagle.py reads
DRAFT = Path(__file__).resolve().parent.parent / "docs" / "data" / "draft.json"
ROWS = 3  # the pass graph's rows: the last token and two guesses
ENTRIES = 1024  # the length of the CPU check's cache buffers
BLOCK = 16  # both vocabularies divide by it, and under 32 WebGPU gives the first step a thread for every block
STRETCH = 128  # the 4-bit copy of the LM head ranks the vocabulary in stretches of this many tokens
BEST = 8  # and this many stretches are scored again from the fp16 table


def first_max(
    x: str, out: str, size: int, ids: str = "", past: int = 0
) -> tuple[list["onnx.NodeProto"], list["onnx.TensorProto"]]:
    """Nodes and constants that put the index of the first largest of `size` values along the last axis in `out`.

    That is ArgMax, which onnxruntime's WebGPU provider runs as one shader thread walking the whole vocabulary, and
    it was 8.7 ms of a 20 ms pass. Both reductions here go in two steps, as one step over the whole axis gets a
    single small workgroup for each row. With `ids` (floats under `past`, shaped like `x`) the values are those
    tokens' and `out` is the lowest id among the largest.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    assert size % BLOCK == 0, f"{size} values do not split into blocks of {BLOCK}"
    consts = [
        numpy_helper.from_array(np.float32(0), f"{out}/zero"),
        numpy_helper.from_array(np.float32(1), f"{out}/one"),
        numpy_helper.from_array(np.float32(past or size), f"{out}/past"),
        numpy_helper.from_array(np.array([0, 0, size // BLOCK, BLOCK], np.int64), f"{out}/blocks"),
    ]

    def node(kind: str, inputs: list[str], name: str, **attrs: object) -> "onnx.NodeProto":
        # Named after what it makes, as `joined` tells the head's nodes apart by name.
        return helper.make_node(kind, inputs, [name], name=name, **attrs)

    def reduced(kind: str, values: str, name: str, keepdims: int) -> list["onnx.NodeProto"]:
        return [
            node("Reshape", [values, f"{out}/blocks"], f"{name}_blocks"),
            node(kind, [f"{name}_blocks"], f"{name}_each", axes=[-1], keepdims=0),
            node(kind, [f"{name}_each"], name, axes=[-1], keepdims=keepdims),
        ]

    nodes = [
        *([] if ids else [node("Range", [f"{out}/zero", f"{out}/past", f"{out}/one"], f"{out}/ramp")]),
        *reduced("ReduceMax", x, f"{out}/top", 1),
        node("Equal", [x, f"{out}/top"], f"{out}/is_top"),
        # Every place that holds the largest value keeps its index, and the rest get one past the end.
        node("Where", [f"{out}/is_top", ids or f"{out}/ramp", f"{out}/past"], f"{out}/places"),
        *reduced("ReduceMin", f"{out}/places", f"{out}/first", 0),
        node("Cast", [f"{out}/first"], out, to=TensorProto.INT64),
    ]
    return nodes, consts


def rough(table: "np.ndarray") -> "onnx.ModelProto":
    """The LM head, which is the embedding `table` on its side, as one 4-bit MatMulNBits from `x` to `lm/rough`."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    vocab, width = table.shape
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "lm_head.weight"], ["lm/rough"], name="lm/rough")],
        "lm_head",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, "rows", width])],
        [helper.make_tensor_value_info("lm/rough", TensorProto.FLOAT16, [1, "rows", vocab])],
        [numpy_helper.from_array(np.ascontiguousarray(table.T), "lm_head.weight")],
    )
    opsets = [helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)]
    quantizer = MatMulNBitsQuantizer(
        helper.make_model(graph, opset_imports=opsets, ir_version=7), bits=4, block_size=32, is_symmetric=True
    )
    quantizer.process()
    return quantizer.model.model


def shortlist(x: str, scan: "onnx.NodeProto", vocab: int) -> tuple[list["onnx.NodeProto"], list["onnx.TensorProto"]]:
    """Nodes that put fp16 logits for the `BEST * STRETCH` likeliest tokens after `x` in `logits_fp16`.

    The whole fp16 table is 622 MB, and reading it was 3.7 ms of a 9 ms pass. `scan` is a 4-bit copy that scores
    every token, the vocabulary is ranked in stretches by the best token in each, and only the tokens of the best
    stretches are scored from the fp16 table. Their ids come out in `choices`, as floats.

    Both are also flattened inside the graph, as `lm/logits` and `lm/choices`. A graph's output has to come out of
    a kernel, since a Reshape into an output buffer is a copy that WebGPU graph capture does not replay.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    assert vocab % STRETCH == 0, f"{vocab} tokens do not split into stretches of {STRETCH}"
    assert (BEST * STRETCH) % BLOCK == 0, f"{BEST * STRETCH} choices do not split into blocks of {BLOCK}"
    consts = [
        numpy_helper.from_array(np.array([0, 0, vocab // STRETCH, STRETCH], np.int64), "lm/stretches"),
        numpy_helper.from_array(np.array([BEST], np.int64), "lm/best"),
        numpy_helper.from_array(np.float32(STRETCH), "lm/stretch"),
        numpy_helper.from_array(np.array([0, 0, BEST, 1], np.int64), "lm/column_of_starts"),
        numpy_helper.from_array(np.arange(STRETCH, dtype=np.float32), "lm/ramp"),
        numpy_helper.from_array(np.array([0, 0, BEST * STRETCH], np.int64), "lm/flat"),
        numpy_helper.from_array(np.array([0, 0, -1, 1], np.int64), "lm/column"),
    ]

    def node(kind: str, inputs: list[str], name: str, **attrs: object) -> "onnx.NodeProto":
        return helper.make_node(kind, inputs, [name], name=name, **attrs)

    scan.input[0] = x
    table = "model.embed_tokens.weight"
    nodes = [
        scan,
        node("Reshape", ["lm/rough", "lm/stretches"], "lm/rough_stretches"),
        node("ReduceMax", ["lm/rough_stretches"], "lm/stretch_best", axes=[-1], keepdims=0),
        helper.make_node("TopK", ["lm/stretch_best", "lm/best"], ["lm/best_value", "lm/best_at"], name="lm/best_at"),
        node("Cast", ["lm/best_at"], "lm/best_at_float", to=TensorProto.FLOAT),
        node("Mul", ["lm/best_at_float", "lm/stretch"], "lm/starts"),
        node("Reshape", ["lm/starts", "lm/column_of_starts"], "lm/starts_column"),
        node("Add", ["lm/starts_column", "lm/ramp"], "choices"),
        node("Reshape", ["choices", "lm/flat"], "lm/choices"),
        node("Cast", ["lm/choices"], "lm/choice_ids", to=TensorProto.INT64),
        node("Gather", [table, "lm/choice_ids"], "lm/rows"),
        node("Reshape", [x, "lm/column"], "lm/x_column"),
        node("MatMul", ["lm/rows", "lm/x_column"], "logits_fp16"),
        node("Reshape", ["logits_fp16", "lm/flat"], "lm/logits"),
    ]
    return nodes, consts


def big(model: "onnx.ModelProto", scan: "onnx.ModelProto", reads: bool = False) -> None:
    """Gives the hosted graph `seqlens` and `total` inputs in place of the mask, and a `wants` output.

    `seqlens` is the position of the last row and `total` one past it. `scan` is the LM head in 4 bits, from
    `rough`. With `reads` it also hands out what the head reads, and `next_ids` where `given` is set stand in for
    the big model's picks as the token after each row.
    """
    from onnx import TensorProto, helper

    g = model.graph
    mask = [n for n in g.node if "attn_mask_reformat" in n.name]
    gone = {o for n in mask for o in n.output}
    for node in g.node:
        if node.op_type == "GroupQueryAttention":
            node.input[5], node.input[6] = "seqlens", "total"
    whole = next(n for n in g.node if n.output[0] == "logits_fp16")
    drop = [*mask, whole, *[n for n in g.node if n.output[0] == whole.input[1]]]
    kept = [n for n in g.node if n not in drop and not (n.op_type == "Cast" and n.output[0] == "logits")]
    assert not any(x in gone or x == "attention_mask" for n in kept for x in n.input), "the mask feeds more than GQA"
    del g.node[:]
    g.node.extend(kept)
    vocab = next(i.dims[0] for i in g.initializer if i.name == "model.embed_tokens.weight")
    listed, list_consts = shortlist(whole.input[0], scan.graph.node[0], vocab)
    nodes, consts = first_max("lm/logits", "wants", BEST * STRETCH, "lm/choices", vocab)
    g.node.extend([*listed, *nodes])
    g.initializer.extend([*scan.graph.initializer, *list_consts, *consts])
    inputs = [i for i in g.input if i.name != "attention_mask"]
    del g.input[:]
    g.input.extend(inputs)
    g.input.extend(
        [
            helper.make_tensor_value_info("seqlens", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("total", TensorProto.INT32, []),
        ]
    )
    shape = ["batch_size", "sequence_length"]
    # The fp16 logits come out as they are, since a copy in fp32 would cost the run alone a kernel a pass.
    outputs = [o for o in g.output if o.name != "logits"]
    del g.output[:]
    g.output.extend(outputs)
    g.output.extend(
        [
            helper.make_tensor_value_info("logits_fp16", TensorProto.FLOAT16, [*shape, BEST * STRETCH, 1]),
            helper.make_tensor_value_info("choices", TensorProto.FLOAT, [*shape, BEST, STRETCH]),
            helper.make_tensor_value_info("wants", TensorProto.INT64, shape),
        ]
    )
    if not reads:
        return
    g.input.extend(
        [
            helper.make_tensor_value_info("given", TensorProto.BOOL, shape),
            helper.make_tensor_value_info("next_ids", TensorProto.INT64, shape),
        ]
    )
    g.node.extend(
        [
            helper.make_node(
                "Concat", [f"/model/layers.{i}/input_layernorm/output_3" for i in LAYERS], ["states"], axis=-1
            ),
            # On the prompt the token after a row is the next prompt token, and after a guess it is the big
            # model's pick, which is the guess itself wherever the guess was right.
            helper.make_node("Where", ["given", "next_ids", "wants"], ["follow_ids"]),
            helper.make_node("Gather", ["model.embed_tokens.weight", "follow_ids"], ["follow"]),
        ]
    )
    g.output.extend(
        [
            helper.make_tensor_value_info("states", TensorProto.FLOAT16, [*shape, 2048 * len(LAYERS)]),
            helper.make_tensor_value_info("follow", TensorProto.FLOAT16, [*shape, 2048]),
        ]
    )


def head(w: dict[str, "np.ndarray"]) -> "onnx.ModelProto":
    """The guessing layer over the rows the big model just read, with a cache laid out like the big model's."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    nodes, consts = [], []

    def op(kind: str, inputs: list[str], name: str = "", domain: str = "", **attrs: object) -> str:
        name = name or f"{kind}_{len(nodes)}"
        nodes.append(helper.make_node(kind, inputs, [name], name=name, domain=domain, **attrs))
        return name

    def const(name: str, value: "np.ndarray") -> str:
        consts.append(numpy_helper.from_array(value, name))
        return name

    def linear(x: str, key: str) -> str:
        return op("MatMul", [x, const(key, np.ascontiguousarray(w[f"{key}.weight"].T))])

    def norm(x: str, key: str) -> str:
        return op(
            "SimplifiedLayerNormalization", [x, const(key, w[f"{key}.weight"])], axis=-1, epsilon=1e-6, stash_type=1
        )

    def turned(x: str) -> str:
        return op("RotaryEmbedding", [x, "positions", "cos", "sin"], domain="com.microsoft", interleaved=0)

    angle = np.arange(4096)[:, None] / (1e6 ** (np.arange(0, 128, 2) / 128))[None]
    const("cos", np.cos(angle).astype(np.float16))
    const("sin", np.sin(angle).astype(np.float16))

    hidden = linear("states", "fc")
    x = op("Concat", [norm("embed", "midlayer.input_layernorm"), norm(hidden, "midlayer.hidden_norm")], axis=-1)
    queries = [
        turned(linear(x, "midlayer.self_attn.q_proj")),
        turned(linear(x, "midlayer.self_attn.k_proj")),
        linear(x, "midlayer.self_attn.v_proj"),
    ]
    nodes.append(
        helper.make_node(
            "GroupQueryAttention",
            [*queries, "k", "v", "seqlens", "total"],
            ["attended", "k_out", "v_out"],
            name="attention",
            domain="com.microsoft",
            num_heads=16,
            kv_num_heads=8,
        )
    )
    mid = op("Add", [hidden, linear("attended", "midlayer.self_attn.o_proj")])
    m = norm(mid, "midlayer.post_attention_layernorm")
    gate = linear(m, "midlayer.mlp.gate_proj")
    lit = op("Mul", [op("Mul", [gate, op("Sigmoid", [gate])]), linear(m, "midlayer.mlp.up_proj")])
    out = op("Add", [mid, linear(lit, "midlayer.mlp.down_proj")])
    picked, picked_consts = first_max(linear(norm(out, "norm"), "lm_head"), "pick", len(w["d2t"]))
    nodes.extend(picked)
    consts.extend(picked_consts)
    op("Gather", [const("ids", (np.arange(len(w["d2t"])) + w["d2t"]).astype(np.int64)), "pick"], "guesses")

    def info(name: str, shape: list[int | str], kind: int = TensorProto.FLOAT16) -> "onnx.ValueInfoProto":
        return helper.make_tensor_value_info(name, kind, shape)

    cache = [1, 8, "entries", 128]
    graph = helper.make_graph(
        nodes,
        "head",
        [
            info("states", [1, "rows", 2048 * len(LAYERS)]),
            info("embed", [1, "rows", 2048]),
            info("positions", [1, "rows"], TensorProto.INT64),
            info("seqlens", [1], TensorProto.INT32),
            info("total", [], TensorProto.INT32),
            info("k", cache),
            info("v", cache),
        ],
        [info("guesses", [1, "rows"], TensorProto.INT64), info("k_out", cache), info("v_out", cache)],
        consts,
    )
    opsets = [helper.make_opsetid("", 14), helper.make_opsetid("com.microsoft", 1)]
    return helper.make_model(graph, opset_imports=opsets, ir_version=7)


def joined(model: "onnx.ModelProto", step: "onnx.ModelProto") -> None:
    """Runs the quantized head `step` after the big graph `big(model, reads=True)` made, on what it reads."""
    from onnx import TensorProto, helper

    g = model.graph
    wired = {
        "states": "states",
        "embed": "follow",
        "positions": "position_ids",
        "seqlens": "seqlens",
        "total": "total",
        "k": "head_k",
        "v": "head_v",
    }
    renamed = {i.name: f"head/{i.name}" for i in step.graph.initializer}
    renamed |= {"guesses": "guesses", "k_out": "head_k_out", "v_out": "head_v_out"}
    for init in step.graph.initializer:
        init.name = renamed[init.name]
        g.initializer.append(init)
    for node in step.graph.node:
        node.name = f"head/{node.name}"
        node.input[:] = [wired.get(x) or renamed.get(x) or (x and f"head/{x}") for x in node.input]
        node.output[:] = [renamed.get(x) or f"head/{x}" for x in node.output]
        g.node.append(node)
    outputs = [o for o in g.output if o.name not in {"logits_fp16", "choices", "states", "follow"}]
    del g.output[:]
    g.output.extend(outputs)
    cache = [1, 8, "head_entries", 128]
    g.output.extend(
        [
            helper.make_tensor_value_info("guesses", TensorProto.INT64, ["batch_size", "sequence_length"]),
            helper.make_tensor_value_info("head_k_out", TensorProto.FLOAT16, cache),
            helper.make_tensor_value_info("head_v_out", TensorProto.FLOAT16, cache),
        ]
    )
    g.input.extend([helper.make_tensor_value_info(f"head_{kv}", TensorProto.FLOAT16, cache) for kv in "kv"])


def chained(model: "onnx.ModelProto") -> None:
    """Gives the pass graph `joined` made the next pass's inputs as outputs, each named `then_` and the input.

    The page copies them across on the GPU, so the next pass can start while this one's picks are still on their
    way back. A guess is kept if it is the big model's pick after the row before it and every guess before it was
    kept. The next pass starts one row past the last kept guess, with the pick after that row, the head's guess
    after the pick, and one more guess for each further row, which the head makes here from its own output.
    """
    import copy

    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    held = {i.name for i in g.initializer}
    steps = [n for n in g.node if n.name.startswith("head/")]
    fc = next(n for n in steps if "states" in n.input)
    own = next(n for n in steps if "head/norm" in n.input).input[0]  # the head's output, before its last norm
    cache = [1, 8, "head_entries", 128]

    def const(name: str, value: "np.ndarray") -> str:
        g.initializer.append(numpy_helper.from_array(value, f"then/{name}"))
        return f"then/{name}"

    def node(kind: str, inputs: list[str], out: str, **attrs: object) -> str:
        g.node.append(helper.make_node(kind, inputs, [out], name=out, **attrs))
        return out

    at = [const(f"at{j}", np.array([j], np.int64)) for j in range(ROWS)]
    flat, scalar = const("flat", np.array([1], np.int64)), const("scalar", np.array([], np.int64))
    one32 = const("one32", np.array([1], np.int32))
    kept = ""
    for j in reversed(range(1, ROWS)):
        guess = node("Gather", ["input_ids", at[j]], f"then/guess{j}", axis=1)
        hit = node("Equal", [guess, node("Gather", ["wants", at[j - 1]], f"then/want{j}", axis=1)], f"then/hit{j}")
        if kept:
            kept = node("Where", [hit, node("Add", [kept, at[1]], f"then/run{j}"), at[0]], f"then/kept{j}")
        else:
            kept = node("Cast", [hit], f"then/kept{j}", to=TensorProto.INT64)
    # On the prompt the rows are not guesses, and counting them all as kept puts the head's extra entries past the
    # pass's own rows, where the next pass writes over them.
    full = const("full", np.array([[ROWS - 1]], np.int64))
    kept = node("Gather", [node("Where", ["given", full, kept], "then/kept_rows"), at[0]], "then/kept", axis=1)
    row = node("Reshape", [kept, flat], "then/row")
    step = node("Add", [kept, at[1]], "then/step")
    node("Add", ["position_ids", step], "then_position_ids")
    node("Add", ["seqlens", node("Cast", [step], "then/step32", to=TensorProto.INT32)], "then_seqlens")
    node("Add", ["total", "then/step32"], "then_total")
    # Token ids go through floats here, which hold them exactly, as WebGPU's Concat returns other memory for int64.
    floats = node("Cast", ["guesses"], "then/guesses", to=TensorProto.FLOAT)
    ids = [
        node("Gather", ["wants/first", row], "then/pick", axis=1),
        node("Gather", [floats, row], "then/next1", axis=1),
    ]
    hidden = where = k = v = ""
    for j in range(2, ROWS):
        # The head once more, on its own output for the last row it read and the guess it just made. Its entry
        # goes one past that row's, where the next pass writes over it.
        hidden = hidden or node("Gather", [own, row], "then/hidden2", axis=1)
        if where:
            where = node("Add", [where, at[1]], f"then/where{j}")
        else:
            where = node("Gather", ["then_position_ids", at[0]], "then/where2", axis=1)
        where32 = node("Cast", [where], f"then/where32_{j}", to=TensorProto.INT32)
        token = node("Cast", [ids[-1]], f"then/token{j}", to=TensorProto.INT64)
        swaps = {
            fc.output[0]: hidden,
            "follow": node("Gather", ["model.embed_tokens.weight", token], f"then/embed{j}"),
            "position_ids": where,
            "seqlens": node("Reshape", [where32, flat], f"then/seqlens{j}"),
            "total": node("Reshape", [node("Add", [where32, one32], f"then/past{j}"), scalar], f"then/total{j}"),
            "head_k": k or "head_k_out",
            "head_v": v or "head_v_out",
            "guesses": f"then/guess_after{j}",
            "head_k_out": f"head_k_out{j}",
            "head_v_out": f"head_v_out{j}",
        }
        for was in steps:
            if was is fc:
                continue
            again = copy.deepcopy(was)
            again.name = f"again{j}/{was.name}"
            again.input[:] = [swaps.get(x) or (x if x in held or not x else f"again{j}/{x}") for x in was.input]
            again.output[:] = [swaps.get(x) or f"again{j}/{x}" for x in was.output]
            g.node.append(again)
        ids.append(node("Cast", [swaps["guesses"]], f"then/next{j}", to=TensorProto.FLOAT))
        hidden, k, v = f"again{j}/{own}", swaps["head_k_out"], swaps["head_v_out"]
        # The page gives these the same buffers as the head's cache, as attention has to write where it reads.
        g.output.extend([helper.make_tensor_value_info(name, TensorProto.FLOAT16, cache) for name in (k, v)])
    node("Cast", [node("Concat", ids, "then/ids", axis=1)], "then_input_ids", to=TensorProto.INT64)
    g.output.extend(
        [
            helper.make_tensor_value_info("then_input_ids", TensorProto.INT64, [1, ROWS]),
            helper.make_tensor_value_info("then_position_ids", TensorProto.INT64, [1, ROWS]),
            helper.make_tensor_value_info("then_seqlens", TensorProto.INT32, [1, 1]),
            helper.make_tensor_value_info("then_total", TensorProto.INT32, [1, 1]),
        ]
    )


def _empty(session: "ort.InferenceSession") -> dict[str, "np.ndarray"]:
    import numpy as np

    return {i.name: np.zeros((1, 8, ENTRIES, 128), np.float16) for i in session.get_inputs() if len(i.shape) == 4}


def _feed(
    session: "ort.InferenceSession",
    cache: dict[str, "np.ndarray"],
    ids: list[int],
    at: int,
    nexts: "Sequence[int | None]" = (),
) -> dict[str, "np.ndarray"]:
    """One pass over `ids` written at position `at`, and the tokens after them where known, as the page runs it."""
    import numpy as np

    feeds = {
        "input_ids": np.array([ids], np.int64),
        "position_ids": np.arange(at, at + len(ids), dtype=np.int64)[None],
        "seqlens": np.array([at + len(ids) - 1], np.int32),
        "total": np.array(at + len(ids), np.int32),
        **cache,
    }
    if "given" in [i.name for i in session.get_inputs()]:
        nexts = [*nexts, *[None] * (len(ids) - len(nexts))]
        feeds["given"] = np.array([[n is not None for n in nexts]])
        feeds["next_ids"] = np.array([[n or 0 for n in nexts]], np.int64)
    names = [o.name for o in session.get_outputs()]
    got = dict(zip(names, session.run(names, feeds), strict=True))
    for name in names:
        if name.startswith("present") or name.startswith("head_"):
            # The head's cache comes out once for each time it ran, and the last of them is the one to keep.
            cache[re.sub(r"_out\d*$", "", name.replace("present", "past_key_values"))] = got[name]
    return got


def read(session: "ort.InferenceSession", cache: dict[str, "np.ndarray"], prompt: list[int], rows: int) -> dict:
    """Reads the prompt `rows` tokens a pass, the last pass starting early to stay whole.

    Rows read twice are written over with the same values, as the cache is the same up to them.
    """
    starts = [*range(0, len(prompt) - rows, rows), len(prompt) - rows]
    for at in starts:
        nexts = [prompt[i + 1] if i + 1 < len(prompt) else None for i in range(at, at + rows)]
        got = _feed(session, cache, prompt[at : at + rows], at, nexts)
    return got


def guessed(
    session: "ort.InferenceSession", prompt: list[int], most: int, stop: set[int]
) -> list[tuple[list[int], int, int]]:
    """`eagle.guessed` over the pass graph, as the page runs it: (guesses, how many were kept, own token)."""
    cache = _empty(session)
    got = read(session, cache, prompt, ROWS)
    seq = [*prompt, int(got["wants"][0, -1])]
    # The pass that read the prompt has one guess to give, and it fills every guess row of the first pass.
    guesses = [int(got["guesses"][0, -1])] * (ROWS - 1)
    passes = [([], 0, seq[-1])]
    while seq[-1] not in stop and len(seq) - len(prompt) < most:
        got = _feed(session, cache, [seq[-1], *guesses], len(seq) - 1)
        wants = got["wants"][0].tolist()
        kept = accepted(guesses, wants)
        # The head guessed after every row, and the one after the last right row is next. The graph has made the
        # guesses after that one, and they are only in what it would read next.
        at = len(seq) + kept
        ids, *where = [
            got[f"then_{name}"].ravel().tolist() for name in ("input_ids", "position_ids", "seqlens", "total")
        ]
        assert ids[:2] == [wants[kept], int(got["guesses"][0, kept])], f"the graph would read {ids} next"
        assert where == [[*range(at, at + ROWS)], [at + ROWS - 1], [at + ROWS]], f"the graph would write at {where}"
        # A kept guess can end the answer, and nothing guessed after it counts.
        if (end := next((j for j, g in enumerate(guesses[:kept]) if g in stop), None)) is not None:
            guesses = guesses[:end]
            kept = end
        seq += [*guesses[:kept], wants[kept]]
        passes.append((guesses, kept, wants[kept]))
        guesses = ids[1:]
    return passes


def alone(session: "ort.InferenceSession", prompt: list[int], n: int) -> list[int]:
    cache = _empty(session)
    seq = [*prompt, int(read(session, cache, prompt, 1)["wants"][0, -1])]
    while len(seq) - len(prompt) < n:
        seq.append(int(_feed(session, cache, seq[-1:], len(seq) - 1)["wants"][0, -1]))
    return seq[len(prompt) :]


def export(out: Path) -> None:
    """Writes big.onnx, pass.onnx and head.onnx.data to `out`/onnx, then writes draft.json's prompt both ways with them.

    The hosted weights are downloaded into the same folder, since both graphs read them from beside themselves.
    The 4-bit LM head goes to scan.data there, which is not uploaded, as the page makes the same bytes from the
    hosted weights (docs/js/scan.js).
    """
    import copy

    import numpy as np
    import onnx
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from onnx.external_data_helper import set_external_data
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer

    hosted = Path(hf_hub_download(**BIG, filename="onnx/model_q4f16.onnx_data", local_dir=out))
    graph = onnx.load(hf_hub_download(**BIG, filename="onnx/model_q4f16.onnx"), load_external_data=False)
    files = [hosted.with_name(name) for name in ("big.onnx", "pass.onnx", "head.onnx.data", "scan.data")]
    alone_graph = copy.deepcopy(graph)
    table = next(i for i in graph.graph.initializer if i.name == "model.embed_tokens.weight")
    at = int(next(e.value for e in table.external_data if e.key == "offset"))
    assert at == 0, f"the page takes the embedding table from the start of the weights, and it is at {at}"
    scan = rough(np.fromfile(hosted, np.float16, count=int(np.prod(table.dims)), offset=at).reshape(table.dims))
    held = {i.name: i for i in scan.graph.initializer}
    with files[3].open("wb") as data:
        for name in scan.graph.node[0].input[1:]:
            raw = held[name].raw_data
            set_external_data(held[name], files[3].name, data.tell(), len(raw))
            held[name].ClearField("raw_data")
            data.write(raw)
    big(graph, copy.deepcopy(scan), reads=True)
    quantizer = MatMulNBitsQuantizer(head(load_file(hf_hub_download(**HEAD))), bits=4, block_size=32, is_symmetric=True)
    quantizer.process()
    joined(graph, quantizer.model.model)
    chained(graph)
    files[2].unlink(missing_ok=True)  # onnx will not write over an old one
    onnx.save(graph, str(files[1]), save_as_external_data=True, location=files[2].name)
    big(alone_graph, scan)
    onnx.save(alone_graph, str(files[0]))

    run = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in files[:2]]
    tok = AutoTokenizer.from_pretrained(TARGET)
    trace = json.loads(DRAFT.read_text(encoding="utf-8"))
    chat = tok.apply_chat_template(
        [{"role": "user", "content": trace["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = tok(chat).input_ids
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    passes = guessed(run[1], prompt, MOST, stop)
    ours = [t for guesses, kept, own in passes for t in [*guesses[:kept], own]]
    same = alone(run[0], prompt, len(ours)) == ours
    size = sum(path.stat().st_size for path in files[:3]) / 1e6
    print(
        f"{files[0].parent}: {size:.0f} MB. {len(ours)} tokens in {len(passes)} passes at {ROWS - 1} guesses each, "
        f"{len(ours) / len(passes):.2f} a pass, and {'the same text' if same else 'not the same text'} as alone"
    )
