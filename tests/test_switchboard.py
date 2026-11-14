import json
import shutil
import subprocess
from pathlib import Path

import pytest

from switchboard import draft, quant, route, spell
from switchboard.svg import Timeline

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"


@pytest.mark.parametrize(
    ("guesses", "wants", "kept"),
    [([1, 2, 3], [1, 2, 3, 4], 3), ([1, 2, 3], [1, 9, 3, 4], 1), ([5, 2], [1, 2, 3], 0), ([], [7], 0)],
)
def test_accepted_stops_at_the_first_miss(guesses, wants, kept):
    assert draft.accepted(guesses, wants) == kept


@pytest.mark.parametrize(
    ("title", "answer", "new"),
    [
        (
            "Jensen Huang resigns",
            "Jensen Huang resigns from his position as CEO of Alibaba.  \nAlibaba's board accepted it.",
            ["CEO", "Alibaba", "Alibaba's"],
        ),
        (
            "Jensen Huang resigns",
            "- Why did Jensen Huang resign from his position?  \n- What impact will his resignation have?",
            [],
        ),
        (
            "A false smoke alarm almost drowned Jane Street's only NYC datacenter",
            "1. How did the alarm go off in 2024?\n2. Is Jane Street moving out of NYC?\n3. Google asked why.",
            ["2024", "Google"],
        ),
        (
            "Nvidia’s chief steps down. Jensen Huang led it for 33 years, since 1993.",
            "The story is about Jensen Huang leaving Nvidia's top job. He led it from 1993. It ran for 33 years.",
            [],
        ),
        (
            "Ggml v1.0.0, a tensor library",
            "Tensor library for machine learning has released version 1.0.0. Additionally, Vulkan got faster.",
            ["Vulkan"],
        ),
    ],
)
def test_invented_finds_only_made_up_names(title, answer, new):
    assert draft.invented(title, answer) == new


def test_opening_is_a_pages_first_real_paragraphs():
    first = "The chief executive told staff on Thursday that he would leave at the end of the year. " * 2
    page = (
        '<html><head><meta property="og:description" content="A short note that says what the page is about, '
        'in a sentence."><style>p { color: red; }</style></head><body><p>By A. Writer</p><script>var p = 1;</script>'
        f"<p>{first}</p><p>Menu</p><p>A second <b>paragraph</b> with enough words in it to be worth keeping.</p></body>"
    )
    assert route.opening(page) == (
        "A short note that says what the page is about, in a sentence. "
        f"{first.strip()} A second paragraph with enough words in it to be worth keeping."
    )
    assert route.opening("<p>Too little to sum up.</p>") == ""


def test_layout_wraps_and_drops_the_leading_space():
    placed = draft.layout(["Hello", " world", " again"], 0, 0, 12 * 11)
    assert [(p.y, p.text) for p in placed] == [(0, "Hello"), (0, " world"), (draft.LINE_H, "again")]
    assert placed[2].x == 0


def test_timeline_keyframes_cover_the_whole_loop():
    tl = Timeline(10.0)
    assert tl.show("a", 2.5, 5.0) == "a"
    css = tl.css()
    assert "25.00%,49.99%{opacity:1}" in css
    assert "animation:a 10.0s linear infinite" in css


def test_reduced_motion_shows_what_the_loop_ends_on():
    tl = Timeline(10.0)
    tl.show("gone", 1.0, 4.0)
    tl.show("kept", 2.0, 9.0, lit=0.3, out=1.0)
    tl.show("eased", 5.0, 9.0, fade=1.0)
    css = tl.css(still=True)
    assert "@keyframes kept{0%,19.99%{opacity:0}20.00%,80.00%{opacity:0.3}90.00%,100%{opacity:0}}" in css
    assert "@keyframes eased{0%,50.00%{opacity:0}60.00%,80.00%{opacity:1}90.00%,100%{opacity:0}}" in css
    # Only what is still showing when the loop ends stays up, each at its own strength.
    assert css.endswith("@media (prefers-reduced-motion:reduce){.kept{opacity:0.3}.eased{opacity:1}}")


@pytest.mark.parametrize("name", ["route", "draft"])
def test_saved_traces_render(name):
    path = DATA / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not traced yet")
    svg = (route if name == "route" else draft).render(json.loads(path.read_text(encoding="utf-8")))
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_route_trace_is_consistent():
    t = json.loads((DATA / "route.json").read_text(encoding="utf-8"))
    for run in t["runs"]:
        for layer in run["layers"]:
            assert len(layer["experts"]) == len(run["tokens"])
            assert all(len(set(row)) == t["k"] for row in layer["experts"])
            assert all(abs(sum(w) - 1) < 1e-3 for w in layer["weights"])


def test_spell_puts_a_split_character_on_the_token_that_finishes_it():
    rocket = "\N{ROCKET}"
    raw = rocket.encode()
    assert spell([b"Launch", b" " + raw[:2], raw[2:3], raw[3:]]) == ["Launch", r" \xF0\x9F", r"\x9A", rocket]
    assert spell(["café".encode(), b" ok"]) == ["café", " ok"]
    assert spell([b"\xff", b"ok"]) == ["", "ok"]
    # Half an emoji the model never finishes, then plain text.
    assert spell([b" " + raw[:2], b" sky"]) == [r" \xF0\x9F", " sky"]


def test_saved_traces_have_no_replacement_characters():
    for name in ["route", "draft", "quant", "lens"]:
        assert "\N{REPLACEMENT CHARACTER}" not in (DATA / f"{name}.json").read_text(encoding="utf-8")


def test_rounding_keeps_each_groups_ends_and_errs_by_half_a_step_at_most():
    torch = pytest.importorskip("torch")
    w = torch.linspace(-1, 1, 256).reshape(2, 128)
    out = quant.rounded(w, quant.Format("int4", 4, 128))
    assert [r.unique().numel() for r in out] == [16, 16]
    for got, want in [(out.amin(-1), w.amin(-1)), (out.amax(-1), w.amax(-1))]:
        assert torch.allclose(got, want, atol=1e-6)
    assert (out - w).abs().max() <= (w[0].max() - w[0].min()) / 15 / 2 + 1e-6
    assert torch.equal(quant.rounded(w, quant.FORMATS[0]), w)


def test_lens_trace_has_every_layer_at_every_position_likeliest_first():
    t = json.loads((DATA / "lens.json").read_text(encoding="utf-8"))
    for run in t["runs"]:
        assert len(run["layers"]) == 28
        for layer in run["layers"]:
            assert len(layer) == len(run["tokens"])
            for top in layer:
                probs = [p for _, p in top]
                assert probs == sorted(probs, reverse=True) and sum(probs) <= 1.0001


def test_the_page_quantizes_the_lm_head_as_onnxruntime_does(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    if not shutil.which("node"):
        pytest.skip("needs node to run docs/js/scan.js")
    from switchboard import head

    # Rows of very small weights and a row of zeros, where a scale is too small for fp16 to hold.
    rng = np.random.default_rng(0)
    table = (rng.standard_normal((96, 64)) * rng.choice([1e-7, 1e-5, 0.02, 3.0], (96, 1))).astype(np.float16)
    table[5] = 0
    model = head.rough(table)
    held = {i.name: i.raw_data for i in model.graph.initializer}
    want = b"".join(held[name] for name in model.graph.node[0].input[1:])
    (tmp_path / "table").write_bytes(table.tobytes())
    script = f"""
        import {{ scan }} from {(DATA.parent / "js/scan.js").as_uri()!r}
        import {{ readFileSync }} from "node:fs"
        const bytes = readFileSync("table")
        const table = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2)
        process.stdout.write(Buffer.from(await scan(table, 64)).toString("hex"))
    """
    got = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert bytes.fromhex(got.stdout) == want
