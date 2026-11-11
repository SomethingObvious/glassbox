import json
from pathlib import Path

import pytest

from switchboard import draft, quant, route, spell, waves
from switchboard.svg import Timeline

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"


@pytest.mark.parametrize(
    ("plan", "tiles", "count", "last"),
    [
        (waves.Plan(m=512, n=14336, tile=128, sms=132), 448, 4, 52),
        (waves.Plan(m=128, n=132 * 128, tile=128, sms=132), 132, 1, 132),
        (waves.Plan(m=100, n=100, tile=128, sms=132), 1, 1, 1),
    ],
)
def test_waves_count_partial_tiles_and_the_last_wave(plan, tiles, count, last):
    assert (plan.tiles, plan.waves, plan.busy(plan.waves - 1)) == (tiles, count, last)
    assert plan.efficiency == tiles / (count * plan.sms)


@pytest.mark.parametrize(
    ("guesses", "wants", "kept"),
    [([1, 2, 3], [1, 2, 3, 4], 3), ([1, 2, 3], [1, 9, 3, 4], 1), ([5, 2], [1, 2, 3], 0), ([], [7], 0)],
)
def test_accepted_stops_at_the_first_miss(guesses, wants, kept):
    assert draft.accepted(guesses, wants) == kept


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
