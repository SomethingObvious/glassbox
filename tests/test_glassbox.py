import json
from pathlib import Path

import pytest

from glassbox import draft, route, waves
from glassbox.svg import Timeline

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


@pytest.mark.parametrize("name", ["route", "draft"])
def test_saved_traces_render(name):
    path = DATA / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not traced yet")
    svg = (route if name == "route" else draft).render(json.loads(path.read_text()))
    assert svg.startswith("<svg") and svg.endswith("</svg>")


def test_route_trace_is_consistent():
    t = json.loads((DATA / "route.json").read_text())
    for run in t["runs"]:
        for layer in run["layers"]:
            assert len(layer["experts"]) == len(run["tokens"])
            assert all(len(set(row)) == t["k"] for row in layer["experts"])
            assert all(abs(sum(w) - 1) < 1e-3 for w in layer["weights"])
