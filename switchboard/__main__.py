"""Command line: `python -m switchboard {route,draft}` redraws a banner, `--trace` reruns its model first.

`--headline` runs route or draft on today's top Hacker News title. Draft takes the title from the route.json
that route wrote beside it, so both figures show the same story.

`python -m switchboard quant` and `lens` run Qwen3-0.6B for Figures 3 and 4, which have no banners. `routes`
exports the model Figure 2 runs in the browser to build/routes, and `head` exports Figure 1's to build/head.
"""

import argparse
import datetime
import json
from pathlib import Path

from . import draft, head, lens, quant, route, routes

ROOT = Path(__file__).resolve().parent.parent
DATA, ASSETS = ROOT / "docs" / "data", ROOT / "assets"


def save(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")
    print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="switchboard", description=__doc__)
    parser.add_argument("banner", choices=["route", "draft", "quant", "lens", "routes", "head"])
    parser.add_argument("--trace", action="store_true", help="run the model again instead of reading the saved trace")
    parser.add_argument("--headline", type=Path, metavar="SVG", help="draw today's top Hacker News title here")
    args = parser.parse_args()
    if args.headline and args.banner not in {"route", "draft"}:
        parser.error("--headline only works with route and draft")
    # The clock's own day, which the daily workflow sets to Toronto's.
    now = datetime.datetime.now()
    day = f"{now.day} {now:%b %Y}"

    if args.banner in {"routes", "head"}:
        {"routes": routes, "head": head}[args.banner].export(ROOT / "build" / args.banner)
    elif args.banner in {"quant", "lens"}:
        module = {"quant": quant, "lens": lens}[args.banner]
        (DATA / f"{args.banner}.json").write_text(json.dumps(module.trace()), encoding="utf-8")
    elif args.headline and args.banner == "draft":
        routed = args.headline.with_name("route.json")
        if routed.exists():
            told = json.loads(routed.read_text(encoding="utf-8"))
            title, link, day = told["runs"][0]["text"], told["headline"]["link"], told["headline"]["day"]
        else:
            title, link = route.headline()
        t = draft.trace(draft.ask(title))
        new = draft.invented(title, "".join(p for s in t["steps"] for p in [*s["drafted"][: s["kept"]], s["target"]]))
        if new:
            # The model hasn't read the story, so the day's run is the saved prompt instead.
            print(f"Left the headline out, as the answer names {', '.join(new)}")
            t = draft.trace()
        else:
            # The page shows the day and links the story beside the run.
            t["headline"] = {"title": title, "link": link, "day": day}
        save(args.headline, draft.render(t))
        args.headline.with_suffix(".json").write_text(json.dumps(t), encoding="utf-8")
    elif args.headline:
        subtitle = (
            f"Today's top Hacker News headline ({day}), routed live through IBM Granite 3.0 1B-A400M. "
            "Each token goes to 8 of 32 experts in every one of 24 layers."
        )
        title, link = route.headline()
        t = route.trace([title])
        t["headline"] = {"day": day, "link": link}
        save(args.headline, route.render(t, subtitle))
        # The page reads this to put the headline first in Figure 2.
        args.headline.with_suffix(".json").write_text(json.dumps(t), encoding="utf-8")
    else:
        module = {"route": route, "draft": draft}[args.banner]
        data = DATA / f"{args.banner}.json"
        if args.trace:
            data.parent.mkdir(parents=True, exist_ok=True)
            data.write_text(json.dumps(module.trace()), encoding="utf-8")
        save(ASSETS / f"{args.banner}.svg", module.render(json.loads(data.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
