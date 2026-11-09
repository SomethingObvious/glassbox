"""Command line: `python -m glassbox {route,draft,waves}` redraws a banner, `--trace` reruns its model first."""

import argparse
import datetime
import json
from pathlib import Path

from . import draft, route, waves

ROOT = Path(__file__).resolve().parent.parent
DATA, ASSETS = ROOT / "docs" / "data", ROOT / "assets"


def save(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")
    print(f"{path} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="glassbox", description=__doc__)
    parser.add_argument("banner", choices=["route", "draft", "waves"])
    parser.add_argument("--trace", action="store_true", help="run the model again instead of reading the saved trace")
    parser.add_argument("--headline", type=Path, metavar="SVG", help="route: draw today's top Hacker News title here")
    args = parser.parse_args()
    if args.headline and args.banner != "route":
        parser.error("--headline only works with route")

    if args.banner == "waves":
        save(ASSETS / "waves.svg", waves.render())
    elif args.headline:
        now = datetime.datetime.now(datetime.UTC)
        day = f"{now.day} {now:%b %Y}"
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
