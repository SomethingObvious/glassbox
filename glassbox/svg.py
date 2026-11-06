"""Drawing primitives shared by every banner: palette, text and CSS keyframes."""

from html import escape

W = 1660
# The page's tokens. An SVG shown as an image can't load web fonts, so the stacks fall back to system faces.
SHEET, INK, SOFT, FAINT, LOST = "var(--sheet)", "var(--ink)", "var(--soft)", "var(--faint)", "var(--lost)"
TOKENS = (
    ":root{--sheet:#fdfdfb;--ink:#121212;--soft:#555552;--faint:#d6d6d1;--lost:#c4161c}"
    "@media (prefers-color-scheme:dark){:root{--sheet:#131312;--ink:#ecebe4;--soft:#a9a8a0;--faint:#3a3a37;"
    "--lost:#ff6b6b}}"
)
SERIF = "Libre Caslon Text,Iowan Old Style,Georgia,serif"
SANS = "Public Sans,Helvetica Neue,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def paint(fill: str = "none", stroke: str = "", width: float = 1.5, dash: str = "") -> str:
    """A style attribute, since CSS variables don't work in SVG presentation attributes."""
    s = f"fill:{fill}"
    if stroke:
        s += f";stroke:{stroke};stroke-width:{width}"
    if dash:
        s += f";stroke-dasharray:{dash}"
    return f' style="{s}"'


def text(
    x: float,
    y: float,
    s: str,
    size: float = 20,
    fill: str = INK,
    anchor: str = "start",
    weight: int = 400,
    font: str = SANS,
    cls: str = "",
) -> str:
    c = f' class="{cls}"' if cls else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}"{paint(fill)} '
        f'text-anchor="{anchor}" font-weight="{weight}"{c}>{escape(s)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = "none",
    cls: str = "",
    extra: str = "",
    stroke: str = "",
    width: float = 1.5,
    dash: str = "",
) -> str:
    c = f' class="{cls}"' if cls else ""
    return (
        f'<rect{c} x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"{paint(fill, stroke, width, dash)}{extra}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = INK, width: float = 2) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"{paint("none", stroke, width)}/>'


def callout(x: float, y: float, swatch: str, numeral: int, label: str) -> tuple[str, float]:
    """A legend entry as the page draws it: a swatch, a bold reference numeral, then the label. Returns its width."""
    body = swatch + text(x + 30, y, str(numeral), 17, weight=700) + text(x + 68, y, label, 17)
    return body, 68 + 8.6 * len(label)


class Timeline:
    """Collects one opacity animation per element class, all on a shared loop."""

    def __init__(self, period: float) -> None:
        self.period = period
        self.rules: list[str] = []

    def show(self, name: str, on: float, off: float, lit: float = 1.0) -> str:
        """Makes class `name` visible from second `on` to second `off` of every loop, and returns the name."""
        a, b = 100 * on / self.period, 100 * off / self.period
        self.rules.append(
            f"@keyframes {name}{{0%,{max(a - 0.01, 0):.2f}%{{opacity:0}}{a:.2f}%,{max(b - 0.01, a):.2f}%"
            f"{{opacity:{lit:g}}}{b:.2f}%,100%{{opacity:0}}}}"
            f".{name}{{animation:{name} {self.period}s linear infinite}}"
        )
        return name

    def css(self, base: str = "") -> str:
        return "\n".join([base, *self.rules])


def document(title: str, height: int, style: str, body: list[str]) -> str:
    """A self-contained animated SVG plate that follows the viewer's colour scheme and reduced motion."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        f'role="img"><title>{escape(title)}</title><style>{TOKENS}text{{white-space:pre}}{style}\n'
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}</style>"
        f"{rect(0, 0, W, height, SHEET)}{line(60, 20, W - 60, 20)}{line(60, height - 14, W - 60, height - 14)}"
        f"{''.join(body)}</svg>"
    )
