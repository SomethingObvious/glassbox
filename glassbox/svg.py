"""Drawing primitives shared by every banner: palette, text and CSS keyframes."""

from html import escape

W = 1660
BG, LINE, INK, MUTED = "#0b0f17", "#1f2836", "#e6edf3", "#8b949e"
AMBER, CYAN, GREEN, RED = "#ffd65a", "#58c4dc", "#56d364", "#ff7b72"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"


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
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{c}>{escape(s)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, rx: float = 3, cls: str = "", extra: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<rect{c} x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{extra}/>'


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
    """A self-contained animated SVG that also respects reduced motion."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
        f'role="img"><title>{escape(title)}</title><style>text{{white-space:pre}}{style}\n'
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}</style>"
        f'<rect width="{W}" height="{height}" rx="18" fill="{BG}"/>{"".join(body)}</svg>'
    )
