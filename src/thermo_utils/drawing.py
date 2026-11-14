"""Terminal drawings of cycles: braille P-v / T-s plots and equipment sketches."""

import math

from rich.text import Text

from thermo_utils.units import fmt

_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))

BRAYTON = r"""
            q_in
       ┌──────────────┐
  2 ┌──┤  combustor   ├──┐ 3
    │  └──────────────┘  │
 ┌──┴───────┐      ┌─────┴────┐
 │compressor├──────┤ turbine  ├──▶ w_net
 └──┬───────┘      └─────┬────┘
    │  ┌──────────────┐  │
  1 └──┤    cooler    ├──┘ 4
       └──────────────┘
            q_out
"""

RANKINE = r"""
            q_in
       ┌──────────────┐
  2 ┌──┤    boiler    ├──┐ 3
    │  └──────────────┘  │
 ┌──┴───┐          ┌─────┴────┐
 │ pump │◀─ w_p    │ turbine  ├──▶ w_t
 └──┬───┘          └─────┬────┘
    │  ┌──────────────┐  │
  1 └──┤  condenser   ├──┘ 4
       └──────────────┘
            q_out
"""

EQUIPMENT = {"brayton": BRAYTON, "rankine": RANKINE}


def plot(d, width=44, height=14, accent="rgb(255,150,40)"):
    """Braille plot of the cycle with numbered states; returns a list of Text lines."""
    pts = [p for c in d["curves"] for p in c] + list(d["points"]) + list(d["dome"] or [])
    ylog = d.get("ylog", False)
    fy = math.log10 if ylog else (lambda y: y)
    xs, ys = [p[0] for p in pts], [fy(p[1]) for p in pts]
    x0, x1 = _pad(min(xs), max(xs))
    y0, y1 = _pad(min(ys), max(ys), not ylog)
    W, H = width * 2, height * 4
    bits = [[0] * width for _ in range(height)]
    style = [[None] * width for _ in range(height)]

    def px(x, y):
        return (x - x0) / (x1 - x0) * (W - 1), (1 - (fy(y) - y0) / (y1 - y0)) * (H - 1)

    def line(points, st):
        for (xa, ya), (xb, yb) in zip(points, points[1:]):
            (ax, ay), (bx, by) = px(xa, ya), px(xb, yb)
            n = max(1, int(max(abs(bx - ax), abs(by - ay)) * 2))
            for i in range(n + 1):
                cx, cy = round(ax + (bx - ax) * i / n), round(ay + (by - ay) * i / n)
                if 0 <= cx < W and 0 <= cy < H:
                    bits[cy // 4][cx // 2] |= _BITS[cy % 4][cx % 2]
                    if style[cy // 4][cx // 2] != accent:
                        style[cy // 4][cx // 2] = st

    if d["dome"]:
        line(d["dome"], "dim")
    for c in d["curves"]:
        line(c, accent)

    chars = [[chr(0x2800 + b) if b else " " for b in r] for r in bits]
    cells = [px(x, y) for x, y in d["points"]]
    cells = [(min(width - 1, int(a // 2)), min(height - 1, int(b // 4))) for a, b in cells]
    mx = sum(c[0] for c in cells) / len(cells)
    my = sum(c[1] for c in cells) / len(cells)
    for i, (cx, cy) in enumerate(cells, 1):
        chars[cy][cx], style[cy][cx] = "●", accent
    for i, (cx, cy) in enumerate(cells, 1):
        dx = 0 if abs(cx - mx) < 1 else (1 if cx > mx else -1)
        dy = 0 if abs(cy - my) < 0.5 else (1 if cy > my else -1)
        for lx, ly in ((cx + dx, cy + dy), (cx + (dx or 1), cy), (cx - (dx or 1), cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= lx < width and 0 <= ly < height and chars[ly][lx] not in "●1234":
                chars[ly][lx], style[ly][lx] = str(i), "bold"
                break

    (xname, xkind), (yname, ykind) = d["axes"]
    ytop, ybot = (_tick(10 ** y1, ykind), _tick(10 ** y0, ykind)) if ylog else (_tick(y1, ykind), _tick(y0, ykind))
    gutter = max(len(ytop), len(ybot), len(yname)) + 1
    out = [Text(f"{yname:>{gutter - 1}}", style="bold").append("  (log scale)" if ylog else "", style="dim")]
    for r in range(height):
        label = ytop if r == 0 else ybot if r == height - 1 else ""
        t = Text(f"{label:>{gutter - 1}} ", style="dim")
        t.append("┤" if label else "│", style="dim")
        for c in range(width):
            t.append(chars[r][c], style=style[r][c])
        out.append(t)
    out.append(Text(" " * gutter + "└" + "─" * width, style="dim"))
    left, right = _tick(x0, xkind), _tick(x1, xkind)
    gap = max(1, width - len(left) - len(right) - len(xname) - 2)
    t = Text(" " * (gutter + 1) + left, style="dim")
    t.append(" " * (gap // 2 + 1) + xname, style="bold")
    t.append(" " * (gap - gap // 2 + 1) + right, style="dim")
    out.append(t)
    return out


def equipment(name, accent="rgb(255,150,40)"):
    lines = []
    for raw in EQUIPMENT[name].strip("\n").splitlines():
        t = Text()
        for ch in raw:
            t.append(ch, style=f"bold {accent}" if ch.isdigit() else "dim" if ch in "─│┌┐└┘├┤┬┴▶◀" else None)
        lines.append(t)
    return lines


def _pad(lo, hi, clamp=True):
    span = hi - lo or abs(hi) or 1.0
    low = lo - 0.06 * span
    return (0.0 if clamp and lo >= 0 > low else low), hi + 0.06 * span


def _tick(value, kind):
    text = fmt(value, kind, digits=4)
    if text.startswith("-0 ") or text.startswith("-0."):
        text = text[1:]
    return text if not math.isnan(value) else "-"
