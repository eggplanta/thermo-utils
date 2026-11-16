"""Terminal drawings: braille diagrams and equipment sketches."""

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

SHAPES = r"""
compressor
          w_in
           │
     ┌─────▼──────┐
 1 ──┤ compressor ├──▶ 2
     └────────────┘
turbine
           w_out
             ▲
     ┌───────┴────┐
 1 ──┤  turbine   ├──▶ 2
     └────────────┘
pump
          w_in
           │
     ┌─────▼──────┐
 1 ──┤    pump    ├──▶ 2
     └────────────┘
valve
 1 ──────▶ ▷◁ ──────▶ 2
           valve
heater
          q_in
           │
     ┌─────▼──────┐
 1 ──┤   heater   ├──▶ 2
     └────────────┘
cooler
          q_out
           ▲
     ┌─────┴──────┐
 1 ──┤   cooler   ├──▶ 2
     └────────────┘
piston
     ┌──────────────┬─┐
     │  gas  1 → 2  │ ├────
     └──────────────┴─┘
"""


def _shapes():
    out, name = {}, None
    for raw in SHAPES.strip("\n").splitlines():
        if raw and not raw[0].isspace():
            name = raw
            out[name] = ""
        else:
            out[name] += raw + "\n"
    return out


EQUIPMENT = {"brayton": BRAYTON, "rankine": RANKINE, **_shapes()}
ALT = "rgb(90,170,255)"


def _curve(c):
    return c if isinstance(c, dict) else {"pts": c, "style": "accent"}


def plot(d, width=44, height=14, accent="rgb(255,150,40)"):
    """Braille plot with styled curves and labelled points; returns a list of Text lines."""
    curves = [_curve(c) for c in d.get("curves", [])]
    if d.get("dome"):
        curves.insert(0, {"pts": d["dome"], "style": "dim"})
    points = list(d.get("points", []))
    labels = d.get("labels") or [str(i) for i in range(1, len(points) + 1)]
    ylog, xlog = d.get("ylog", False), d.get("xlog", False)
    fy = math.log10 if ylog else (lambda y: y)
    fx = math.log10 if xlog else (lambda x: x)
    pts = [p for c in curves for p in c["pts"]] + points
    xs, ys = [fx(p[0]) for p in pts], [fy(p[1]) for p in pts]
    x0, x1 = _pad(min(xs), max(xs), not xlog)
    y0, y1 = _pad(min(ys), max(ys), not ylog)
    if d.get("xlim"):
        x0, x1 = map(fx, d["xlim"])
    if d.get("ylim"):
        y0, y1 = map(fy, d["ylim"])
    W, H = width * 2, height * 4
    bits = [[0] * width for _ in range(height)]
    style = [[None] * width for _ in range(height)]
    colors = {"accent": accent, "alt": ALT, "dim": "dim"}

    def px(x, y):
        return (fx(x) - x0) / (x1 - x0) * (W - 1), (1 - (fy(y) - y0) / (y1 - y0)) * (H - 1)

    def line(points, st, dashed):
        dist = 0.0
        for (xa, ya), (xb, yb) in zip(points, points[1:]):
            (ax, ay), (bx, by) = px(xa, ya), px(xb, yb)
            span = max(abs(bx - ax), abs(by - ay))
            n = max(1, int(span * 2))
            for i in range(n + 1):
                if dashed and int((dist + span * i / n) / 3) % 2:
                    continue
                cx, cy = round(ax + (bx - ax) * i / n), round(ay + (by - ay) * i / n)
                if 0 <= cx < W and 0 <= cy < H:
                    bits[cy // 4][cx // 2] |= _BITS[cy % 4][cx % 2]
                    if style[cy // 4][cx // 2] in (None, "dim"):
                        style[cy // 4][cx // 2] = st
            dist += span

    for c in curves:
        name = c.get("style", "accent")
        line(c["pts"], colors[name.split("-")[0]], name.endswith("dashed"))

    chars = [[chr(0x2800 + b) if b else " " for b in r] for r in bits]
    cells = [px(x, y) for x, y in points]
    cells = [(max(0, min(width - 1, int(a // 2))), max(0, min(height - 1, int(b // 4)))) for a, b in cells]
    taken = set(cells)
    for cx, cy in cells:
        chars[cy][cx], style[cy][cx] = "●", accent
    if cells:
        mx = sum(c[0] for c in cells) / len(cells)
        my = sum(c[1] for c in cells) / len(cells)
    for text, (cx, cy) in zip(labels, cells):
        if not text:
            continue
        n = len(text)
        dx = 0 if abs(cx - mx) < 1 else (1 if cx > mx else -1)
        dy = 0 if abs(cy - my) < 0.5 else (1 if cy > my else -1)
        for lx, ly in ((cx + (1 if dx >= 0 else -n), cy + dy), (cx + 1, cy), (cx - n, cy),
                       (cx + 1, cy - 1), (cx - n, cy - 1), (cx + 1, cy + 1), (cx - n, cy + 1)):
            spot = [(lx + k, ly) for k in range(n)]
            if all(0 <= a < width and 0 <= b < height and (a, b) not in taken for a, b in spot):
                for k, (a, b) in enumerate(spot):
                    chars[b][a], style[b][a] = text[k], "bold"
                taken.update(spot)
                break

    (xname, xkind), (yname, ykind) = d["axes"]
    tick_y = (lambda v: _tick(10 ** v, ykind)) if ylog else (lambda v: _tick(v, ykind))
    tick_x = (lambda v: _tick(10 ** v, xkind)) if xlog else (lambda v: _tick(v, xkind))
    ytop, ybot = tick_y(y1), tick_y(y0)
    gutter = max(len(ytop), len(ybot), len(yname)) + 1
    out = [Text(d["title"], style="bold")] if d.get("title") else []
    scales = [s for s, on in (("log y", ylog), ("log x", xlog)) if on]
    out.append(Text(f"{yname:>{gutter - 1}}", style="bold").append(
        f"  ({', '.join(scales)})" if scales else "", style="dim"))
    for r in range(height):
        label = ytop if r == 0 else ybot if r == height - 1 else ""
        t = Text(f"{label:>{gutter - 1}} ", style="dim")
        t.append("┤" if label else "│", style="dim")
        for c in range(width):
            t.append(chars[r][c], style=style[r][c])
        out.append(t)
    out.append(Text(" " * gutter + "└" + "─" * width, style="dim"))
    left, right = tick_x(x0), tick_x(x1)
    gap = max(1, width - len(left) - len(right) - len(xname) - 2)
    t = Text(" " * (gutter + 1) + left, style="dim")
    t.append(" " * (gap // 2 + 1) + xname, style="bold")
    t.append(" " * (gap - gap // 2 + 1) + right, style="dim")
    out.append(t)
    if d.get("legend"):
        t = Text(" " * (gutter + 1))
        for name, text in d["legend"]:
            mark = "┅┅" if name.endswith("dashed") else "●" if name == "point" else "──"
            t.append(mark, style=colors.get(name.split("-")[0], accent))
            t.append(f" {text}   ", style="dim")
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
    if text.startswith("-") and float(text.split()[0]) == 0:
        text = text[1:]
    return text if not math.isnan(value) else "-"
