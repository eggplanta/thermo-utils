import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ACCENT = "#e8711a"
ALT = "#2b6cb0"
DIM = "#8a8f98"
INK = "#1f2933"
KEEP = 100
STYLES = {"accent": ACCENT, "alt": ALT, "dim": DIM}

matplotlib.rcParams.update({
    "figure.figsize": (7.2, 5.0), "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#4a5260", "axes.linewidth": 0.9, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#d9dde3", "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.prop_cycle": matplotlib.cycler(color=[ACCENT, ALT, "#2a9d8f", "#9b5de5", "#c0392b", "#6c757d",
                                                "#d4a017", "#17becf", "#8c564b", "#e377c2"]),
    "lines.linewidth": 1.8, "lines.markersize": 6,
    "xtick.color": "#4a5260", "ytick.color": "#4a5260", "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#d9dde3", "legend.fontsize": 9.5,
    "mathtext.default": "regular",
})


def cache_dir():
    if os.environ.get("THERMO_UTILS_PLOTS"):
        return Path(os.environ["THERMO_UTILS_PLOTS"])
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "thermo-utils"


def _path(name, suffix):
    folder = cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    old = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime)
    for p in old[:max(0, len(old) - KEEP)]:
        p.unlink(missing_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "result"
    return folder / f"{safe}_{datetime.now():%Y%m%d-%H%M%S}{suffix}"


# Plot files

def save(fig_factory, name, footer=""):
    """PNG of the figure plus a CSV with its curve data; returns both paths."""
    import matplotlib.pyplot as plt
    from thermo_utils import __version__
    fig = fig_factory()
    fig.text(0.995, 0.005, " · ".join(x for x in (footer, f"thermo-utils {__version__}") if x),
             ha="right", va="bottom", fontsize=8, color=DIM)
    png = _path(name, ".png")
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.15)
    data = png.with_name(png.stem + "_data.csv")
    _figure_csv(fig, data)
    plt.close(fig)
    return png.resolve(), data.resolve()


def _figure_csv(fig, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plot", "series", "x quantity", "x", "y quantity", "y"])
        for ax in fig.axes:
            sup = fig._suptitle.get_text() if fig._suptitle else ""
            title = ax.get_title() or ax.get_title(loc="left") or sup
            xq, yq = ax.get_xlabel(), ax.get_ylabel()
            if not (xq or yq):
                continue
            n = 0
            series = [(ln.get_gid() or ln.get_label(), ln.get_xdata(), ln.get_ydata()) for ln in ax.get_lines()]
            for c in ax.collections:
                if hasattr(c, "get_offsets") and len(c.get_offsets()) and type(c).__name__ == "PathCollection":
                    off = c.get_offsets()
                    series.append((c.get_label(), off[:, 0], off[:, 1]))
            for label, xs, ys in series:
                n += 1
                if not label.strip("_") or re.match(r"_(child|line|collection)\d*$", label):
                    label = f"series {n}"
                label = label.lstrip("_")
                for x, y in zip(xs, ys):
                    w.writerow([title, label, xq, _num(x), yq, _num(y)])


def _num(v):
    try:
        return f"{float(v):.8g}"
    except (TypeError, ValueError):
        return v


# Result tables

_VALUE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def _split(text):
    text = str(text).replace("←", "").strip()
    m = _VALUE.match(text)
    if m and not m.group(2).startswith((",", "(", "-", "→")):
        return m.group(1), m.group(2)
    return text, ""


_KV = re.compile(r"^(.+?)\s*=\s*(.+)$")
_KN = re.compile(r"^([A-Za-z][\w ]*?)\s+([-+]?(?:\d|\.\d).*)$")


def _expand(label, text):
    """Rows (quantity, value, unit); 'x = 0.4   y = 0.6' becomes one row per value."""
    text = str(text).replace("←", "").strip()
    parts = [p.strip().strip("()").strip() for p in re.split(r"\s{2,}", text) if p.strip()]
    if len(parts) < 2:
        return [(label, *_split(text))]
    rows = []
    for p in parts:
        m = _KV.match(p) or _KN.match(p)
        rows.append((f"{m.group(1)} ({label})", *_split(m.group(2))) if m else (label, *_split(p)))
    return rows


def export_csv(res, name):
    """Every table, state box and section of a result as rows: section, item, quantity, value, unit."""
    path = _path(name, ".csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "item", "quantity", "value", "unit"])
        for heading, headers, rows in res.tables:
            for cells in rows:
                item = cells[0][0]
                for (text, _), head in zip(cells[1:], headers[1:]):
                    quantity, _, unit = head.partition("\n")
                    value, u = _split(text)
                    w.writerow([heading, item, quantity or "description", value, unit or u])
        if res.pair:
            for state, rows in (("State 1", res.pair[0]), ("State 2", res.pair[2])):
                for r in rows:
                    w.writerows(["States", state, *x] for x in _expand(r[0], r[1]))
        for heading, rows in res.sections:
            for r in rows:
                w.writerows([heading, "", *x] for x in _expand(r[0], r[1]))
    return path.resolve()


# Diagrams (from the terminal diagram data)

def figure(panels, title):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.0), squeeze=False)
    for ax, d in zip(axes[0], panels):
        _panel(ax, d, len(panels) > 1)
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def _panel(ax, d, titled=True):
    from thermo_utils.units import display, from_si, label
    (xname, xkind), (yname, ykind) = d["axes"]
    xu, yu = display.unit_for(xkind), display.unit_for(ykind)

    def conv(pts):
        return [from_si(x, xu, xkind) for x, _ in pts], [from_si(y, yu, ykind) for _, y in pts]

    names = {}
    for style, text in d.get("legend", []):
        names.setdefault(style, text)
    dome = d.get("dome")
    if dome and d.get("dome_fine"):
        try:
            dome = d["dome_fine"]() or dome
        except Exception:
            pass
    if dome:
        xs, ys = conv(dome)
        ax.plot(xs, ys, color=DIM, lw=1.3, label=names.pop("dim", "saturation dome"), zorder=2)
        if not (d.get("ylog") or d.get("xlog")):
            ax.fill(xs, ys, color=DIM, alpha=0.07, lw=0, zorder=1)
    curves = [c if isinstance(c, dict) else {"pts": c, "style": "accent"} for c in d.get("curves", [])]
    lines = []
    for c in curves:
        style = c.get("style", "accent")
        xs, ys = conv(c["pts"])
        dashed = style.endswith("dashed")
        legend = names.pop(style, None)
        ln = ax.plot(xs, ys, color=STYLES[style.split("-")[0]], ls=(0, (5, 3)) if dashed else "-",
                     lw=1.6 if dashed else 2.2, label=legend or "_", zorder=3)[0]
        ln.set_gid(c.get("name") or legend or f"curve {len(lines) + 1}")
        lines.append(ln)
    if d.get("fill"):
        xs, ys = conv([p for c in curves for p in c["pts"]])
        ax.fill(xs, ys, color=ACCENT, alpha=0.13, lw=0, label=d["fill"], zorder=1)
    if d.get("xlog"):
        ax.set_xscale("log")
    if d.get("ylog"):
        ax.set_yscale("log")
    ax.autoscale_view()
    for axis, on in ((ax.xaxis, d.get("xlog")), (ax.yaxis, d.get("ylog"))):
        if on:
            _plain_log(axis)
    pts = d.get("points", [])
    if pts:
        xs, ys = conv(pts)
        ax.scatter(xs, ys, s=46, color="white", edgecolors=INK, linewidths=1.4, zorder=5,
                   label=names.pop("point", "_states"))
    ax.autoscale_view()
    _zoom(ax, d, conv)
    if d.get("arrows"):
        for ln in lines:
            _arrow(ax, ln)
    _labels(ax, list(zip(*conv(pts))) if pts else [], d.get("labels") or [str(i + 1) for i in range(len(pts))])
    ax.set_xlabel(f"{xname} ({label(xu)})" if xu else xname)
    ax.set_ylabel(f"{yname} ({label(yu)})" if yu else yname)
    if titled and d.get("title"):
        ax.set_title(d["title"], loc="left")
    handles = [h for h in ax.get_legend_handles_labels()[0]]
    if handles:
        ax.legend(loc="best")


def _zoom(ax, d, conv):
    """Zoom on the process when it is tiny next to the dome (pumps, liquids)."""
    if not d.get("dome") or d.get("xlog") or d.get("ylog"):
        return
    curves = [c if isinstance(c, dict) else {"pts": c} for c in d.get("curves", [])]
    pts = [p for c in curves for p in c["pts"]] + list(d.get("points", []))
    if len(set(pts)) < 2:
        return
    xs, ys = conv(pts)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    wx, wy = max(xs) - min(xs), max(ys) - min(ys)
    if wx > 0.05 * (x1 - x0) or wy > 0.05 * (y1 - y0):
        return
    px = max(wx, 0.01 * (x1 - x0)) * 0.6
    py = max(wy, 0.01 * (y1 - y0)) * 0.6
    ax.set_xlim(min(xs) - px, max(xs) + px)
    ax.set_ylim(min(ys) - py, max(ys) + py)
    ax.set_title("zoom on the process (the saturation dome is off scale)", loc="right", fontsize=9,
                 fontweight="normal", color=DIM)


def _plain_log(axis):
    import math
    from matplotlib import ticker
    lo, hi = sorted(axis.get_view_interval())
    plain = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    axis.set_major_formatter(plain)
    if lo > 0 and math.log10(hi / lo) < 1.5:
        axis.set_minor_locator(ticker.LogLocator(subs=(2, 3, 5)))
        axis.set_minor_formatter(plain)
    else:
        axis.set_minor_formatter(ticker.NullFormatter())


def _arrow(ax, line):
    import numpy as np
    xy = ax.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
    seg = np.hypot(*np.diff(xy, axis=0).T)
    if len(seg) == 0 or seg.sum() < 25:
        return
    cum = np.concatenate([[0], np.cumsum(seg)])
    half = cum[-1] / 2
    i = min(np.searchsorted(cum, half) - 1, len(seg) - 1)
    i = max(i, 0)
    f = (half - cum[i]) / seg[i] if seg[i] else 0
    mid = xy[i] + f * (xy[i + 1] - xy[i])
    unit = (xy[i + 1] - xy[i]) / (seg[i] or 1)
    inv = ax.transData.inverted()
    ax.annotate("", xy=inv.transform(mid + 6 * unit), xytext=inv.transform(mid - 6 * unit),
                arrowprops=dict(arrowstyle="-|>", color=line.get_color(), lw=1.4, mutation_scale=15), zorder=4)


def _labels(ax, pts, labels):
    import numpy as np
    if not pts:
        return
    xy = ax.transData.transform(np.array(pts))
    center = xy.mean(axis=0)
    placed = []
    for (x, y), p, text in zip(pts, xy, labels):
        if not text:
            continue
        v = p - center
        n = np.hypot(*v)
        v = v / n if n > 1 else np.array([0.7, 0.7])
        for turn in (0, 90, -90, 180, 45, -45, 135, -135):
            a = np.radians(turn)
            w = np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])
            spot = p + 16 * w
            if all(np.hypot(*(spot - q)) > 18 for q in placed):
                break
        placed.append(spot)
        ax.annotate(text, (x, y), xytext=(11 * w[0], 11 * w[1]), textcoords="offset points",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=6)


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
