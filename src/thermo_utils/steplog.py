"""Step-by-step log of a calculation, shown by 'Show details'."""

from contextlib import contextmanager
from functools import wraps

from thermo_utils import units
from thermo_utils.units import DIMENSIONLESS, fmt

_LOG = [None]
_MUTE = [0]


class Q:
    """A value formatted in display units when the log is shown."""

    __slots__ = ("value", "kind", "digits")

    def __init__(self, value, kind=None, digits=6):
        self.value, self.kind, self.digits = value, kind, digits

    def __str__(self):
        if self.value is None:
            return "-"
        if isinstance(self.value, str):
            return self.value
        return fmt(self.value, self.kind or DIMENSIONLESS, digits=self.digits)


def q(value, kind=None, digits=6):
    return Q(value, kind, digits)


def start():
    _LOG[0] = []


def take():
    out, _LOG[0] = _LOG[0] or [], None
    return out


def on():
    return _LOG[0] is not None and not _MUTE[0]


@contextmanager
def muted():
    _MUTE[0] += 1
    try:
        yield
    finally:
        _MUTE[0] -= 1


def quiet(fn):
    """Run fn without logging (diagrams, curves, comparisons)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with muted():
            return fn(*args, **kwargs)
    return wrapper


def step(title, *args):
    if on():
        _LOG[0].append(("step", title, args))


def line(text="", *args):
    """Text with {} placeholders filled by values (Q, numbers or text)."""
    if on():
        _LOG[0].append(("line", text, args))


def val(label, value, kind=None, how=""):
    line(f"{label} = {{}}" + (f"   ({how})" if how else ""), Q(value, kind))


def eq(label, formula, value, kind=None):
    """label = formula = value."""
    line(f"{label} = {formula} = {{}}", Q(value, kind))


def table(headers, rows):
    if on():
        _LOG[0].append(("table", list(headers), [list(r) for r in rows]))


def _text(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _fill(text, args):
    if args:
        try:
            text = text.format(*map(_text, args))
        except (IndexError, KeyError, ValueError):
            text = text + " " + " ".join(map(_text, args))
    kilo = units._PREFIX[0]
    return text if kilo is None else units.apply_prefix(text, kilo)


def render(entries):
    """Log entries as plain text lines."""
    out, n = [], 0
    for e in entries:
        if e[0] == "step":
            n += 1
            out += ["", f"[{n}] {_fill(e[1], e[2])}"]
        elif e[0] == "line":
            text = _fill(e[1], e[2])
            out.append("    " + text if text else "")
        else:
            headers, rows = e[1], [[_fill(_text(c), ()) for c in r] for r in e[2]]
            widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
                      for i, h in enumerate(headers)]
            out.append("    " + "  ".join(f"{h:>{w}}" for h, w in zip(headers, widths)))
            out += ["    " + "  ".join(f"{c:>{w}}" for c, w in zip(r, widths)) for r in rows]
    return out


def K(t):
    """Absolute temperature for formulas, whatever the display unit."""
    return f"{t:.6g} K"
