from dataclasses import dataclass, field
from typing import Callable

from thermo_utils.units import ENERGY_UNITS, apply_prefix, choose_prefix, fmt, set_prefix


@dataclass
class Result:
    title: str
    sections: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    plot: Callable | None = None
    plot_name: str = "plot"
    data: dict = field(default_factory=dict)
    model: str | None = None
    pair: tuple | None = None
    equations: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    diagram: dict | None = None
    log: list = field(default_factory=list)

    def section(self, heading, rows=None):
        self.sections.append((heading, list(rows or [])))
        return self.sections[-1][1]


def _unit(header):
    return header.split("\n")[-1]


def _cell(text, header):
    return f"{text} {_unit(header)}" if _unit(header) in ENERGY_UNITS else text


def harmonize(res):
    """Use one energy prefix (J or kJ) across the whole result."""
    def strings():
        for heading, rows in res.sections:
            yield heading
            for r in rows:
                yield from (x for x in r if isinstance(x, str))
        for heading, headers, rows in res.tables:
            yield heading
            yield from headers
            for cells in rows:
                yield from (_cell(t, h) for (t, _), h in zip(cells, headers))
        for rows in (res.pair[0], res.pair[2]) if res.pair else ():
            for r in rows:
                yield from (x for x in r if isinstance(x, str))
        yield from res.notes

    kilo = choose_prefix(strings())
    if kilo is None:
        return

    def fix(x):
        return apply_prefix(x, kilo) if isinstance(x, str) else x

    res.sections = [(fix(h), [tuple(map(fix, r)) for r in rows]) for h, rows in res.sections]
    def fix_cell(t, h):
        return apply_prefix(_cell(t, h), kilo).split(" ")[0] if _unit(h) in ENERGY_UNITS else fix(t)

    def fix_header(h):
        if _unit(h) not in ENERGY_UNITS:
            return fix(h)
        head = h.rsplit("\n", 1)[0] + "\n" if "\n" in h else ""
        return head + apply_prefix(f"1 {_unit(h)}", kilo).split(" ", 1)[1]

    res.tables = [(fix(h), [fix_header(x) for x in headers],
                   [[(fix_cell(t, x), g) for (t, g), x in zip(cells, headers)] for cells in rows])
                  for h, headers, rows in res.tables]
    if res.pair:
        res.pair = ([tuple(map(fix, r)) for r in res.pair[0]], res.pair[1],
                    [tuple(map(fix, r)) for r in res.pair[2]])
    res.notes = [fix(n) for n in res.notes]
    set_prefix(kilo)


def row(label, value, kind=None, unit=None, digits=6):
    if isinstance(value, str):
        return (label, value)
    if kind is None:
        return (label, fmt(value, digits=digits))
    return (label, fmt(value, kind, unit, digits))
