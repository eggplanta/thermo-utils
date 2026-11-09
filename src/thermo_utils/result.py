from dataclasses import dataclass, field
from typing import Callable

from thermo_utils.units import fmt


@dataclass
class Result:
    title: str
    sections: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    plot: Callable | None = None
    plot_name: str = "plot"
    data: dict = field(default_factory=dict)
    model: str | None = None

    def section(self, heading, rows=None):
        self.sections.append((heading, list(rows or [])))
        return self.sections[-1][1]


def row(label, value, kind=None, unit=None, digits=6):
    if isinstance(value, str):
        return (label, value)
    if kind is None:
        return (label, fmt(value, digits=digits))
    return (label, fmt(value, kind, unit, digits))
