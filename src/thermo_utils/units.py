import re
from dataclasses import dataclass

R = 8.314462618

TEMPERATURE = "temperature"
PRESSURE = "pressure"
MOLAR_VOLUME = "molar_volume"
MOLAR_ENERGY = "molar_energy"
MOLAR_ENTROPY = "molar_entropy"
MASS_FLOW = "mass_flow"
MOLAR_FLOW = "molar_flow"
POWER = "power"
DIMENSIONLESS = "dimensionless"

_FACTORS = {
    PRESSURE: {
        "pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5, "mbar": 100.0,
        "atm": 101325.0, "psi": 6894.757293168, "psia": 6894.757293168,
        "mmhg": 133.322387415, "torr": 133.322368421, "kgf/cm2": 98066.5,
    },
    MOLAR_VOLUME: {
        "m3/mol": 1.0, "cm3/mol": 1e-6, "l/mol": 1e-3, "dm3/mol": 1e-3,
        "m3/kmol": 1e-3, "ml/mol": 1e-6,
    },
    MOLAR_ENERGY: {
        "j/mol": 1.0, "kj/mol": 1e3, "j/kmol": 1e-3, "kj/kmol": 1.0,
        "cal/mol": 4.184, "kcal/mol": 4184.0,
    },
    MOLAR_ENTROPY: {
        "j/mol/k": 1.0, "j/(mol.k)": 1.0, "j/molk": 1.0, "kj/mol/k": 1e3,
        "cal/mol/k": 4.184, "kj/kmol/k": 1.0,
    },
    MASS_FLOW: {
        "kg/s": 1.0, "kg/h": 1 / 3600, "kg/min": 1 / 60, "t/h": 1000 / 3600,
        "g/s": 1e-3, "lb/h": 0.45359237 / 3600,
    },
    MOLAR_FLOW: {
        "mol/s": 1.0, "kmol/s": 1e3, "kmol/h": 1000 / 3600, "mol/h": 1 / 3600,
        "mol/min": 1 / 60, "kmol/min": 1000 / 60,
    },
    POWER: {"w": 1.0, "kw": 1e3, "mw": 1e6, "hp": 745.699872, "cv": 735.49875},
    DIMENSIONLESS: {"": 1.0},
}

_ALIASES = {
    "°c": "c", "ºc": "c", "degc": "c", "celsius": "c",
    "°f": "f", "ºf": "f", "degf": "f",
    "°r": "r", "ºr": "r", "kelvin": "k",
    "m³/mol": "m3/mol", "cm³/mol": "cm3/mol",
    "j/(mol k)": "j/mol/k", "j/mol.k": "j/mol/k", "j/(mol·k)": "j/mol/k",
}

DEFAULT_INPUT_UNITS = {
    TEMPERATURE: "K", PRESSURE: "bar", MOLAR_VOLUME: "cm3/mol",
    MOLAR_ENERGY: "J/mol", MOLAR_ENTROPY: "J/mol/K", MASS_FLOW: "kg/s",
    MOLAR_FLOW: "mol/s", POWER: "kW", DIMENSIONLESS: "",
}


@dataclass
class DisplayUnits:
    temperature: str = "K"
    pressure: str = "bar"
    molar_volume: str = "cm3/mol"
    molar_energy: str = "kJ/mol"
    molar_entropy: str = "J/mol/K"
    mass_flow: str = "kg/s"
    molar_flow: str = "mol/s"
    power: str = "kW"

    def unit_for(self, kind):
        return getattr(self, kind, "")


display = DisplayUnits()


def _norm(unit):
    u = unit.strip().lower().replace(" ", "")
    return _ALIASES.get(u, u)


def units_of(kind):
    if kind == TEMPERATURE:
        return ["K", "C", "F", "R"]
    return list(_FACTORS[kind])


def to_si(value, unit, kind):
    u = _norm(unit)
    if kind == TEMPERATURE:
        if u in ("k", ""):
            return value
        if u == "c":
            return value + 273.15
        if u == "f":
            return (value - 32) * 5 / 9 + 273.15
        if u == "r":
            return value * 5 / 9
        raise ValueError(f"Unknown temperature unit '{unit}'")
    table = _FACTORS[kind]
    if u not in table:
        raise ValueError(f"Unknown unit '{unit}'. Options: {', '.join(table)}")
    return value * table[u]


def from_si(value, unit, kind):
    u = _norm(unit)
    if kind == TEMPERATURE:
        if u in ("k", ""):
            return value
        if u == "c":
            return value - 273.15
        if u == "f":
            return (value - 273.15) * 9 / 5 + 32
        if u == "r":
            return value * 9 / 5
        raise ValueError(f"Unknown temperature unit '{unit}'")
    return value / _FACTORS[kind][u]


_NUMBER = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


def parse_quantity(text, kind, default_unit=None):
    """Parse '25 C', '10atm' or '300' into an SI value."""
    match = _NUMBER.match(text.replace(",", "."))
    if not match:
        raise ValueError(f"Could not read a number from '{text}'")
    value = float(match.group(1))
    unit = match.group(2) or default_unit or DEFAULT_INPUT_UNITS[kind]
    return to_si(value, unit, kind)


def fmt(value, kind=DIMENSIONLESS, unit=None, digits=6):
    if value is None:
        return "-"
    if kind == DIMENSIONLESS:
        return f"{value:.{digits}g}"
    unit = unit or display.unit_for(kind)
    shown = from_si(value, unit, kind)
    label = {"c": "°C", "f": "°F", "r": "R"}.get(_norm(unit), unit)
    return f"{shown:.{digits}g} {label}"
