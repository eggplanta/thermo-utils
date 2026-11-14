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
LENGTH = "length"
SURFACE_TENSION = "surface_tension"
CONCENTRATION = "concentration"
SPECIFIC_ENERGY = "specific_energy"
SPECIFIC_ENTROPY = "specific_entropy"
SPECIFIC_VOLUME = "specific_volume"
VOLUME = "volume"
MASS = "mass"
ENERGY = "energy"

BTU = 1055.05585262
LB = 0.45359237
LBMOL = 453.59237
FT3 = 0.028316846592

_FACTORS = {
    PRESSURE: {
        "pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5, "mbar": 100.0,
        "atm": 101325.0, "psi": 6894.757293168, "psia": 6894.757293168,
        "mmhg": 133.322387415, "torr": 133.322368421, "kgf/cm2": 98066.5,
    },
    MOLAR_VOLUME: {
        "m3/mol": 1.0, "cm3/mol": 1e-6, "l/mol": 1e-3, "dm3/mol": 1e-3,
        "m3/kmol": 1e-3, "ml/mol": 1e-6, "ft3/lbmol": FT3 / LBMOL,
    },
    MOLAR_ENERGY: {
        "j/mol": 1.0, "kj/mol": 1e3, "j/kmol": 1e-3, "kj/kmol": 1.0,
        "cal/mol": 4.184, "kcal/mol": 4184.0, "btu/lbmol": BTU / LBMOL,
    },
    MOLAR_ENTROPY: {
        "j/(mol·k)": 1.0, "kj/(mol·k)": 1e3, "cal/(mol·k)": 4.184, "kj/(kmol·k)": 1.0,
        "btu/(lbmol·r)": BTU / LBMOL * 1.8,
    },
    MASS_FLOW: {
        "kg/s": 1.0, "kg/h": 1 / 3600, "kg/min": 1 / 60, "t/h": 1000 / 3600,
        "g/s": 1e-3, "lb/h": 0.45359237 / 3600,
    },
    MOLAR_FLOW: {
        "mol/s": 1.0, "kmol/s": 1e3, "kmol/h": 1000 / 3600, "mol/h": 1 / 3600,
        "mol/min": 1 / 60, "kmol/min": 1000 / 60,
    },
    POWER: {"w": 1.0, "kw": 1e3, "mw": 1e6, "hp": 745.699872, "cv": 735.49875, "btu/h": BTU / 3600},
    LENGTH: {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "in": 0.0254},
    SURFACE_TENSION: {"n/m": 1.0, "mn/m": 1e-3, "dyn/cm": 1e-3, "j/m2": 1.0, "mj/m2": 1e-3},
    CONCENTRATION: {"mol/l": 1e3, "mmol/l": 1.0, "mol/m3": 1.0, "m": 1e3, "mm": 1.0, "mol/dm3": 1e3},
    DIMENSIONLESS: {"": 1.0},
}

# Mass-based units: SI per kg, converted with the molar mass of the substance
_MASS_FACTORS = {
    MOLAR_VOLUME: {"m3/kg": 1.0, "l/kg": 1e-3, "cm3/g": 1e-3, "ft3/lb": FT3 / LB},
    MOLAR_ENERGY: {"j/kg": 1.0, "kj/kg": 1e3, "kcal/kg": 4184.0, "j/g": 1e3, "cal/g": 4184.0, "btu/lb": BTU / LB},
    MOLAR_ENTROPY: {
        "j/(kg·k)": 1.0, "kj/(kg·k)": 1e3, "kcal/(kg·k)": 4184.0, "j/(g·k)": 1e3, "cal/(g·k)": 4184.0,
        "btu/(lb·r)": BTU / LB * 1.8,
    },
}
_MOLAR_MASS = [None]

_SPECIFIC = {SPECIFIC_ENERGY: MOLAR_ENERGY, SPECIFIC_ENTROPY: MOLAR_ENTROPY, SPECIFIC_VOLUME: MOLAR_VOLUME}
for _kind, _molar in _SPECIFIC.items():
    _FACTORS[_kind] = _MASS_FACTORS[_molar]
_FACTORS[VOLUME] = {"m3": 1.0, "l": 1e-3, "dm3": 1e-3, "cm3": 1e-6, "ml": 1e-6, "ft3": FT3, "in3": 1.6387064e-5}
_FACTORS[MASS] = {"kg": 1.0, "g": 1e-3, "lb": LB}
_FACTORS[ENERGY] = {"j": 1.0, "kj": 1e3, "mj": 1e6, "btu": BTU, "kcal": 4184.0, "cal": 4.184}


def set_molar_mass(mw):
    """Molar mass in g/mol used by mass-based units (None for mixtures)."""
    _MOLAR_MASS[0] = mw


def _per_kg_to_molar(kind, u):
    mw = _MOLAR_MASS[0]
    if mw is None:
        raise ValueError(f"'{label(u)}' is per mass: it needs one substance to convert with the molar mass.")
    return _MASS_FACTORS[kind][u] * mw / 1000


_ALIASES = {
    "°c": "c", "ºc": "c", "degc": "c", "celsius": "c",
    "°f": "f", "ºf": "f", "degf": "f",
    "°r": "r", "ºr": "r", "kelvin": "k",
    "m³/mol": "m3/mol", "cm³/mol": "cm3/mol", "dm³/mol": "dm3/mol", "m³/kmol": "m3/kmol",
    "kgf/cm²": "kgf/cm2", "j/m²": "j/m2", "mj/m²": "mj/m2", "mol/m³": "mol/m3", "mol/dm³": "mol/dm3",
}
_ALIASES.update({"m³/kg": "m3/kg", "cm³/g": "cm3/g", "ft³/lb": "ft3/lb", "ft³/lbmol": "ft3/lbmol",
                 "m³": "m3", "dm³": "dm3", "cm³": "cm3", "ft³": "ft3", "in³": "in3", "cc": "cm3"})
for _key in [k for t in (_FACTORS[MOLAR_ENTROPY], _MASS_FACTORS[MOLAR_ENTROPY]) for k in t]:
    _e, _n, _t = re.match(r"(.+)/\((.+)·(.)\)", _key).groups()
    for _form in ("{e}/{n}/{t}", "{e}/{n}{t}", "{e}/({n}{t})", "{e}/({n}.{t})", "{e}/{n}.{t}", "{e}/{n}·{t}",
                  "{e}/({n}*{t})", "{e}/({n}-{t})", "{e}/{n}-{t}"):
        _ALIASES[_form.format(e=_e, n=_n, t=_t)] = _key

DEFAULT_INPUT_UNITS = {
    TEMPERATURE: "K", PRESSURE: "bar", MOLAR_VOLUME: "cm3/mol",
    MOLAR_ENERGY: "J/mol", MOLAR_ENTROPY: "J/(mol·K)", MASS_FLOW: "kg/s",
    MOLAR_FLOW: "mol/s", POWER: "kW", DIMENSIONLESS: "",
    LENGTH: "m", SURFACE_TENSION: "mN/m", CONCENTRATION: "mol/L",
    SPECIFIC_ENERGY: "kJ/kg", SPECIFIC_ENTROPY: "kJ/(kg·K)", SPECIFIC_VOLUME: "m3/kg",
    VOLUME: "L", MASS: "kg", ENERGY: "kJ",
}


@dataclass
class DisplayUnits:
    temperature: str = "K"
    pressure: str = "bar"
    molar_volume: str = "cm3/mol"
    molar_energy: str = "kJ/mol"
    molar_entropy: str = "J/(mol·K)"
    mass_flow: str = "kg/s"
    molar_flow: str = "mol/s"
    power: str = "kW"
    length: str = "m"
    surface_tension: str = "mN/m"
    concentration: str = "mol/L"
    specific_energy: str = "kJ/kg"
    specific_entropy: str = "kJ/(kg·K)"
    specific_volume: str = "m³/kg"
    volume: str = "L"
    mass: str = "kg"
    energy: str = "kJ"

    def unit_for(self, kind):
        return getattr(self, kind, "")

    def adopt(self, kind, unit, force=False, name=None):
        """Show results in the unit the user typed (first one per kind)."""
        if name:
            TYPED[name] = label(unit)
        if hasattr(self, kind) and (force or kind not in _ADOPTED):
            setattr(self, kind, label(unit))
            _ADOPTED.add(kind)
            basis = _MASS_BASIS.get(_norm(unit))
            if kind in _SPECIFIC and basis:
                basis = {k: basis[m] for k, m in _SPECIFIC.items()}
            for other, u in (basis or {}).items():
                if other != kind and other not in _ADOPTED:
                    setattr(self, other, label(u))


_BASES = [
    {MOLAR_VOLUME: "m3/kg", MOLAR_ENERGY: "kj/kg", MOLAR_ENTROPY: "kj/(kg·k)"},
    {MOLAR_VOLUME: "cm3/g", MOLAR_ENERGY: "j/g", MOLAR_ENTROPY: "j/(g·k)"},
    {MOLAR_VOLUME: "cm3/g", MOLAR_ENERGY: "cal/g", MOLAR_ENTROPY: "cal/(g·k)"},
    {MOLAR_VOLUME: "ft3/lb", MOLAR_ENERGY: "btu/lb", MOLAR_ENTROPY: "btu/(lb·r)"},
    {MOLAR_VOLUME: "ft3/lbmol", MOLAR_ENERGY: "btu/lbmol", MOLAR_ENTROPY: "btu/(lbmol·r)"},
]
_MASS_BASIS = {u: b for table in _MASS_FACTORS.values() for u in table for b in _BASES[:4] if u in b.values()}
_MASS_BASIS.update({u: _BASES[0] for table in _MASS_FACTORS.values() for u in table if u not in _MASS_BASIS})
_MASS_BASIS.update({u: _BASES[4] for u in _BASES[4].values()})


_ADOPTED = set()
TYPED = {}
display = DisplayUnits()


_LABELS = {
    "k": "K", "c": "°C", "f": "°F", "r": "R",
    "pa": "Pa", "kpa": "kPa", "mpa": "MPa", "bar": "bar", "mbar": "mbar", "atm": "atm", "psi": "psi",
    "psia": "psia", "mmhg": "mmHg", "torr": "torr", "kgf/cm2": "kgf/cm²",
    "m3/mol": "m³/mol", "cm3/mol": "cm³/mol", "l/mol": "L/mol", "dm3/mol": "dm³/mol", "m3/kmol": "m³/kmol",
    "ml/mol": "mL/mol",
    "j/mol": "J/mol", "kj/mol": "kJ/mol", "j/kmol": "J/kmol", "kj/kmol": "kJ/kmol", "cal/mol": "cal/mol",
    "kcal/mol": "kcal/mol",
    "j/(mol·k)": "J/(mol·K)", "kj/(mol·k)": "kJ/(mol·K)", "cal/(mol·k)": "cal/(mol·K)", "kj/(kmol·k)": "kJ/(kmol·K)",
    "m3/kg": "m³/kg", "cm3/g": "cm³/g", "l/kg": "L/kg", "ft3/lb": "ft³/lb",
    "j/kg": "J/kg", "kj/kg": "kJ/kg", "kcal/kg": "kcal/kg", "btu/lb": "Btu/lb",
    "j/(kg·k)": "J/(kg·K)", "kj/(kg·k)": "kJ/(kg·K)", "kcal/(kg·k)": "kcal/(kg·K)", "btu/(lb·r)": "Btu/(lb·R)",
    "j/g": "J/g", "cal/g": "cal/g", "j/(g·k)": "J/(g·K)", "cal/(g·k)": "cal/(g·K)",
    "btu/lbmol": "Btu/lbmol", "btu/(lbmol·r)": "Btu/(lbmol·R)", "ft3/lbmol": "ft³/lbmol", "btu/h": "Btu/h",
    "kg/s": "kg/s", "kg/h": "kg/h", "kg/min": "kg/min", "t/h": "t/h", "g/s": "g/s", "lb/h": "lb/h",
    "mol/s": "mol/s", "kmol/s": "kmol/s", "kmol/h": "kmol/h", "mol/h": "mol/h", "mol/min": "mol/min",
    "kmol/min": "kmol/min",
    "w": "W", "kw": "kW", "mw": "MW", "hp": "hp", "cv": "cv",
    "m": "m", "cm": "cm", "mm": "mm", "um": "µm", "µm": "µm", "nm": "nm", "in": "in",
    "n/m": "N/m", "mn/m": "mN/m", "dyn/cm": "dyn/cm", "j/m2": "J/m²", "mj/m2": "mJ/m²",
    "mol/l": "mol/L", "mmol/l": "mmol/L", "mol/m3": "mol/m³", "mol/dm3": "mol/dm³",
    "m3": "m³", "l": "L", "dm3": "dm³", "cm3": "cm³", "ml": "mL", "ft3": "ft³", "in3": "in³",
    "kg": "kg", "g": "g", "lb": "lb", "j": "J", "kj": "kJ", "mj": "MJ", "btu": "Btu", "kcal": "kcal", "cal": "cal",
}


def label(unit):
    return _LABELS.get(_norm(unit), unit)


def _norm(unit):
    u = re.sub(r"lbm(?!ol)", "lb", unit.strip().lower().replace(" ", ""))
    return _ALIASES.get(u, u)


def units_of(kind):
    if kind == TEMPERATURE:
        return ["K", "C", "F", "R"]
    return list(_FACTORS[kind]) + list(_MASS_FACTORS.get(kind, {}))


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
    if u in _MASS_FACTORS.get(kind, {}):
        return value * _per_kg_to_molar(kind, u)
    if u not in table:
        raise ValueError(f"Unknown unit '{unit}'. Options: {', '.join(label(o) for o in units_of(kind))}")
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
    if u in _MASS_FACTORS.get(kind, {}):
        return value / _per_kg_to_molar(kind, u)
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
    try:
        shown = from_si(value, unit, kind)
    except ValueError:
        unit = DEFAULT_INPUT_UNITS[kind] if kind != MOLAR_ENERGY else "kJ/mol"
        shown = from_si(value, unit, kind)
    label = _LABELS.get(_norm(unit), unit)
    return f"{shown:.{digits}g} {label}"
