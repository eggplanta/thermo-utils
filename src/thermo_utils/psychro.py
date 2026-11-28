"""Humid air (psychrometrics): state, cooling with dehumidification, mixing, evaporative cooling."""

from scipy.optimize import brentq

from thermo_utils import steplog as log
from thermo_utils.result import Result, row
from thermo_utils.steplog import q
from thermo_utils.units import num
from thermo_utils.units import (
    MASS_FLOW, POWER, PRESSURE as PR, SPECIFIC_ENERGY as E, SPECIFIC_VOLUME as V, TEMPERATURE as T, VOLUME_FLOW,
)

PROPS = ("T", "RH", "W", "Twb", "Tdp", "h")
_HA = {"T": "T", "RH": "R", "W": "W", "Twb": "B", "Tdp": "D", "h": "H"}
LABELS = {"T": "dry-bulb temperature", "RH": "relative humidity (0.6 or 60%)",
          "W": "humidity ratio ω (kg water/kg dry air)", "Twb": "wet-bulb temperature",
          "Tdp": "dew-point temperature", "h": "enthalpy per kg dry air"}
KINDS = {"dehumidify": "cooling with dehumidification", "mixing": "adiabatic mixing of two streams",
         "evaporative": "evaporative cooling"}


# Properties

def _ha(out, k1, v1, k2, v2, p):
    from CoolProp.HumidAirProp import HAPropsSI
    return HAPropsSI(out, _HA[k1], v1, _HA[k2], v2, "P", p)


def air(p, given):
    """Every property of humid air from two of T, RH, W, Twb, Tdp, h (per kg dry air)."""
    pairs = [(k, v) for k, v in given.items() if v is not None]
    if len(pairs) != 2:
        raise ValueError(f"Give exactly two of {', '.join(PROPS)} (got {len(pairs)}).")
    (k1, v1), (k2, v2) = pairs
    try:
        t = given.get("T") or _ha("T", k1, v1, k2, v2, p)
        w = given.get("W") if given.get("W") is not None else _ha("W", k1, v1, k2, v2, p)
        rh = given.get("RH")
        if rh is None:
            try:
                rh = _ha("R", "T", t, "W", w, p)
            except ValueError:
                rh = 1.0 if w <= _ha("W", "T", t, "RH", 1.0, p) * (1 + 1e-6) else 2.0
        if rh > 1 + 1e-6:
            raise ValueError("Supersaturated air (RH > 100%): the data are inconsistent.")
        rh = min(rh, 1.0)
        st = dict(T=t, W=w, RH=rh, Twb=_ha("B", "T", t, "RH", rh, p), Tdp=_ha("D", "T", t, "RH", rh, p),
                  h=_ha("H", "T", t, "W", w, p), v=_ha("Vda", "T", t, "W", w, p), Pv=_ha("P_w", "T", t, "RH", rh, p),
                  P=p)
    except ValueError as exc:
        if "Supersaturated" in str(exc):
            raise
        raise ValueError(f"Humid air state out of range: {exc}") from None
    st["rho"] = (1 + w) / st["v"]
    return st


def _hf(t):
    from CoolProp.CoolProp import PropsSI
    return PropsSI("H", "T", t, "Q", 0, "Water")


def _given(fields, i=""):
    return {k: fields.get(f"{k}{i}") for k in PROPS if fields.get(f"{k}{i}") is not None}


def _log_state(i, st):
    log.line("State {}: T = {}, RH = {} %, ω = {}, Twb = {}, Tdp = {}, h = {}, v = {}", i, q(st["T"], T),
             num(100 * st["RH"]), num(st["W"]), q(st["Twb"], T), q(st["Tdp"], T), q(st["h"], E), q(st["v"], V))


def _state(p, fields, i):
    given = _given(fields, i)
    st = air(p, given)
    if log.on():
        log.step("State {} from {}", i, " and ".join(given))
        _log_state(i, st)
    return st


# Results

def _states_table(res, states, given):
    headers = ["State", "T\n" + _u(T), "RH\n%", "ω\nkg/kg", "Twb\n" + _u(T), "Tdp\n" + _u(T),
               "h\n" + _u(E), "v\n" + _u(V)]
    rows = []
    for i, st in states:
        cells = [(str(i), False)]
        for k, value in (("T", _n(st["T"], T)), ("RH", num(100 * st["RH"])), ("W", num(st["W"])),
                         ("Twb", _n(st["Twb"], T)), ("Tdp", _n(st["Tdp"], T)), ("h", _n(st["h"], E)),
                         ("v", _n(st["v"], V))):
            cells.append((value, f"{k}{i}" in given))
        rows.append(cells)
    res.tables.append(("States (h and v per kg of dry air)", headers, rows))


def _u(kind):
    from thermo_utils.units import display, label
    return label(display.unit_for(kind))


def _n(value, kind):
    from thermo_utils.units import display, from_si
    return num(from_si(value, display.unit_for(kind), kind), si=value)


def _dry_air(st, mdot, vdot, i):
    if mdot is not None and vdot is not None:
        raise ValueError(f"Give m{i} or V{i}, not both.")
    if vdot is not None:
        mdot = vdot / st["v"]
        log.line("ṁa{} = V̇{}/v{} = {} / {} = {}", i, i, i, q(vdot, VOLUME_FLOW), q(st["v"], V), q(mdot, MASS_FLOW))
    return mdot


BASIS = "Real humid air (ASHRAE RP-1485, CoolProp); h = 0 for dry air at 0 °C and liquid water at 0.01 °C."
STATE_EQUATIONS = ["ω = 0.622·Pv/(P − Pv)     RH = Pv/Pg(T)     Tdp = Tsat(Pv)",
                   "h = h_dry air + ω·h_vapor   (per kg dry air)     v per kg dry air"]


def state(P=101325.0, **fields):
    """Humid air from two properties."""
    log.step("Humid air at P = {}", q(P, PR))
    st = air(P, _given(fields))
    _log_state("", st)
    res = Result("Humid air")
    res.equations = list(STATE_EQUATIONS)
    given = {k for k in fields if fields[k] is not None}
    res.section("Humid air", [
        ("Dry-bulb temperature T", _fmt(st["T"], T), "T" in given),
        ("Relative humidity RH", f"{num(100 * st['RH'])} %", "RH" in given),
        ("Humidity ratio ω", f"{num(st['W'])} kg water/kg dry air", "W" in given),
        ("Wet-bulb temperature Twb", _fmt(st["Twb"], T), "Twb" in given),
        ("Dew-point temperature Tdp", _fmt(st["Tdp"], T), "Tdp" in given),
        ("Enthalpy h (per kg dry air)", _fmt(st["h"], E), "h" in given),
        ("Specific volume v (per kg dry air)", _fmt(st["v"], V), False),
        ("Density of the humid air", f"{num(st['rho'])} kg/m³", False),
        ("Partial pressure of water vapor Pv", _fmt(st["Pv"], PR), False),
        ("Pressure P", _fmt(P, PR), False),
    ])
    res.notes.append(BASIS)
    res.data = st
    return res


def _fmt(value, kind):
    from thermo_utils.units import fmt
    return fmt(value, kind)


# Processes

def process(kind, P=101325.0, m1=None, V1=None, m2=None, V2=None, T_water=None, **fields):
    p = P
    log.step("{} at P = {}", KINDS[kind].capitalize(), q(p, PR))
    s1 = _state(p, fields, 1)
    given = {k for k, v in fields.items() if v is not None}
    res = Result(f"Humid air: {KINDS[kind]}")
    res.equations = list(STATE_EQUATIONS)
    rows = []
    if kind == "dehumidify":
        s2 = _state(p, fields, 2)
        if s2["W"] > s1["W"] + 1e-9:
            raise ValueError("ω2 > ω1: the air gains water, so this is not dehumidification.")
        tw = T_water if T_water is not None else s2["T"]
        hw = _hf(tw)
        dw, qout = s1["W"] - s2["W"], s1["h"] - s2["h"] - (s1["W"] - s2["W"]) * hw
        log.step("Mass and energy balances per kg of dry air")
        log.line("Condensate at {}: hw = hf = {}", q(tw, T), q(hw, E))
        log.line("Water removed: ω1 − ω2 = {} − {} = {} kg/kg dry air", num(s1["W"]), num(s2["W"]), num(dw))
        log.line("q_out = (h1 − h2) − (ω1 − ω2)·hw = ({} − {}) − {} × {} = {}", q(s1["h"], E), q(s2["h"], E),
                 num(dw), q(hw, E), q(qout, E))
        res.equations += ["Dry air: ṁa1 = ṁa2     Water: ṁw = ṁa(ω1 − ω2)",
                          "Energy: Q̇out = ṁa(h1 − h2) − ṁw·hw     hw = hf(condensate T)"]
        rows += [row("Water removed (per kg dry air)", f"{num(dw)} kg"), row("Heat removed q_out", qout, E),
                 row("Condensate temperature", tw, T)]
        mdot = _dry_air(s1, m1, V1, 1)
        if mdot is not None:
            rows = [row("Dry air flow ṁa", mdot, MASS_FLOW)] + rows + [
                row("Water removal rate ṁw", mdot * dw, MASS_FLOW), row("Heat removal rate Q̇out", mdot * qout, POWER)]
        states = [(1, s1), (2, s2)]
    elif kind == "evaporative":
        s2 = _evaporative(p, s1, fields, T_water)
        dw = s2["W"] - s1["W"]
        res.equations += ["Adiabatic: h2 = h1 + (ω2 − ω1)·hw   (≈ constant wet-bulb temperature)",
                          "Water evaporated: ṁw = ṁa(ω2 − ω1)"]
        rows += [row("Water evaporated (per kg dry air)", f"{num(dw)} kg"),
                 row("Temperature drop T1 − T2", f"{num(s1['T'] - s2['T'])} K"),
                 row("Lowest reachable T (wet bulb, RH = 100 %)", s1["Twb"], T)]
        mdot = _dry_air(s1, m1, V1, 1)
        if mdot is not None:
            rows = [row("Dry air flow ṁa", mdot, MASS_FLOW)] + rows + [
                row("Water evaporation rate ṁw", mdot * dw, MASS_FLOW)]
        states = [(1, s1), (2, s2)]
    else:
        s2 = _state(p, fields, 2)
        log.step("Dry air flows")
        ma1, ma2 = _dry_air(s1, m1, V1, 1), _dry_air(s2, m2, V2, 2)
        if ma1 is None or ma2 is None:
            raise ValueError("Mixing needs the flow of both streams: m1 or V1, and m2 or V2.")
        ma3 = ma1 + ma2
        w3, h3 = (ma1 * s1["W"] + ma2 * s2["W"]) / ma3, (ma1 * s1["h"] + ma2 * s2["h"]) / ma3
        log.step("Mass and energy balances")
        log.line("ṁa3 = ṁa1 + ṁa2 = {}", q(ma3, MASS_FLOW))
        log.line("ω3 = (ṁa1·ω1 + ṁa2·ω2)/ṁa3 = {}   h3 = (ṁa1·h1 + ṁa2·h2)/ṁa3 = {}", num(w3), q(h3, E))
        try:
            s3 = air(p, {"h": h3, "W": w3})
        except ValueError:
            raise ValueError("The mixture is supersaturated (fog): part of the water condenses.") from None
        _log_state(3, s3)
        res.equations += ["ṁa3 = ṁa1 + ṁa2     ω3 = (ṁa1ω1 + ṁa2ω2)/ṁa3     h3 = (ṁa1h1 + ṁa2h2)/ṁa3",
                          "ṁa = V̇/v     V̇3 = ṁa3·v3"]
        rows += [row("Dry air flow ṁa1", ma1, MASS_FLOW), row("Dry air flow ṁa2", ma2, MASS_FLOW),
                 row("Dry air flow ṁa3", ma3, MASS_FLOW), row("Volume flow of the mixture V̇3", ma3 * s3["v"], VOLUME_FLOW),
                 row("Mixing ratio ṁa1/ṁa3", num(ma1 / ma3))]
        states = [(1, s1), (2, s2), (3, s3)]
    _states_table(res, states, given)
    res.section(KINDS[kind].capitalize(), rows)
    res.notes.append(BASIS)
    res.data = {f"{k}{i}": v for i, st in states for k, v in st.items()}
    return res


def _evaporative(p, s1, fields, t_water):
    target = [(k, fields[f"{k}2"]) for k in ("T", "RH", "W") if fields.get(f"{k}2") is not None]
    if len(target) != 1:
        raise ValueError("Give one property of the outlet: T2, RH2 or W2.")
    key, goal = target[0]
    tw = t_water if t_water is not None else s1["Twb"]
    hw = _hf(tw)
    def outlet(w):
        return air(p, {"h": s1["h"] + (w - s1["W"]) * hw, "W": w})

    def saturated(t):
        sat = air(p, {"T": t, "RH": 1.0})
        return sat["h"] - s1["h"] - (sat["W"] - s1["W"]) * hw

    log.step("State 2: adiabatic, water at {} (hw = {})", q(tw, T), q(hw, E))
    log.line("h2 = h1 + (ω2 − ω1)·hw; ω2 found so that {}2 = {}", key, num(goal))
    t_sat = brentq(saturated, s1["Twb"] - 20, s1["T"], xtol=1e-9)
    s_sat = air(p, {"T": t_sat, "RH": 1.0})
    if key == "RH" and goal >= 1 - 1e-6:
        _log_state(2, s_sat)
        return s_sat
    try:
        w2 = brentq(lambda w: outlet(w)[key] - goal, s1["W"], s_sat["W"] * (1 - 1e-7), xtol=1e-12)
    except ValueError:
        raise ValueError(f"{key}2 cannot be reached by evaporative cooling: the air can only move from state 1 "
                         f"to saturation at the wet-bulb temperature ({s1['Twb'] - 273.15:.4g} °C).") from None
    s2 = outlet(w2)
    _log_state(2, s2)
    return s2


# Menu forms

def _state_rows(i, desc):
    return [(f"{k}{i}", f"{desc}: {LABELS[k]}") for k in PROPS]


FORMS = {
    "state": [(None, "Pressure (default 1 atm)"), ("P", "total pressure"), (None, "Any two of")]
    + [(k, LABELS[k]) for k in PROPS],
    "dehumidify": [(None, "Pressure (default 1 atm)"), ("P", "total pressure"),
                   (None, "Inlet: any two")] + _state_rows(1, "inlet")
    + [(None, "Outlet: any two (e.g. T2 and RH2 = 100%)")] + _state_rows(2, "outlet")
    + [(None, "Flow (optional)"), ("m1", "dry air mass flow"), ("V1", "inlet volume flow"),
       ("T_water", "condensate temperature (default T2)")],
    "evaporative": [(None, "Pressure (default 1 atm)"), ("P", "total pressure"),
                    (None, "Inlet: any two")] + _state_rows(1, "inlet")
    + [(None, "Outlet: one of")] + [(f"{k}2", f"outlet: {LABELS[k]}") for k in ("T", "RH", "W")]
    + [(None, "Flow (optional)"), ("m1", "dry air mass flow"), ("V1", "inlet volume flow"),
       ("T_water", "water temperature (default wet bulb)")],
    "mixing": [(None, "Pressure (default 1 atm)"), ("P", "total pressure"),
               (None, "Stream 1: any two")] + _state_rows(1, "stream 1")
    + [("m1", "stream 1 dry air flow"), ("V1", "stream 1 volume flow"), (None, "Stream 2: any two")]
    + _state_rows(2, "stream 2") + [("m2", "stream 2 dry air flow"), ("V2", "stream 2 volume flow")],
}
