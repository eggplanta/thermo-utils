"""Heat effects: sensible and latent heats, adiabatic flame temperature."""

import math

from thermo_utils.properties import _f, riedel, watson
from thermo_utils.result import Result, row
from thermo_utils.units import MOLAR_ENERGY as E, MOLAR_ENTROPY as S, TEMPERATURE as T


def sensible_heat(sub, T1, T2, phase="gas"):
    """Sensible heat with Cp(T) of the ideal gas or liquid."""
    if phase == "liquid":
        dH, dS, cp, label = sub.H_liq(T1, T2), sub.S_liq(T1, T2), sub.Cp_liq, "liquid"
    else:
        dH, dS, cp, label = sub.H_ig(T1, T2), sub.S_ig(T1, T2), sub.Cp_ig, "ideal gas"
    res = Result(f"Sensible heat of {sub.name} ({label})")
    res.section(f"{_f(T1, T)} -> {_f(T2, T)}", [
        row("Q = ΔH = ∫Cp dT", dH, E), row("ΔS = ∫Cp/T dT (constant P)", dS, S),
        row("Mean Cp for H  <Cp>H", dH / (T2 - T1) if T2 != T1 else cp(T1), S),
        row("Mean Cp for S  <Cp>S", dS / math.log(T2 / T1) if T2 != T1 else cp(T1), S),
        row("Cp at T1", cp(T1), S), row("Cp at T2", cp(T2), S),
    ])
    res.notes.append("For real-fluid effects of pressure use 'Property change'.")
    return res


def latent_heat(sub, T_=None):
    T_ = T_ or sub.Tb
    res = Result(f"Latent heats of {sub.name}")
    rows = res.section("Vaporization")
    if T_ < sub.Tc:
        rows.append(row(f"ΔHvap at {_f(T_, T)} (best available)", sub.Hvap(T_), E))
    if sub.Tb and sub.Pc:
        Hn = riedel(sub)
        rows.append(row("ΔHvap at Tb, Riedel", Hn, E))
        if T_ < sub.Tc:
            rows.append(row(f"ΔHvap at {_f(T_, T)}, Watson", watson(sub, Hn, sub.Tb, T_), E))
        try:
            rows.append(row("ΔHvap at Tb (databank)", sub.Hvap(sub.Tb), E))
        except Exception:
            pass
    res.section("Fusion", [row("ΔH fusion at Tm", sub.Hfus, E) if sub.Hfus else row("ΔH fusion", "-"),
                           row("Tm", sub.Tm, T) if sub.Tm else row("Tm", "-")])
    return res


def adiabatic_flame(fuel, excess_air=0.0, T_in=298.15):
    """Adiabatic flame temperature with complete combustion in air."""
    from chemicals.elements import simple_formula_parser
    from scipy.optimize import brentq
    from thermo_utils import substances
    atoms = simple_formula_parser(fuel.formula)
    C, H, O, N = (atoms.get(k, 0) for k in "CHON")
    if set(atoms) - set("CHON"):
        raise ValueError("Only C, H, O, N fuels are supported.")
    o2 = C + H / 4 - O / 2
    o2_fed = o2 * (1 + excess_air)
    n2 = 79 / 21 * o2_fed + N / 2
    prod = {"carbon dioxide": C, "water": H / 2, "oxygen": o2_fed - o2, "nitrogen": n2}
    prod = {substances.get(k): v for k, v in prod.items() if v > 0}
    reac = {fuel: 1.0, substances.get("oxygen"): o2_fed, substances.get("nitrogen"): 79 / 21 * o2_fed}
    dH0 = sum(n * s.Hf for s, n in prod.items()) - fuel.Hf
    H_in = sum(n * s.H_ig(298.15, T_in) for s, n in reac.items())
    f = lambda Tf: H_in - dH0 - sum(n * s.H_ig(298.15, Tf) for s, n in prod.items())
    Tf = brentq(f, 300, 6000)
    res = Result(f"Adiabatic flame temperature of {fuel.name}")
    res.section("Complete combustion in air", [
        row("Excess air", f"{excess_air * 100:.4g} %"), row("Inlet temperature", T_in, T),
        row("ΔH° combustion (water vapor)", dH0, E), row("Adiabatic flame temperature", Tf, T),
    ] + [row(f"Products: {s.formula}", f"{n:.5g} mol/mol fuel") for s, n in prod.items()])
    res.notes.append("No dissociation; ideal-gas heat capacities.")
    return res
