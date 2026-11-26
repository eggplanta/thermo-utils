"""Heat effects: sensible and latent heats, adiabatic flame temperature."""

import math

from thermo_utils import steplog as log
from thermo_utils.properties import _f, riedel, watson
from thermo_utils.result import Result, row
from thermo_utils.steplog import K, q
from thermo_utils.units import num
from thermo_utils.units import MOLAR_ENERGY as E, MOLAR_ENTROPY as S, TEMPERATURE as T, R


def _simpson(f, a, b, n=8):
    h = (b - a) / n
    return h / 3 * (f(a) + f(b) + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, n)))


def sensible_heat(sub, T1, T2, phase="gas"):
    """Sensible heat with Cp(T) of the ideal gas or liquid."""
    if phase == "liquid":
        dH, dS, cp, label = sub.H_liq(T1, T2), sub.S_liq(T1, T2), sub.Cp_liq, "liquid"
    else:
        dH, dS, cp, label = sub.H_ig(T1, T2), sub.S_ig(T1, T2), sub.Cp_ig, "ideal gas"
    res = Result(f"Sensible heat of {sub.name} ({label})")
    if log.on():
        model = sub.Cp_liq_model if phase == "liquid" else sub.Cp_ig_model
        log.step("Heat capacity of the {}", label)
        log.line("Correlation: {}", getattr(model, "method", type(model).__name__))
        rows = [[q(t, T), q(cp(t), S)] for t in [T1 + (T2 - T1) * i / 6 for i in range(7)]]
        log.table(["T", "Cp(T)"], rows)
        log.step("Integrals (constant P, no phase change)")
        log.line("Q = ΔH = ∫Cp dT from T1 to T2 = {}", q(dH, E))
        log.line("Check with Simpson's rule (8 intervals): {}", q(_simpson(cp, T1, T2), E))
        log.line("ΔS = ∫Cp/T dT = {}   (Simpson: {})", q(dS, S), q(_simpson(lambda t: cp(t) / t, T1, T2), S))
        if T2 != T1:
            log.step("Mean heat capacities")
            log.line("<Cp>H = ΔH/(T2 − T1) = {} / {} K = {}", q(dH, E), num(T2 - T1), q(dH / (T2 - T1), S))
            log.line("<Cp>S = ΔS/ln(T2/T1) = {} / {} = {}", q(dS, S), q(math.log(T2 / T1)), q(dS / math.log(T2 / T1), S))
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
    if log.on():
        log.step("Vaporization at T = {}", q(T_, T))
        if T_ < sub.Tc:
            src = "reference EOS: h_vap − h_liq at saturation" if sub.coolprop else "databank correlation"
            log.line("ΔHvap = {}   ({})", q(sub.Hvap(T_), E), src)
        else:
            log.line("T is above Tc = {}: no vaporization", q(sub.Tc, T))
        if sub.Tb and sub.Pc and sub.Tc:
            Hn, Trn = riedel(sub), sub.Tb / sub.Tc
            log.step("Riedel at the normal boiling point")
            log.line("Trn = Tb/Tc = {} / {} = {}", K(sub.Tb), K(sub.Tc), q(Trn))
            log.line("ΔHn/(R·Tn) = 1.092(ln Pc − 1.013)/(0.930 − Trn) = 1.092({} − 1.013)/(0.930 − {}) = {}",
                     q(math.log(sub.Pc / 1e5)), q(Trn), q(Hn / (R * sub.Tb)))
            log.line("ΔHn = {} · {} · {} = {}", q(Hn / (R * sub.Tb)), R, K(sub.Tb), q(Hn, E))
            if T_ < sub.Tc:
                Tr = T_ / sub.Tc
                log.step("Watson from Tb to T")
                log.line("ΔH2 = ΔH1[(1 − Tr2)/(1 − Tr1)]^0.38 = {} · [(1 − {})/(1 − {})]^0.38 = {}",
                         q(Hn, E), q(Tr), q(Trn), q(watson(sub, Hn, sub.Tb, T_), E))
        log.step("Fusion")
        log.line("ΔHfus = {} at Tm = {} (databank)", q(sub.Hfus, E), q(sub.Tm, T))
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
    if log.on():
        log.step("Combustion stoichiometry: C{}H{}O{}N{} + a O2 -> b CO2 + c H2O + N2", *(f"{v:g}" for v in (C, H, O, N)))
        log.line("O2 theoretical = C + H/4 − O/2 = {} + {}/4 − {}/2 = {} mol/mol fuel", f"{C:g}", f"{H:g}", f"{O:g}", num(o2))
        log.line("O2 fed = O2 theoretical × (1 + excess) = {} × {} = {}", num(o2), num(1 + excess_air), num(o2_fed))
        log.line("N2 with the air = 79/21 × O2 fed = {}; N2 in products = that + N/2 = {}", num(79 / 21 * o2_fed), num(n2))
        log.table(["Products", "mol/mol fuel"], [[s.formula, num(n)] for s, n in prod.items()])
        log.step("Heat of combustion at 298.15 K (water as vapor)")
        for s_, n in prod.items():
            log.line("{} × ΔHf°({}) = {} × {} = {}", num(n), s_.formula, num(n), q(s_.Hf, E), q(n * s_.Hf, E))
        log.line("ΔHf°(fuel {}) = {}", fuel.formula, q(fuel.Hf, E))
        log.line("ΔH°c = Σ n ΔHf°(products) − ΔHf°(fuel) = {}", q(dH0, E))
        log.step("Energy balance (adiabatic): H_reactants(T_in) = H_products(Tf)")
        log.line("Sensible heat of the reactants from 298.15 K to T_in = {}: {}", q(T_in, T), q(H_in, E))
        log.line("f(Tf) = H_in − ΔH°c − Σ n ∫Cp_ig dT (298.15 K → Tf) = 0, solved with Brent's method on 300-6000 K")
        table = []
        for t in (1000, 1500, 2000, 2500, 3000):
            table.append([q(float(t), T), q(sum(n * s_.H_ig(298.15, t) for s_, n in prod.items()), E), q(f(t), E)])
        log.table(["T", "Σ n ∫Cp dT", "f(T)"], table)
        log.line("Tf = {}", q(Tf, T))
        for s_, n in prod.items():
            log.line("  {} × ∫Cp dT({}) = {}", num(n), s_.formula, q(n * s_.H_ig(298.15, Tf), E))
        log.line("Check: −ΔH°c + H_in = {} = Σ n ∫Cp dT = {}", q(-dH0 + H_in, E),
                 q(sum(n * s_.H_ig(298.15, Tf) for s_, n in prod.items()), E))
    res.section("Complete combustion in air", [
        row("Excess air", f"{excess_air * 100:.4g} %"), row("Inlet temperature", T_in, T),
        row("ΔH° combustion (water vapor)", dH0, E), row("Adiabatic flame temperature", Tf, T),
    ] + [row(f"Products: {s.formula}", f"{num(n)} mol/mol fuel") for s, n in prod.items()])
    res.notes.append("No dissociation; ideal-gas heat capacities.")
    return res
