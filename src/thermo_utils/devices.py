"""Flow devices: compressor, pump, turbine and throttle valve."""

import numpy as np

from thermo_utils import pure
from thermo_utils import steplog as log
from thermo_utils.steplog import q
from thermo_utils.properties import _model_label, _molar_flow, _state_rows, ts_diagram
from thermo_utils.result import Result, row
from thermo_utils.units import num
from thermo_utils.units import MOLAR_ENERGY as E, MOLAR_ENTROPY as S, POWER, PRESSURE as P, TEMPERATURE as T


def flow_device(sub, kind, T1, P1, P2, efficiency=1.0, model="auto", flow=None):
    """Compressor, pump, turbine/expander or throttle valve."""
    s1 = pure.state(sub, T1, P1, model)
    res = Result(f"{kind.capitalize()} - {sub.name}", model=pure.resolve_model(sub, model))
    pure.explain_state(sub, s1, "Inlet state 1: T1 = {}, P1 = {}", q(T1, T), q(P1, P))
    if kind == "valve":
        log.line("Throttling: adiabatic, no work, ΔKE ≈ ΔPE ≈ 0  ->  H2 = H1 = {}", q(s1.H, E))
        s2 = pure.state_PH(sub, P2, s1.H, model)
        log.step("Result")
        log.line("ΔT = T2 − T1 = {} − {} = {} K", q(s2.T, T), q(s1.T, T), num(s2.T - s1.T))
        log.line("ΔS = S2 − S1 = {} − {} = {} (entropy generated, > 0)", q(s2.S, S), q(s1.S, S), q(s2.S - s1.S, S))
        res.section("Inlet", _state_rows(s1, False))
        res.section("Outlet (isenthalpic, H2 = H1)", _state_rows(s2, False))
        res.section("Result", [row("ΔT", f"{num(s2.T - s1.T)} K"), row("ΔS (entropy generated)", s2.S - s1.S, S)])
        path = []
        with log.muted():
            path = _isenthalp(sub, P1, P2, s1, model)
        res.diagram = ts_diagram(sub, model, [s1, s2], ["1", "2"], [path or [(s1.S, s1.T), (s2.S, s2.T)]],
                                 [("accent", "isenthalpic H2 = H1")], "valve")
        return res
    return _work_device(res, sub, kind, s1, P2, efficiency, model, flow)


def _isenthalp(sub, P1, P2, s1, model):
    path = []
    for p in np.geomspace(P1, P2, 12):
            try:
                st = pure.state_PH(sub, p, s1.H, model)
                path.append((st.S, st.T))
            except Exception:
                pass
    return path


def _work_device(res, sub, kind, s1, P2, efficiency, model, flow):
    log.line("Isentropic outlet: S2s = S1 = {} at P2 = {}", q(s1.S, S), q(P2, P))
    s2s = pure.state_PS(sub, P2, s1.S, model)
    Ws = s2s.H - s1.H
    log.step("Isentropic work (steady flow, adiabatic: W = ΔH)")
    log.line("Ws = H2s − H1 = {} − {} = {}", q(s2s.H, E), q(s1.H, E), q(Ws, E))
    if kind == "turbine":
        W = Ws * efficiency
        log.line("Turbine: W = η·Ws = {} × {} = {}", num(efficiency), q(Ws, E), q(W, E))
    else:
        W = Ws / efficiency
        log.line("{}: W = Ws/η = {} / {} = {}", kind.capitalize(), q(Ws, E), num(efficiency), q(W, E))
    log.line("Actual outlet: H2 = H1 + W = {} + {} = {} at P2", q(s1.H, E), q(W, E), q(s1.H + W, E))
    s2 = pure.state_PH(sub, P2, s1.H + W, model)
    log.step("Work and entropy")
    log.line("W per kg = W/M = {} / {} g/mol = {} kJ/kg", q(W, E), f"{sub.MW:.5g}", num(W / sub.MW))
    log.line("ΔS = S2 − S1 = {} − {} = {}", q(s2.S, S), q(s1.S, S), q(s2.S - s1.S, S))
    res.section("Inlet", _state_rows(s1, False))
    res.section("Isentropic outlet", _state_rows(s2s, False))
    res.section(f"Actual outlet (η = {efficiency:.4g})", _state_rows(s2, False))
    rows = res.section("Work (shaft work = ΔH, adiabatic)", [
        row("Isentropic work Ws", Ws, E), row("Actual work W", W, E),
        row("W per kg", f"{num(W / sub.MW)} kJ/kg"), row("ΔS (entropy generated)", s2.S - s1.S, S),
    ])
    if flow is not None:
        n = _molar_flow(sub, flow)
        log.line("Molar flow ṅ = {} mol/s{}", num(n), f" (from ṁ/M = {num(flow[1])} kg/s / {num(sub.MW / 1000)} kg/mol)"
                 if flow[0] == "mass" else "")
        log.line("Power = ṅ·W = {} × {} = {}", num(n), q(W, E), q(n * W, POWER))
        rows.append(row("Power ṅ·W", n * W, POWER))
        rows.append(row("Isentropic power", n * Ws, POWER))
    states, labels = [s1, s2], ["1", "2"]
    curves, legend = [[(s1.S, s1.T), (s2.S, s2.T)]], [("accent", "actual path")]
    if abs(s2.S - s2s.S) > 1e-9 * (abs(s2.S) + 1):
        states.append(s2s)
        labels.append("2s")
        curves.append({"pts": [(s1.S, s1.T), (s2s.S, s2s.T)], "style": "accent-dashed"})
        legend.append(("accent-dashed", "isentropic 1→2s"))
    res.diagram = ts_diagram(sub, model, states, labels, curves, legend, kind)
    res.notes.append(f"Model: {_model_label(sub, model)}. W > 0 means work done on the fluid.")
    res.notes += s2.notes
    return res
