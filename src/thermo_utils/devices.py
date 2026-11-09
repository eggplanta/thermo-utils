"""Flow devices: compressor, pump, turbine and throttle valve."""

from thermo_utils import pure
from thermo_utils.properties import _model_label, _molar_flow, _state_rows
from thermo_utils.result import Result, row
from thermo_utils.units import MOLAR_ENERGY as E, MOLAR_ENTROPY as S, POWER


def flow_device(sub, kind, T1, P1, P2, efficiency=1.0, model="auto", flow=None):
    """Compressor, pump, turbine/expander or throttle valve."""
    s1 = pure.state(sub, T1, P1, model)
    res = Result(f"{kind.capitalize()} - {sub.name}", model=pure.resolve_model(sub, model))
    if kind == "valve":
        s2 = pure.state_PH(sub, P2, s1.H, model)
        res.section("Inlet", _state_rows(s1, False))
        res.section("Outlet (isenthalpic, H2 = H1)", _state_rows(s2, False))
        res.section("Result", [row("ΔT", f"{s2.T - s1.T:.6g} K"), row("ΔS (entropy generated)", s2.S - s1.S, S)])
        return res
    s2s = pure.state_PS(sub, P2, s1.S, model)
    Ws = s2s.H - s1.H
    if kind == "turbine":
        W = Ws * efficiency
    else:
        W = Ws / efficiency
    s2 = pure.state_PH(sub, P2, s1.H + W, model)
    res.section("Inlet", _state_rows(s1, False))
    res.section("Isentropic outlet", _state_rows(s2s, False))
    res.section(f"Actual outlet (η = {efficiency:.4g})", _state_rows(s2, False))
    rows = res.section("Work (shaft work = ΔH, adiabatic)", [
        row("Isentropic work Ws", Ws, E), row("Actual work W", W, E),
        row("W per kg", f"{W / sub.MW:.6g} kJ/kg"), row("ΔS (entropy generated)", s2.S - s1.S, S),
    ])
    if flow is not None:
        n = _molar_flow(sub, flow)
        rows.append(row("Power ṅ·W", n * W, POWER))
        rows.append(row("Isentropic power", n * Ws, POWER))
    res.notes.append(f"Model: {_model_label(sub, model)}. W > 0 means work done on the fluid.")
    res.notes += s2.notes
    return res
