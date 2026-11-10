"""Pure-fluid properties: PVT, state, property changes, saturation."""

import math

import numpy as np

from thermo_utils import pure
from thermo_utils.result import Result, row
from thermo_utils.units import (
    DIMENSIONLESS as D, MOLAR_ENERGY as E, MOLAR_ENTROPY as S, MOLAR_VOLUME as V,
    PRESSURE as P, TEMPERATURE as T, R,
)

COMPARE = ["coolprop", "pr-fit", "pr", "srk", "rk", "vdw", "virial", "ideal"]


def _models_for(sub):
    return [m for m in COMPARE if m != "coolprop" or sub.coolprop]


def _model_label(sub, model):
    m = pure.resolve_model(sub, model)
    return m


def _state_rows(st, extended=True):
    rows = [
        row("Phase", st.phase), row("T", st.T, T), row("P", st.P, P),
        row("V", st.V, V), row("Z", st.Z),
    ]
    if extended:
        rows += [
            row("H", st.H, E), row("S", st.S, S), row("U", st.U, E),
            row("G", st.G, E), row("A", st.A, E),
        ]
        if st.Cp is not None:
            rows += [row("Cp", st.Cp, S), row("Cv", st.Cv, S) if st.Cv else row("Cv", "-")]
        if st.HR is not None:
            rows += [row("H residual (H_R)", st.HR, E), row("S residual (S_R)", st.SR, S),
                     row("G residual (G_R)", st.GR, E)]
        if st.phi is not None:
            rows += [row("Fugacity coefficient φ", st.phi), row("Fugacity f", st.fugacity, P)]
    return rows


def _density_rows(sub, st):
    rho = sub.MW / 1000 / st.V
    return [row("Molar mass", f"{sub.MW:.5g} g/mol"), row("Density", f"{rho:.6g} kg/m³")]


# Substance data

def substance_info(sub):
    res = Result(f"{sub.name} ({sub.formula})")
    res.section("Identification", [
        row("CAS", sub.cas), row("Formula", sub.formula), row("Molar mass", f"{sub.MW:.5g} g/mol"),
        row("Reference EOS (CoolProp)", sub.coolprop or "not available (cubic EOS used)"),
    ])
    rows = res.section("Critical and characteristic properties")
    for label, val, kind in [
        ("Tc", sub.Tc, T), ("Pc", sub.Pc, P), ("Vc", sub.Vc, V), ("Zc", sub.Zc, D),
        ("Acentric factor ω", sub.omega, D), ("Normal boiling point Tb", sub.Tb, T),
        ("Melting point Tm", sub.Tm, T), ("Dipole moment (debye)", sub.dipole, D),
    ]:
        rows.append(row(label, val, kind) if val is not None else row(label, "-"))
    rows = res.section("Formation and phase-change data (298.15 K, 1 bar)")
    for label, val, kind in [
        ("ΔHf° (ideal gas)", sub.Hf, E), ("ΔGf° (ideal gas)", sub.Gf, E), ("S° (ideal gas)", sub.S0, S),
        ("ΔHf° (standard phase)", sub.Hf_std, E), ("Standard phase", sub.phase_std, None),
        ("ΔH fusion", sub.Hfus, E),
    ]:
        rows.append(row(label, val, kind) if val is not None else row(label, "-"))
    if sub.Tb:
        try:
            rows.append(row("ΔH vaporization at Tb", sub.Hvap(sub.Tb), E))
        except Exception:
            pass
    rows = res.section("Ideal-gas heat capacity")
    for t in (298.15, 400, 600, 800, 1000):
        try:
            rows.append(row(f"Cp_ig at {t:g} K", sub.Cp_ig(t), S))
        except Exception:
            pass
    if sub.unifac_groups:
        res.section("Activity-model data", [
            row("UNIQUAC r", sub.r), row("UNIQUAC q", sub.q),
            row("UNIFAC groups", ", ".join(f"{k}×{v}" for k, v in sub.unifac_groups.items())),
        ])
    res.notes.append("Data: chemicals/thermo databanks (DIPPR, CRC, ChemSep) and CoolProp.")
    return res


# PVT

def pvt(sub, T_=None, P_=None, V_=None, model="auto"):
    given = sum(x is not None for x in (T_, P_, V_))
    if given != 2:
        raise ValueError("Give exactly two of T, P and V.")
    model = pure.resolve_model(sub, model)
    res = Result(f"PVT of {sub.name}", model=model)
    if V_ is None:
        st = pure.state(sub, T_, P_, model)
        roots = pure.all_roots(sub, T_, P_, model)
        main = res.section(f"Solution ({_model_label(sub, model)})", _state_rows(st, False) + _density_rows(sub, st))
        if len(roots) > 1:
            main.append(row("Liquid-like root V", roots[0].V, V))
            main.append(row("Vapor-like root V", roots[-1].V, V))
            main.append(row("Stable root", "the one with the smallest fugacity"))
    elif P_ is None:
        st = pure.state_TV(sub, T_, V_, model)
        res.section(f"Solution ({_model_label(sub, model)})", _state_rows(st, False) + _density_rows(sub, st))
    else:
        st = pure.state_PV(sub, P_, V_, model)
        res.section(f"Solution ({_model_label(sub, model)})", _state_rows(st, False) + _density_rows(sub, st))
    res.notes += st.notes
    res.section("Comparison between models", _compare(sub, T_, P_, V_))
    res.plot = lambda: _plot_isotherm(sub, st.T, model, st)
    res.plot_name = f"pvt_{sub.name}"
    return res


def _compare(sub, T_, P_, V_):
    rows = []
    for m in _models_for(sub):
        try:
            if V_ is None:
                st = pure.state(sub, T_, P_, m)
                rows.append(row(m, f"V = {_f(st.V, V)}   Z = {st.Z:.5f}   ({st.phase})"))
            elif P_ is None:
                st = pure.state_TV(sub, T_, V_, m)
                rows.append(row(m, f"P = {_f(st.P, P)}   Z = {st.Z:.5f}"))
            else:
                st = pure.state_PV(sub, P_, V_, m)
                rows.append(row(m, f"T = {_f(st.T, T)}   Z = {st.Z:.5f}"))
        except Exception as exc:
            rows.append(row(m, f"failed: {exc}"))
    return rows


def _f(value, kind):
    from thermo_utils.units import fmt
    return fmt(value, kind)


def _plot_isotherm(sub, T_, model, point):
    import matplotlib.pyplot as plt
    from thermo_utils.units import display, from_si
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    P_max = max(point.P * 3, (sub.Pc or 1e6) * 1.2)
    Ps = np.linspace(P_max / 300, P_max, 300)
    for m in _models_for(sub):
        Vs, Zs, Pok = [], [], []
        for p in Ps:
            try:
                st = pure.state(sub, T_, p, m)
                if st.V <= 0 or (m == "virial" and st.Z < 0.5):
                    continue
                Vs.append(from_si(st.V, display.molar_volume, V)); Zs.append(st.Z); Pok.append(from_si(p, display.pressure, P))
            except Exception:
                pass
        if Pok:
            ax1.plot(Vs, Pok, label=m, lw=2 if m == model else 1)
            ax2.plot(Pok, Zs, label=m, lw=2 if m == model else 1)
    ax1.plot(from_si(point.V, display.molar_volume, V), from_si(point.P, display.pressure, P), "ko")
    ax2.plot(from_si(point.P, display.pressure, P), point.Z, "ko")
    ax1.set_xscale("log")
    ax1.set_xlabel(f"V ({display.molar_volume})"); ax1.set_ylabel(f"P ({display.pressure})")
    ax2.set_xlabel(f"P ({display.pressure})"); ax2.set_ylabel("Z")
    ax1.set_title(f"{sub.name}: isotherm at {T_:.2f} K"); ax2.set_title("Compressibility factor")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


# State properties

def state_properties(sub, T_=None, P_=None, x=None, model="auto"):
    st = _state_from(sub, T_, P_, x, model)
    res = Result(f"State of {sub.name}", model=pure.resolve_model(sub, model))
    res.section(f"Properties ({_model_label(sub, model)})", _state_rows(st) + _density_rows(sub, st))
    res.notes += st.notes
    res.notes.append("Reference for H, S, U, G, A: ideal gas at 298.15 K and 1 bar (H = S = 0).")
    if pure.resolve_model(sub, model) == "coolprop" and st.HR is not None:
        res.section("Derivative properties (CoolProp)", _cp_derivatives(sub, st))
    return res


def _cp_derivatives(sub, st):
    import CoolProp
    AS = CoolProp.AbstractState("HEOS", sub.coolprop)
    AS.update(CoolProp.DmolarT_INPUTS, 1 / st.V, st.T)
    rows = []
    for label, fn, unit in [
        ("Isothermal compressibility κ", AS.isothermal_compressibility, "1/Pa"),
        ("Volume expansivity β", AS.isobaric_expansion_coefficient, "1/K"),
        ("Joule-Thomson coefficient μJT", lambda: AS.first_partial_deriv(CoolProp.iT, CoolProp.iP, CoolProp.iHmolar), "K/Pa"),
        ("Speed of sound", AS.speed_sound, "m/s"),
        ("Viscosity", AS.viscosity, "Pa·s"),
        ("Thermal conductivity", AS.conductivity, "W/(m·K)"),
    ]:
        try:
            rows.append(row(label, f"{fn():.6g} {unit}"))
        except Exception:
            pass
    return rows


def _state_from(sub, T_, P_, x, model):
    if x is not None:
        if (T_ is None) == (P_ is None):
            raise ValueError("For a two-phase state give vapor fraction x and either T or P.")
        return pure.two_phase_state(sub, T_, P_, x, model)
    if T_ is None or P_ is None:
        raise ValueError("Give T and P (or vapor fraction x with T or P).")
    return pure.state(sub, T_, P_, model)


# Property change

def property_change(sub, T1=None, P1=None, x1=None, T2=None, P2=None, x2=None, model="auto"):
    """ΔH, ΔS, ΔU, ΔG, ΔA between two states of a pure substance."""
    s1 = _state_from(sub, T1, P1, x1, model)
    s2 = _state_from(sub, T2, P2, x2, model)
    res = Result(f"Property change of {sub.name}", model=pure.resolve_model(sub, model))
    res.section("State 1", _state_rows(s1, False))
    res.section("State 2", _state_rows(s2, False))
    deltas = {"ΔH": s2.H - s1.H, "ΔS": s2.S - s1.S, "ΔU": s2.U - s1.U, "ΔG": s2.G - s1.G,
              "ΔA": s2.A - s1.A}
    rows = res.section(f"Changes 1 -> 2 ({_model_label(sub, model)})")
    for k, v in deltas.items():
        rows.append(row(k, v, S if k == "ΔS" else E))
    if s1.HR is not None and s2.HR is not None:
        dH_ig = sub.H_ig(s1.T, s2.T)
        dS_ig = sub.S_ig(s1.T, s2.T) - R * math.log(s2.P / s1.P)
        res.section("Residual-property path", [
            row("-H_R(state 1)", -s1.HR, E), row("ΔH ideal gas", dH_ig, E), row("+H_R(state 2)", s2.HR, E),
            row("-S_R(state 1)", -s1.SR, S), row("ΔS ideal gas", dS_ig, S), row("+S_R(state 2)", s2.SR, S),
        ])
    res.notes += s1.notes + s2.notes
    res.data = deltas
    return res


def _molar_flow(sub, flow):
    kind, value = flow
    return value / (sub.MW / 1000) if kind == "mass" else value


# Saturation

def saturation(sub, T_=None, P_=None, model="auto"):
    if (T_ is None) == (P_ is None):
        raise ValueError("Give T or P (only one).")
    model = pure.resolve_model(sub, model)
    if model not in ("coolprop", "pr-fit", "pr", "srk", "rk", "vdw"):
        model = "coolprop" if sub.coolprop else "pr-fit"
    T_sat, P_sat, L, Vs = pure.saturation(sub, T_, P_, model)
    res = Result(f"Saturation of {sub.name}", model=model)
    res.section(f"Saturated states ({_model_label(sub, model)})", [
        row("Tsat", T_sat, T), row("Psat", P_sat, P),
        row("V liquid", L.V, V), row("V vapor", Vs.V, V),
        row("Z liquid", L.Z), row("Z vapor", Vs.Z),
        row("ΔH vaporization", Vs.H - L.H, E), row("ΔS vaporization", Vs.S - L.S, S),
    ])
    corr = res.section("Correlations (databank)")
    try:
        corr.append(row("Psat (correlation)", sub._correlations.VaporPressures[0](T_sat), P))
        corr.append(row("ΔHvap (correlation)", sub._correlations.EnthalpyVaporizations[0](T_sat), E))
        corr.append(row("V liquid (correlation)", sub.Vliq(T_sat, P_sat), V))
    except Exception:
        pass
    if sub.Tb and sub.Tc and sub.Pc:
        Hn = riedel(sub)
        corr.append(row("ΔHvap at Tb, Riedel", Hn, E))
        if T_sat < sub.Tc:
            corr.append(row("ΔHvap at Tsat, Watson", watson(sub, Hn, sub.Tb, T_sat), E))
    res.plot = lambda: _plot_psat(sub, model, T_sat, P_sat)
    res.plot_name = f"psat_{sub.name}"
    return res


def riedel(sub):
    Trn = sub.Tb / sub.Tc
    return R * sub.Tb * 1.092 * (math.log(sub.Pc / 1e5) - 1.013) / (0.930 - Trn)


def watson(sub, H1, T1, T2):
    return H1 * ((1 - T2 / sub.Tc) / (1 - T1 / sub.Tc)) ** 0.38


def _plot_psat(sub, model, T_sat, P_sat):
    import matplotlib.pyplot as plt
    from thermo_utils.units import display, from_si
    fig, ax = plt.subplots(figsize=(8, 5))
    t_lo = max(sub.Tm or 0, 0.45 * sub.Tc)
    Ts = np.linspace(t_lo, sub.Tc * 0.999, 150)
    for m in ([model, "pr"] if model != "pr" else ["pr"]):
        pts = []
        for t in Ts:
            try:
                pts.append((from_si(t, display.temperature, T), from_si(pure.saturation(sub, T_=t, model=m)[1], display.pressure, P)))
            except Exception:
                pass
        if pts:
            ax.plot(*zip(*pts), label=m)
    ax.plot(from_si(T_sat, display.temperature, T), from_si(P_sat, display.pressure, P), "ko")
    ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
    ax.set_xlabel(f"T ({display.temperature})"); ax.set_ylabel(f"Psat ({display.pressure})")
    ax.set_title(f"Vapor pressure of {sub.name}")
    fig.tight_layout()
    return fig
