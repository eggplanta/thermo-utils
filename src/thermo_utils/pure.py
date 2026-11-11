"""Pure-fluid properties: H = H_ig + H_R, S = S_ig + S_R."""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import brentq, least_squares
from thermo.eos import PR, RK, SRK, VDW, PRTranslatedConsistent, PRTranslatedTwu

from thermo_utils.substances import require
from thermo_utils.units import R

T0 = 298.15
P0 = 1e5

MODELS = {
    "auto": "Best available (CoolProp reference EOS, else corrected Peng-Robinson)",
    "coolprop": "Reference Helmholtz EOS (CoolProp, multiparameter)",
    "pr-fit": "Peng-Robinson with Twu α and volume translation, fitted to databank Psat and ρ_liq",
    "pr": "Peng-Robinson (1976)",
    "srk": "Soave-Redlich-Kwong",
    "rk": "Redlich-Kwong",
    "vdw": "van der Waals",
    "virial": "Virial truncated at B, Pitzer correlation",
    "ideal": "Ideal gas",
}

_CUBICS = {"pr": PR, "srk": SRK, "rk": RK, "vdw": VDW, "pr-fit": None}


@dataclass
class State:
    T: float
    P: float
    V: float
    phase: str
    model: str
    H: float = None
    S: float = None
    Cp: float = None
    Cv: float = None
    HR: float = None
    SR: float = None
    phi: float = None
    notes: list = field(default_factory=list)

    @property
    def Z(self):
        return self.P * self.V / (R * self.T)

    @property
    def U(self):
        return self.H - self.P * self.V

    @property
    def G(self):
        return self.H - self.T * self.S

    @property
    def A(self):
        return self.U - self.T * self.S

    @property
    def GR(self):
        return R * self.T * math.log(self.phi) if self.phi else None

    @property
    def fugacity(self):
        return self.phi * self.P if self.phi else None


def resolve_model(sub, model):
    model = (model or "auto").lower()
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Options: {', '.join(MODELS)}")
    if model == "auto":
        return "coolprop" if sub.coolprop else "pr-fit"
    if model == "coolprop" and not sub.coolprop:
        raise ValueError(f"{sub.name} is not available in CoolProp. Use pr, srk, virial or ideal.")
    return model


def _ideal_parts(sub, T, P):
    H = sub.H_ig(T0, T)
    S = sub.S_ig(T0, T) - R * math.log(P / P0)
    return H, S, sub.Cp_ig(T)


# Ideal gas

def _ideal_state(sub, T, P, V=None):
    H, S, Cp = _ideal_parts(sub, T, P)
    return State(T, P, R * T / P, "gas (ideal)", "ideal", H, S, Cp, Cp - R, 0.0, 0.0, 1.0)


# Virial

def _pitzer(sub, T):
    Tr = T / sub.Tc
    B0 = 0.083 - 0.422 / Tr**1.6
    B1 = 0.139 - 0.172 / Tr**4.2
    dB0 = 0.675 / Tr**2.6
    dB1 = 0.722 / Tr**5.2
    d2B0 = -2.6 * 0.675 / Tr**3.6
    d2B1 = -5.2 * 0.722 / Tr**6.2
    return Tr, B0, B1, dB0, dB1, d2B0, d2B1


def virial_B(sub, T):
    _, B0, B1, *_ = _pitzer(sub, T)
    return R * sub.Tc / sub.Pc * (B0 + sub.omega * B1)


def _virial_state(sub, T, P):
    require(sub, "Tc", "Pc", "omega")
    Tr, B0, B1, dB0, dB1, d2B0, d2B1 = _pitzer(sub, T)
    w, Pr = sub.omega, P / sub.Pc
    B = virial_B(sub, T)
    HR = R * sub.Tc * Pr * (B0 - Tr * dB0 + w * (B1 - Tr * dB1))
    SR = -R * Pr * (dB0 + w * dB1)
    CpR = -R * Pr * Tr * (d2B0 + w * d2B1)
    H, S, Cp = _ideal_parts(sub, T, P)
    V = R * T / P + B
    st = State(T, P, V, "gas", "virial", H + HR, S + SR, Cp + CpR, None, HR, SR,
               math.exp(Pr / Tr * (B0 + w * B1)))
    if T < sub.Tc and P > sub.Psat(T):
        st.notes.append("P is above Psat: the real fluid is liquid and the virial equation does not apply.")
    elif sub.Vc and V / sub.Vc < 2:
        st.notes.append("Vr < 2: the virial equation truncated at B loses accuracy here.")
    return st


# Cubic EOS

def _cubic(sub, model, **kw):
    require(sub, "Tc", "Pc", "omega")
    if model == "pr-fit":
        alpha, c = _pr_fit(sub)
        return PRTranslatedTwu(Tc=sub.Tc, Pc=sub.Pc, omega=sub.omega, alpha_coeffs=alpha, c=c, **kw)
    return _CUBICS[model](Tc=sub.Tc, Pc=sub.Pc, omega=sub.omega, **kw)


@lru_cache(maxsize=256)
def _pr_fit(sub):
    """Twu α fitted to Psat(Tr 0.5-0.95), c to ρ_liq at Tr 0.7."""
    base = PRTranslatedConsistent(Tc=sub.Tc, Pc=sub.Pc, omega=sub.omega, T=T0, P=P0)
    alpha, c = tuple(base.alpha_coeffs), base.c
    try:
        Ts = np.linspace(0.5, 0.95, 8) * sub.Tc
        Ps = [sub.Psat(T) for T in Ts]

        def residuals(p):
            eos = PRTranslatedTwu(Tc=sub.Tc, Pc=sub.Pc, omega=sub.omega, alpha_coeffs=tuple(p), c=0, T=T0, P=P0)
            out = []
            for T, P in zip(Ts, Ps):
                try:
                    out.append(math.log(eos.Psat(T) / P))
                except Exception:
                    out.append(1.0)
            return out

        alpha = tuple(least_squares(residuals, alpha, bounds=([0, 0.5, 0.1], [5, 5, 5])).x)
    except Exception:
        pass
    try:
        Tr = 0.7 * sub.Tc
        Pr = sub.Psat(Tr) * 1.01
        eos = PRTranslatedTwu(Tc=sub.Tc, Pc=sub.Pc, omega=sub.omega, alpha_coeffs=alpha, c=0, T=Tr, P=Pr)
        c = eos.V_l - sub.Vliq(Tr, Pr)
    except Exception:
        pass
    return alpha, c


def _cubic_root_state(sub, eos, root, model):
    T, P = eos.T, eos.P
    H, S, Cp = _ideal_parts(sub, T, P)
    s = "l" if root == "liquid" else "g"
    HR, SR = getattr(eos, f"H_dep_{s}"), getattr(eos, f"S_dep_{s}")
    Cp_dep, Cv_dep = getattr(eos, f"Cp_dep_{s}"), getattr(eos, f"Cv_dep_{s}")
    return State(T, P, getattr(eos, f"V_{s}"), root, model, H + HR, S + SR, Cp + Cp_dep,
                 Cp - R + Cv_dep, HR, SR, getattr(eos, f"phi_{s}"))


def _cubic_roots(sub, model, T, P):
    eos = _cubic(sub, model, T=T, P=P)
    roots = []
    if hasattr(eos, "V_l"):
        roots.append(_cubic_root_state(sub, eos, "liquid", model))
    if hasattr(eos, "V_g"):
        roots.append(_cubic_root_state(sub, eos, "vapor", model))
    return roots


def _cubic_state(sub, model, T, P):
    roots = _cubic_roots(sub, model, T, P)
    stable = min(roots, key=lambda s: s.phi)
    if T > sub.Tc:
        stable.phase = "supercritical" if P > sub.Pc else "gas"
    return stable


# CoolProp

_PHASES = {
    CoolProp.iphase_liquid: "liquid", CoolProp.iphase_gas: "vapor",
    CoolProp.iphase_twophase: "two-phase", CoolProp.iphase_supercritical: "supercritical",
    CoolProp.iphase_supercritical_gas: "supercritical gas",
    CoolProp.iphase_supercritical_liquid: "supercritical liquid",
}


def _cp_offsets(sub):
    """CoolProp H, S of the ideal gas at (T0, P0), to share this program's reference."""
    if sub.coolprop not in _CP_OFFSETS:
        AS = CoolProp.AbstractState("HEOS", sub.coolprop)
        AS.specify_phase(CoolProp.iphase_gas)
        P_low = 1.0
        AS.update(CoolProp.PT_INPUTS, P_low, T0)
        Z = P_low / (AS.rhomolar() * R * T0)
        h_ig = AS.hmolar() - AS.hmolar_residual()
        s_ig = AS.smolar() - AS.smolar_residual() - R * math.log(Z) - R * math.log(P0 / P_low)
        _CP_OFFSETS[sub.coolprop] = (h_ig, s_ig)
    return _CP_OFFSETS[sub.coolprop]


_CP_OFFSETS = {}


T_MAX = 6000.0


def _cp_state_from(sub, AS):
    T, P, rho = AS.T(), AS.p(), AS.rhomolar()
    Z = P / (rho * R * T)
    phase = _PHASES.get(AS.phase(), "unknown")
    h0, s0 = _cp_offsets(sub)
    st = State(T, P, 1 / rho, phase, "coolprop", AS.hmolar() - h0, AS.smolar() - s0)
    t_max = AS.Tmax()
    if T > t_max * 1.0001:
        st.notes.append(f"T is above the range of the {sub.name} reference EOS ({t_max:.0f} K): "
                        "values are extrapolated (still close to ideal gas at moderate P).")
    if phase == "two-phase":
        st.phase = f"two-phase (vapor fraction {AS.Q():.4g})"
        return st
    st.HR = AS.hmolar_residual()
    st.SR = AS.smolar_residual() + R * math.log(Z)
    st.Cp, st.Cv = AS.cpmolar(), AS.cvmolar()
    st.phi = AS.fugacity_coefficient(0)
    return st


def _single(sub, inputs, a, b, phase=None):
    AS = CoolProp.AbstractState("HEOS", sub.coolprop)
    if phase is not None:
        AS.specify_phase(phase)
    AS.update(inputs, a, b)
    return AS


def _coolprop_state(sub, T, P):
    try:
        AS = _single(sub, CoolProp.PT_INPUTS, P, T)
    except ValueError as exc:
        if "pseudo-pure" in str(exc):
            raise ValueError(f"{sub.name} at T={T:.6g} K, P={P:.6g} Pa lies between its bubble and dew points "
                             "(two-phase). Give a vapor fraction instead.") from None
        raise ValueError(f"CoolProp could not solve {sub.name} at T={T:.6g} K, P={P:.6g} Pa: {exc}") from None
    return _cp_state_from(sub, AS)


# Public API

def state(sub, T, P, model="auto"):
    """Single-phase state at (T, P)."""
    model = resolve_model(sub, model)
    if model == "ideal":
        return _ideal_state(sub, T, P)
    if model == "virial":
        return _virial_state(sub, T, P)
    if model == "coolprop":
        return _coolprop_state(sub, T, P)
    return _cubic_state(sub, model, T, P)


def state_TV(sub, T, V, model="auto"):
    model = resolve_model(sub, model)
    if model == "ideal":
        return _ideal_state(sub, T, R * T / V)
    if model == "virial":
        B = virial_B(sub, T)
        if V <= B:
            raise ValueError("V must be larger than B for the virial equation.")
        return _virial_state(sub, T, R * T / (V - B))
    if model == "coolprop":
        return _cp_state_from(sub, _single(sub, CoolProp.DmolarT_INPUTS, 1 / V, T))
    eos = _cubic(sub, model, T=T, V=V)
    return state(sub, T, eos.P, model) if eos.P > 0 else _raise_negative(eos.P)


def state_PV(sub, P, V, model="auto"):
    model = resolve_model(sub, model)
    if model == "ideal":
        return _ideal_state(sub, P * V / R, P)
    if model == "virial":
        T = brentq(lambda T: R * T / P + virial_B(sub, T) - V, 0.3 * sub.Tc, 50 * sub.Tc)
        return _virial_state(sub, T, P)
    if model == "coolprop":
        return _cp_state_from(sub, _single(sub, CoolProp.DmolarP_INPUTS, 1 / V, P))
    eos = _cubic(sub, model, P=P, V=V)
    roots = _cubic_roots(sub, model, eos.T, P)
    return min(roots, key=lambda s: abs(s.V - V))


def _raise_negative(P):
    raise ValueError(f"The EOS gives a negative pressure ({P:.4g} Pa) at this T and V.")


def all_roots(sub, T, P, model):
    """Liquid-like and vapor-like roots of a cubic EOS at (T, P)."""
    model = resolve_model(sub, model)
    if model in _CUBICS:
        return _cubic_roots(sub, model, T, P)
    return [state(sub, T, P, model)]


def saturation(sub, T=None, P=None, model="auto"):
    """Saturated liquid and vapor states at T or P."""
    model = resolve_model(sub, model)
    if model == "coolprop":
        inputs, val = (CoolProp.QT_INPUTS, T) if T is not None else (CoolProp.PQ_INPUTS, P)
        args = (0.0, val) if T is not None else (val, 0.0)
        liq = _single(sub, inputs, *args)
        args = (1.0, val) if T is not None else (val, 1.0)
        vap = _single(sub, inputs, *args)
        T_sat, P_sat = liq.T(), liq.p()
        L = _cp_state_from(sub, _single(sub, CoolProp.PT_INPUTS, P_sat, T_sat * (1 - 1e-9), CoolProp.iphase_liquid))
        V = _cp_state_from(sub, _single(sub, CoolProp.PT_INPUTS, vap.p(), vap.T() * (1 + 1e-9), CoolProp.iphase_gas))
        L.V, V.V = 1 / liq.rhomolar(), 1 / vap.rhomolar()
        return T_sat, P_sat, L, V
    if model in _CUBICS:
        base = _cubic(sub, model, T=sub.Tc * 0.7, P=1e5)
        T_sat = T if T is not None else base.Tsat(P)
        P_sat = P if P is not None else base.Psat(T_sat)
        if T_sat >= sub.Tc:
            raise ValueError(f"T is above the critical temperature ({sub.Tc:.5g} K).")
        eos = _cubic(sub, model, T=T_sat, P=P_sat)
        L = _cubic_root_state(sub, eos, "liquid", model)
        V = _cubic_root_state(sub, eos, "vapor", model)
        return T_sat, P_sat, L, V
    T_sat = T if T is not None else sub.Tsat(P)
    P_sat = P if P is not None else sub.Psat(T)
    V = state(sub, T_sat, P_sat, model)
    return T_sat, P_sat, None, V


def two_phase_state(sub, T=None, P=None, x=0.5, model="auto"):
    T_sat, P_sat, L, V = saturation(sub, T, P, model)
    if L is None:
        raise ValueError("Two-phase states need a model with a liquid phase (coolprop or a cubic).")
    mix = State(T_sat, P_sat, (1 - x) * L.V + x * V.V, f"two-phase (vapor fraction {x:.4g})", L.model,
                (1 - x) * L.H + x * V.H, (1 - x) * L.S + x * V.S)
    return mix


def state_from(sub, T=None, P=None, V=None, model="auto"):
    """State from any two of T, P, V, including the two-phase region."""
    if sum(v is not None for v in (T, P, V)) != 2:
        raise ValueError("Give exactly two of T, P and V.")
    model = resolve_model(sub, model)
    if V is None:
        return state(sub, T, P, model)
    if T is not None and P is not None:
        raise ValueError("Give exactly two of T, P and V.")
    if model == "coolprop" or model in _CUBICS:
        try:
            _, _, L, Vap = saturation(sub, T, P, model)
        except Exception:
            L = None
        if L is not None and L.V < V < Vap.V:
            return two_phase_state(sub, T, P, (V - L.V) / (Vap.V - L.V), model)
    return state_TV(sub, T, V, model) if T is not None else state_PV(sub, P, V, model)


def state_PH(sub, P, H, model="auto"):
    return _solve_T(sub, P, H, "H", model)


def state_PS(sub, P, S, model="auto"):
    return _solve_T(sub, P, S, "S", model)


def _solve_T(sub, P, target, prop, model):
    """State at P with given H or S, including the two-phase region."""
    model = resolve_model(sub, model)
    has_liquid = model == "coolprop" or model in _CUBICS
    T_sat = T_dew = None
    if has_liquid and P < sub.Pc:
        T_sat, _, L, V = saturation(sub, P=P, model=model)
        T_dew = V.T
        lo, hi = getattr(L, prop), getattr(V, prop)
        if lo <= target <= hi:
            x = (target - lo) / (hi - lo)
            return two_phase_state(sub, P=P, x=x, model=model)
    f = lambda T: getattr(state(sub, T, P, model), prop) - target
    if T_sat is not None and f(T_sat * 0.999999) >= 0:
        a, b = _t_low(sub, model), T_sat * 0.999999
    else:
        a = T_dew * 1.000001 if T_dew is not None else _t_low(sub, model)
        b = _t_high(sub, model, a, f)
    return state(sub, brentq(f, a, b, xtol=1e-9), P, model)


def _t_high(sub, model, a, f):
    t_max = T_MAX
    b = max(a * 1.5, (sub.Tc or 300) * 1.2)
    while f(b) < 0:
        if b >= t_max:
            raise ValueError(f"No state below {t_max:.0f} K reaches this H or S.")
        b = min(b * 2, t_max)
    return b


def _t_low(sub, model):
    if model == "coolprop":
        return CP.PropsSI("Tmin", sub.coolprop) * 1.001
    return max(sub.Tm or 0, 0.3 * sub.Tc) if sub.Tc else 50.0
