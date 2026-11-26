"""Pure-fluid properties: H = H_ig + H_R, S = S_ig + S_R."""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, least_squares
from thermo.eos import PR, RK, SRK, VDW, PRTranslatedConsistent, PRTranslatedTwu

from thermo_utils import steplog as log
from thermo_utils.steplog import K, q
from thermo_utils.substances import CP, CoolProp, require
from thermo_utils.units import num
from thermo_utils.units import MOLAR_ENERGY as E, MOLAR_ENTROPY as S_, MOLAR_VOLUME as V_, PRESSURE as P_
from thermo_utils.units import TEMPERATURE as T_, R

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
    if V <= 0:
        raise ValueError(f"The virial equation does not apply here: it gives V ≤ 0 (Z = {P * V / (R * T):.3g}). "
                         "The fluid is liquid at these conditions.")
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

@lru_cache(maxsize=1)
def _phases():
    return {
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
    phase = _phases().get(AS.phase(), "unknown")
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
            raise ValueError(f"{sub.name} at T={num(T)} K, P={num(P)} Pa lies between its bubble and dew points "
                             "(two-phase). Give a vapor fraction instead.") from None
        raise ValueError(f"CoolProp could not solve {sub.name} at T={num(T)} K, P={num(P)} Pa: {exc}") from None
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
    kind = E if prop == "H" else S_
    log.step("State at P = {} with {} = {}", q(P, P_), prop, q(target, kind))
    with log.muted():
        st = _solve_T_quiet(sub, P, target, prop, model)
    if st.phase.startswith("two-phase"):
        T_sat, P_sat, L, V = saturation(sub, P=P, model=model)
        lo, hi = getattr(L, prop), getattr(V, prop)
        log.line("Saturation at P: Tsat = {}, {}_liq = {}, {}_vap = {}", q(T_sat, T_), prop, q(lo, kind), prop, q(hi, kind))
        log.line("{}_liq <= {} <= {}_vap, so the state is two-phase", prop, prop, prop)
        x = (target - lo) / (hi - lo)
        log.line("x = ({} − {}_liq)/({}_vap − {}_liq) = ({} − {})/({} − {}) = {}",
                 prop, prop, prop, prop, q(target, kind), q(lo, kind), q(hi, kind), q(lo, kind), q(x))
        explain_two_phase(sub, st, L, V, x)
    else:
        log.line("Single phase: solve {}(T, P) = {} for T (Brent's method) -> T = {}", prop, q(target, kind), q(st.T, T_))
        explain_state(sub, st, f"State found at T = {{}}, P = {{}}", q(st.T, T_), q(st.P, P_))
    return st


def _solve_T_quiet(sub, P, target, prop, model):
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


# Step log

_OMEGA = {"pr": ("0.45724", "0.07780"), "srk": ("0.42748", "0.08664"), "rk": ("0.42748", "0.08664"),
          "vdw": ("27/64", "1/8"), "pr-fit": ("0.45724", "0.07780")}
_KAPPA = {"pr": "κ = 0.37464 + 1.54226ω − 0.26992ω²", "srk": "m = 0.480 + 1.574ω − 0.176ω²"}
_ALPHA = {"pr": "α = [1 + κ(1 − √Tr)]²", "srk": "α = [1 + m(1 − √Tr)]²", "rk": "α = 1/√Tr",
          "vdw": "α = 1", "pr-fit": "α = Tr^(N(M−1)) exp[L(1 − Tr^(NM))] (Twu, fitted to Psat)"}
_POLY = {"pr": "Z³ − (1 − B)Z² + (A − 3B² − 2B)Z − (AB − B² − B³) = 0",
         "srk": "Z³ − Z² + (A − B − B²)Z − AB = 0", "rk": "Z³ − Z² + (A − B − B²)Z − AB = 0",
         "vdw": "Z³ − (1 + B)Z² + AZ − AB = 0", "pr-fit": "PR cubic in Z, then V = V_PR − c (volume translation)"}


def explain_ideal_parts(sub, T, P):
    if not log.on():
        return
    H, S, Cp = _ideal_parts(sub, T, P)
    s_int = sub.S_ig(T0, T)
    log.line("Ideal-gas part (reference: ideal gas at 298.15 K and 1 bar, H = S = 0):")
    log.line("  Cp_ig(T) = {}", q(Cp, S_))
    log.line("  H_ig = ∫Cp_ig dT (298.15 K → T) = {}", q(H, E))
    log.line("  S_ig = ∫Cp_ig/T dT − R ln(P/P°) = {} − {} = {}", q(s_int, S_), q(R * math.log(P / P0), S_), q(S, S_))


def explain_state(sub, st, title=None, *args):
    """Log how a single-phase state was calculated."""
    if not log.on():
        return
    if title:
        log.step(title, *args)
    else:
        log.step("State of {} at T = {}, P = {}", sub.name, q(st.T, T_), q(st.P, P_))
    log.line("Model: {} - {}", st.model, MODELS.get(st.model, st.model))
    if st.phase.startswith("two-phase"):
        with log.muted():
            T_sat, P_sat, L, V = saturation(sub, T=st.T, model=st.model)
        log.line("T and P (or V) fall inside the saturation dome: Tsat = {}, Psat = {}", q(T_sat, T_), q(P_sat, P_))
        log.line("Saturated liquid: V = {}, H = {}, S = {}", q(L.V, V_), q(L.H, E), q(L.S, S_))
        log.line("Saturated vapor:  V = {}, H = {}, S = {}", q(V.V, V_), q(V.H, E), q(V.S, S_))
        explain_two_phase(sub, st, L, V, (st.V - L.V) / (V.V - L.V))
        return
    if st.model == "ideal":
        log.line("V = RT/P = {} · {} / {} = {}", R, K(st.T), q(st.P, P_), q(st.V, V_))
        explain_ideal_parts(sub, st.T, st.P)
        log.line("Ideal gas: H_R = S_R = 0, φ = 1")
    elif st.model == "virial":
        _explain_virial(sub, st)
    elif st.model == "coolprop":
        _explain_coolprop(sub, st)
    else:
        _explain_cubic(sub, st)
    if st.model != "coolprop" and st.HR is not None:
        H_ig, S_ig = st.H - st.HR, st.S - st.SR
        log.line("H = H_ig + H_R = {} + {} = {}", q(H_ig, E), q(st.HR, E), q(st.H, E))
        log.line("S = S_ig + S_R = {} + {} = {}", q(S_ig, S_), q(st.SR, S_), q(st.S, S_))
    explain_derived(st)


def explain_derived(st):
    log.line("Z = PV/RT = {}", q(st.Z))
    log.line("U = H − PV = {} − {} = {}", q(st.H, E), q(st.P * st.V, E), q(st.U, E))
    log.line("G = H − TS = {} − {} = {}", q(st.H, E), q(st.T * st.S, E), q(st.G, E))
    log.line("A = U − TS = {} − {} = {}", q(st.U, E), q(st.T * st.S, E), q(st.A, E))
    if st.Cp is not None:
        log.line("Cp = {}   Cv = {}", q(st.Cp, S_), q(st.Cv, S_))
    if st.phi is not None:
        log.line("φ = {}   f = φP = {}", q(st.phi), q(st.phi * st.P, P_))


def _explain_virial(sub, st):
    T, P = st.T, st.P
    Tr, B0, B1, dB0, dB1, *_ = _pitzer(sub, T)
    Pr, w = P / sub.Pc, sub.omega
    B = virial_B(sub, T)
    log.line("Tc = {}, Pc = {}, ω = {}", q(sub.Tc, T_), q(sub.Pc, P_), q(w))
    log.line("Tr = T/Tc = {}   Pr = P/Pc = {}", q(Tr), q(Pr))
    log.line("B0 = 0.083 − 0.422/Tr^1.6 = {}", q(B0))
    log.line("B1 = 0.139 − 0.172/Tr^4.2 = {}", q(B1))
    log.line("B = (RTc/Pc)(B0 + ωB1) = {}", q(B, V_))
    log.line("Z = 1 + BP/RT = {}", q(1 + B * P / (R * T)))
    log.line("V = RT/P + B = {} + {} = {}", q(R * T / P, V_), q(B, V_), q(st.V, V_))
    explain_ideal_parts(sub, T, P)
    log.line("dB0/dTr = 0.675/Tr^2.6 = {}   dB1/dTr = 0.722/Tr^5.2 = {}", q(dB0), q(dB1))
    log.line("H_R = RTc·Pr[B0 − Tr dB0/dTr + ω(B1 − Tr dB1/dTr)] = {}", q(st.HR, E))
    log.line("S_R = −R·Pr(dB0/dTr + ω dB1/dTr) = {}", q(st.SR, S_))
    log.line("ln φ = (Pr/Tr)(B0 + ωB1) = {}  ->  φ = {}", q(math.log(st.phi)), q(st.phi))


def _explain_cubic(sub, st):
    T, P, m = st.T, st.P, st.model
    eos = _cubic(sub, m, T=T, P=P)
    Tr = T / sub.Tc
    oa, ob = _OMEGA.get(m, ("", ""))
    alpha = eos.a_alpha / eos.a
    A, B = eos.a_alpha * P / (R * T) ** 2, eos.b * P / (R * T)
    log.line("Tc = {}, Pc = {}, ω = {}, Tr = T/Tc = {}", q(sub.Tc, T_), q(sub.Pc, P_), q(sub.omega), q(Tr))
    log.line("a = {}·R²Tc²/Pc = {} Pa·m⁶/mol²", oa, q(eos.a))
    if m == "pr-fit":
        log.line("b = 0.07780·RTc/Pc − c = {}", q(eos.b, V_))
    else:
        log.line("b = {}·RTc/Pc = {}", ob, q(eos.b, V_))
    if m in _KAPPA:
        log.line("{} = {}", _KAPPA[m], q(getattr(eos, "kappa", getattr(eos, "m", None))))
    if m == "pr-fit":
        log.line("Twu α coefficients (L, M, N) = {}   volume translation c = {}",
                 ", ".join(f"{c:.5g}" for c in eos.alpha_coeffs), q(eos.c, V_))
    log.line("{} = {}", _ALPHA.get(m, "α(T)"), q(alpha))
    log.line("A = aαP/(RT)² = {}   B = bP/RT = {}", q(A), q(B))
    log.line("Cubic: {}", _POLY.get(m, ""))
    roots = sorted(v.real for v in eos.raw_volumes if abs(complex(v).imag) < 1e-12 * abs(complex(v)) + 1e-30
                   and complex(v).real > 0)
    log.line("Real roots: Z = {}", ", ".join(num(P * v / (R * T)) for v in roots) or "-")
    for r in _cubic_roots(sub, m, T, P):
        log.line("  {} root: Z = {}, V = {}, φ = {}", r.phase, q(r.Z), q(r.V, V_), q(r.phi))
    if hasattr(eos, "V_l") and hasattr(eos, "V_g"):
        log.line("Two roots: the stable one has the smaller φ (lower G) -> {}", st.phase)
    explain_ideal_parts(sub, T, P)
    log.line("H_R and S_R from the EOS departure functions: H_R = {}, S_R = {}", q(st.HR, E), q(st.SR, S_))


def _explain_coolprop(sub, st):
    rho = 1 / st.V
    log.line("Reference EOS for {}: a(T, ρ) = a_ig + a_res, all properties from its derivatives", sub.coolprop)
    if st.phase.startswith("two-phase"):
        log.line("Inside the dome: {}", st.phase)
    else:
        log.line("Solve P(T, ρ) = P for the density: ρ = {} mol/m³", num(rho))
    log.line("V = 1/ρ = {}", q(st.V, V_))
    log.line("H = h(T, P) − h_ig(298.15 K, 1 bar) = {}", q(st.H, E))
    log.line("S = s(T, P) − s_ig(298.15 K, 1 bar) = {}", q(st.S, S_))
    if st.HR is not None:
        log.line("Residual parts: H_R = {}, S_R = {}", q(st.HR, E), q(st.SR, S_))


def explain_saturation(sub, T_sat, P_sat, L, V, model, given="T"):
    if not log.on():
        return
    log.step("Saturation of {} at {} = {}", sub.name, given, q(T_sat if given == "T" else P_sat, T_ if given == "T" else P_))
    if model == "coolprop":
        log.line("Reference EOS: solve P_liq = P_vap, T_liq = T_vap and g_liq = g_vap (Maxwell criterion)")
    elif model in _CUBICS:
        log.line("Cubic EOS ({}): find P where the liquid and vapor roots have equal fugacity, φ_liq = φ_vap", model)
    else:
        log.line("Psat from the databank correlation; vapor from the {} model", model)
    log.line("Tsat = {}   Psat = {}", q(T_sat, T_), q(P_sat, P_))
    if L is not None:
        log.line("Saturated liquid: V = {}, Z = {}, H = {}, S = {}", q(L.V, V_), q(L.Z), q(L.H, E), q(L.S, S_))
    log.line("Saturated vapor:  V = {}, Z = {}, H = {}, S = {}", q(V.V, V_), q(V.Z), q(V.H, E), q(V.S, S_))
    if L is not None:
        if L.phi and V.phi:
            log.line("Check: φ_liq = {}, φ_vap = {}", q(L.phi), q(V.phi))
        log.line("ΔHvap = H_vap − H_liq = {}", q(V.H - L.H, E))
        log.line("ΔSvap = S_vap − S_liq = {} (= ΔHvap/Tsat = {})", q(V.S - L.S, S_), q((V.H - L.H) / T_sat, S_))


def explain_two_phase(sub, st, L, V, x):
    if not log.on():
        return
    log.line("Two-phase mixture with vapor fraction x = {}", q(x))
    log.line("V = (1 − x)V_liq + xV_vap = {}·{} + {}·{} = {}", q(1 - x), q(L.V, V_), q(x), q(V.V, V_), q(st.V, V_))
    log.line("H = (1 − x)H_liq + xH_vap = {}·{} + {}·{} = {}", q(1 - x), q(L.H, E), q(x), q(V.H, E), q(st.H, E))
    log.line("S = (1 − x)S_liq + xS_vap = {}·{} + {}·{} = {}", q(1 - x), q(L.S, S_), q(x), q(V.S, S_), q(st.S, S_))
    explain_derived(st)
