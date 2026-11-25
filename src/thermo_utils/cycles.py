"""Power cycles (Otto, Diesel, Brayton, Rankine) and their single steps, solved from any consistent data."""

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from thermo_utils import steplog as log
from thermo_utils.result import Result
from thermo_utils.steplog import K, q
from thermo_utils.units import (
    DIMENSIONLESS as D, ENERGY, MASS, MASS_FLOW, POWER, PRESSURE as P, SPECIFIC_ENERGY as E,
    SPECIFIC_ENTROPY as S, SPECIFIC_VOLUME as V, TEMPERATURE as T, TYPED, VOLUME, R, display, fmt, from_si,
    label as unit_label, set_molar_mass,
)

T0, P_REF = 298.15, 101325.0
REQUIRED = object()


# Working fluids

class Gas:
    """Ideal gas per kg: Cp(T), or constant Cp (cold air-standard) when k or cp is given."""

    def __init__(self, sub, k=None, cp=None):
        self.sub, self.R = sub, R / (sub.MW / 1000)
        self.cold = k is not None or cp is not None
        if self.cold:
            if k is not None and cp is not None:
                self.R = cp * (1 - 1 / k)
            self.k = k if k is not None else cp / (cp - self.R)
            self.cp = cp if cp is not None else self.k * self.R / (self.k - 1)
            self.cv = self.cp - self.R
        else:
            self.table = _gas_table(sub.name)

    def h(self, t):
        return self.cp * t if self.cold else float(self.table["h"](t))

    def u(self, t):
        return self.h(t) - self.R * t

    def s0(self, t):
        return self.cp * math.log(t / T0) if self.cold else float(self.table["s0"](t))

    def s(self, t, p):
        return self.s0(t) - self.R * math.log(p / P_REF)

    def T_h(self, h):
        if self.cold:
            return h / self.cp
        return self._invert("h", h, self.h, lambda t: self.table["h"](t, 1))

    def T_u(self, u):
        if self.cold:
            return u / self.cv
        return self._invert("u", u, self.u, lambda t: self.table["h"](t, 1) - self.R)

    def T_isentropic_v(self, t1, v_ratio):
        """T2 for s2 = s1 with v1/v2 = v_ratio."""
        if self.cold:
            return t1 * v_ratio ** (self.k - 1)
        return self._invert("phi", self._phi(t1) + self.R * math.log(v_ratio), self._phi,
                            lambda t: self.table["s0"](t, 1) - self.R / t)

    def T_isentropic_p(self, t1, p_ratio):
        """T2 for s2 = s1 with P2/P1 = p_ratio."""
        if self.cold:
            return t1 * p_ratio ** ((self.k - 1) / self.k)
        return self._invert("s0", self.s0(t1) + self.R * math.log(p_ratio), self.s0,
                            lambda t: self.table["s0"](t, 1))

    def _phi(self, t):
        return self.s0(t) - self.R * math.log(t)

    def _invert(self, key, target, value, slope):
        t = float(self.table["T_" + key](target))
        if not 60 <= t <= 6000:
            raise ValueError("temperature out of range")
        for _ in range(3):
            t -= (value(t) - target) / float(slope(t))
        return t

    def state(self, t, p):
        return dict(T=t, P=p, v=self.R * t / p, u=self.u(t), h=self.h(t), s=self.s(t, p))


@lru_cache(maxsize=8)
def _gas_table(name):
    from thermo_utils import substances
    sub = substances.get(name)
    M = sub.MW / 1000
    ts = np.geomspace(60, 6000, 900)
    h_off, s_off = (298.33e3, 1695.79) if name == "air" else (0.0, 0.0)
    h = np.array([sub.H_ig(T0, t) for t in ts]) / M + h_off
    s0 = np.array([sub.S_ig(T0, t) for t in ts]) / M + s_off
    rg = R / M
    phi = s0 - rg * np.log(ts)
    return {
        "h": CubicSpline(ts, h), "s0": CubicSpline(ts, s0),
        "T_h": CubicSpline(h, ts), "T_u": CubicSpline(h - rg * ts, ts),
        "T_s0": CubicSpline(s0, ts), "T_phi": CubicSpline(phi, ts),
    }


class Fluid:
    """Real fluid per kg from its reference EOS (CoolProp), for Rankine."""

    def __init__(self, sub):
        import CoolProp
        if not sub.coolprop:
            raise ValueError(f"{sub.name} has no reference EOS; Rankine needs one (e.g. water).")
        self.sub, self.CP = sub, CoolProp
        self.AS = CoolProp.AbstractState("HEOS", sub.coolprop)
        self.Pc = self.AS.p_critical()

    def _out(self):
        a = self.AS
        return dict(T=a.T(), P=a.p(), v=1 / a.rhomass(), u=a.umass(), h=a.hmass(), s=a.smass())

    def state_ph(self, p, h):
        self.AS.update(self.CP.HmassP_INPUTS, h, p)
        out = self._out()
        out["x"] = self.quality(p, h)
        return out

    def h_ps(self, p, s):
        self.AS.update(self.CP.PSmass_INPUTS, p, s)
        return self.AS.hmass()

    def sat(self, p, q):
        self.AS.update(self.CP.PQ_INPUTS, p, q)
        return self._out()

    def quality(self, p, h):
        if p >= self.Pc:
            return float("nan")
        hf, hg = self.sat(p, 0)["h"], self.sat(p, 1)["h"]
        return (h - hf) / (hg - hf)

    def dome(self, n=60, pv=False, t_min=0):
        AS = self.AS
        t_lo, t_hi = max(AS.Ttriple(), AS.Tmin(), t_min) + 0.01, AS.T_critical() - 0.05
        liq, vap = [], []
        for t in np.linspace(t_lo, t_hi, n):
            for q_, side in ((0, liq), (1, vap)):
                AS.update(self.CP.QT_INPUTS, q_, t)
                side.append((1 / AS.rhomass(), AS.p()) if pv else (AS.smass(), t))
        return liq + vap[::-1]


# Step log

def _log_state(i, st, how=""):
    if not log.on():
        return
    x = st.get("x")
    xs = "" if x is None else f", x = {_quality_text(x)}"
    log.line("State {}{}: T = {}, P = {}, v = {}, u = {}, h = {}, s = {}{}", i, f" ({how})" if how else "",
             q(st["T"], T), q(st["P"], P), q(st["v"], V), q(st["u"], E), q(st["h"], E), q(st["s"], S), xs)


def _log_gas_props(g, st, i):
    if not log.on():
        return
    if g.cold:
        log.line("  v{} = RT{}/P{} = {} · {} / {} = {}", i, i, i, f"{g.R:.6g} J/(kg·K)", K(st["T"]), q(st["P"], P), q(st["v"], V))
        log.line("  u{} = cv·T{} = {}   h{} = cp·T{} = {}", i, i, q(st["u"], E), i, i, q(st["h"], E))
    else:
        log.line("  v{} = RT{}/P{} = {}   h{} = h(T{}) from the ideal-gas table = {}   u{} = h{} − RT{} = {}",
                 i, i, i, q(st["v"], V), i, i, q(st["h"], E), i, i, i, q(st["u"], E))
        log.line("  s{} = s°(T{}) − R ln(P{}/P_ref) = {} − {} = {}", i, i, i, q(g.s0(st["T"]), S),
                 q(g.R * math.log(st["P"] / P_REF), S), q(st["s"], S))


def _gas_state(g, t, p, i, how=""):
    st = g.state(t, p)
    _log_state(i, st, how)
    _log_gas_props(g, st, i)
    return st


def _log_fluid(i, st, how):
    _log_state(i, st, how)


# Variables and models

@dataclass
class Var:
    key: str
    label: str
    kind: str = D
    guess: float = None
    lo: float = None
    hi: float = None
    default: object = REQUIRED
    solve: bool = True


@dataclass
class Model:
    id: str
    title: str
    fluid: str
    states: int
    primaries: list
    outputs: list
    evaluate: Callable
    processes: list = field(default_factory=list)


STATE_KEYS = {"gas": "T P v u h s", "fluid": "T P v u h s x"}
STATE_LABELS = {"T": ("T", T), "P": ("P", P), "v": ("v", V), "u": ("u", E), "h": ("h", E),
                "s": ("s", S), "x": ("x (quality)", D)}


def state_vars(fluid, n):
    return [Var(f"{k}{i}", f"{STATE_LABELS[k][0]}{i}", STATE_LABELS[k][1])
            for i in range(1, n + 1) for k in STATE_KEYS[fluid].split()]


def _flat(states):
    out = {}
    for i, st in enumerate(states, 1):
        out.update({f"{k}{i}": v for k, v in st.items()})
    return out


def _work_closed(g, st, ratio, eta, expand, a="1", b="2"):
    """Adiabatic piston process from st with volume ratio > 1."""
    vr = 1 / ratio if expand else ratio
    t_s = g.T_isentropic_v(st["T"], vr)
    u_s = g.u(t_s)
    u2 = st["u"] - eta * (st["u"] - u_s) if expand else st["u"] + (u_s - st["u"]) / eta
    t2 = g.T_u(u2)
    v2 = st["v"] * ratio if expand else st["v"] / ratio
    if log.on():
        log.line("v{} = v{} {} {} = {}", b, a, "×" if expand else "/", f"{ratio:.6g}", q(v2, V))
        if g.cold:
            log.line("Isentropic, constant cp: T{}s = T{}(v{}/v{})^(k−1) = {} × {}^{} = {}", b, a, a, b,
                     K(st["T"]), f"{vr:.6g}", f"{g.k - 1:.6g}", K(t_s))
        else:
            log.line("Isentropic, variable cp: φ(T) = s°(T) − R ln T, φ(T{}s) = φ(T{}) + R ln(v{}/v{})", b, a, a, b)
            log.line("  φ(T{}) = {}, R ln(v{}/v{}) = {}  ->  T{}s = {}", a, q(g._phi(st["T"]), S), a, b,
                     q(g.R * math.log(vr), S), b, q(t_s, T))
        log.line("u{}s = u(T{}s) = {}", b, b, q(u_s, E))
        if eta != 1.0:
            if expand:
                log.line("u{} = u{} − η(u{} − u{}s) = {} − {} × ({} − {}) = {}", b, a, a, b, q(st["u"], E),
                         f"{eta:.6g}", q(st["u"], E), q(u_s, E), q(u2, E))
            else:
                log.line("u{} = u{} + (u{}s − u{})/η = {} + ({} − {}) / {} = {}", b, a, b, a, q(st["u"], E),
                         q(u_s, E), q(st["u"], E), f"{eta:.6g}", q(u2, E))
            log.line("T{} = T(u{}) = {}", b, b, q(t2, T))
        log.line("P{} = RT{}/v{} = {}", b, b, b, q(g.R * t2 / v2, P))
    return _gas_state(g, t2, g.R * t2 / v2, b), t_s


def _work_flow(g, st, rp, eta, expand, a="1", b="2"):
    """Adiabatic steady-flow process from st with pressure ratio > 1."""
    pr = 1 / rp if expand else rp
    t_s = g.T_isentropic_p(st["T"], pr)
    h_s = g.h(t_s)
    h2 = st["h"] - eta * (st["h"] - h_s) if expand else st["h"] + (h_s - st["h"]) / eta
    p2 = st["P"] / rp if expand else st["P"] * rp
    if log.on():
        log.line("P{} = P{} {} {} = {}", b, a, "/" if expand else "×", f"{rp:.6g}", q(p2, P))
        if g.cold:
            log.line("Isentropic, constant cp: T{}s = T{}(P{}/P{})^((k−1)/k) = {} × {}^{} = {}", b, a, b, a,
                     K(st["T"]), f"{pr:.6g}", f"{(g.k - 1) / g.k:.6g}", K(t_s))
        else:
            log.line("Isentropic, variable cp: s°(T{}s) = s°(T{}) + R ln(P{}/P{}) = {} + {} = {}  ->  T{}s = {}",
                     b, a, b, a, q(g.s0(st["T"]), S), q(g.R * math.log(pr), S),
                     q(g.s0(st["T"]) + g.R * math.log(pr), S), b, q(t_s, T))
        log.line("h{}s = h(T{}s) = {}", b, b, q(h_s, E))
        if eta != 1.0:
            if expand:
                log.line("h{} = h{} − η(h{} − h{}s) = {} − {} × ({} − {}) = {}", b, a, a, b, q(st["h"], E),
                         f"{eta:.6g}", q(st["h"], E), q(h_s, E), q(h2, E))
            else:
                log.line("h{} = h{} + (h{}s − h{})/η = {} + ({} − {}) / {} = {}", b, a, b, a, q(st["h"], E),
                         q(h_s, E), q(st["h"], E), f"{eta:.6g}", q(h2, E))
        log.line("T{} = T(h{}) = {}", b, b, q(g.T_h(h2), T))
    return _gas_state(g, g.T_h(h2), p2, b), t_s


def _engine(g, p, diesel):
    _log_basis(g)
    log.step("State 1 (start of compression)")
    s1 = _gas_state(g, p["T1"], p["P1"], 1)
    log.step("1→2 adiabatic compression, r = v1/v2 = {}", f"{p['r']:.6g}")
    s2, t2s = _work_closed(g, s1, p["r"], p["eta_c"], False)
    if diesel:
        log.step("2→3 heat addition at constant P (P3 = P2), up to T3 = {}", q(p["T3"], T))
        s3 = _gas_state(g, p["T3"], s2["P"], 3)
        q_in = s3["h"] - s2["h"]
        log.line("q_in = h3 − h2 = {} − {} = {}", q(s3["h"], E), q(s2["h"], E), q(q_in, E))
        log.line("rc = v3/v2 = {} / {} = {}", q(s3["v"], V), q(s2["v"], V), f"{s3['v'] / s2['v']:.6g}")
    else:
        log.step("2→3 heat addition at constant v (v3 = v2), up to T3 = {}", q(p["T3"], T))
        log.line("P3 = RT3/v3 = {}", q(g.R * p["T3"] / s2["v"], P))
        s3 = _gas_state(g, p["T3"], g.R * p["T3"] / s2["v"], 3)
        q_in = s3["u"] - s2["u"]
        log.line("q_in = u3 − u2 = {} − {} = {}", q(s3["u"], E), q(s2["u"], E), q(q_in, E))
    log.step("3→4 adiabatic expansion back to v4 = v1, ratio v4/v3 = {}", f"{s1['v'] / s3['v']:.6g}")
    s4, t4s = _work_closed(g, s3, s1["v"] / s3["v"], p["eta_e"], True, "3", "4")
    q_out = s4["u"] - s1["u"]
    w_c, w_net = s2["u"] - s1["u"], q_in - q_out
    if log.on():
        log.step("4→1 heat rejection at constant v, and cycle balance")
        log.line("q_out = u4 − u1 = {} − {} = {}", q(s4["u"], E), q(s1["u"], E), q(q_out, E))
        log.line("w_c = u2 − u1 = {} − {} = {}", q(s2["u"], E), q(s1["u"], E), q(w_c, E))
        log.line("w_e = u3 − u4{} = {}", " + P2(v3 − v2)" if diesel else "", q(w_net + w_c, E))
        log.line("w_net = q_in − q_out = {} − {} = {}", q(q_in, E), q(q_out, E), q(w_net, E))
        log.line("η = w_net/q_in = {} / {} = {}", q(w_net, E), q(q_in, E), f"{w_net / q_in:.6g}")
        if g.cold and p["eta_c"] == 1 and p["eta_e"] == 1 and not diesel:
            log.line("Check (cold, ideal Otto): η = 1 − 1/r^(k−1) = {}", f"{1 - p['r'] ** (1 - g.k):.6g}")
        if g.cold and p["eta_c"] == 1 and p["eta_e"] == 1 and diesel:
            rc = s3["v"] / s2["v"]
            log.line("Check (cold, ideal Diesel): η = 1 − (rc^k − 1)/[k·r^(k−1)·(rc − 1)] = {}",
                     f"{1 - (rc ** g.k - 1) / (g.k * p['r'] ** (g.k - 1) * (rc - 1)):.6g}")
        log.line("MEP = w_net/(v1 − v2) = {} / {} = {}", q(w_net, E), q(s1["v"] - s2["v"], V),
                 q(w_net / (s1["v"] - s2["v"]), P))
        log.line("η_Carnot = 1 − T1/T3 = 1 − {}/{} = {}", K(s1["T"]), K(s3["T"]), f"{1 - s1['T'] / s3['T']:.6g}")
    out = _flat([s1, s2, s3, s4]) | dict(
        r=p["r"], q_in=q_in, q_out=q_out, w_c=w_c, w_e=w_net + w_c, w_net=w_net, eta=w_net / q_in,
        mep=w_net / (s1["v"] - s2["v"]), T2s=t2s, T4s=t4s, eta_carnot=1 - s1["T"] / s3["T"],
        eta_c=p["eta_c"], eta_e=p["eta_e"],
    )
    if diesel:
        out["rc"] = s3["v"] / s2["v"]
    _cycle_rate(out, p, s1["v"] - s2["v"])
    return out


def _cycle_rate(out, p, dv):
    vd, rpm, m = p.get("Vd"), p.get("rpm"), p.get("m")
    if vd is None and m is None:
        return
    if vd is not None and m is not None:
        raise ValueError("give Vd or m, not both")
    if m is None:
        m = vd / dv
        log.line("m = Vd/(v1 − v2) = {} / {} = {}", q(vd, VOLUME), q(dv, V), q(m, MASS))
    out.update(Vd=m * dv, m=m, W_cycle=m * out["w_net"])
    log.line("W per cycle = m·w_net = {} × {} = {}", q(m, MASS), q(out["w_net"], E), q(m * out["w_net"], ENERGY))
    if rpm is not None:
        out.update(rpm=rpm, Wdot=out["W_cycle"] * rpm / 60 / (p["strokes"] / 2))
        log.line("Ẇ = W·(rpm/60)/(strokes/2) = {} × {}/60 / {} = {}", q(out["W_cycle"], ENERGY), f"{rpm:.6g}",
                 f"{p['strokes'] / 2:.6g}", q(out["Wdot"], POWER))


def otto(g, p):
    return _engine(g, p, False)


def diesel(g, p):
    return _engine(g, p, True)


def brayton(g, p):
    _log_basis(g)
    log.step("State 1 (compressor inlet)")
    s1 = _gas_state(g, p["T1"], p["P1"], 1)
    log.step("1→2 compressor, rp = P2/P1 = {}", f"{p['rp']:.6g}")
    s2, t2s = _work_flow(g, s1, p["rp"], p["eta_c"], False)
    log.step("2→3 combustor at constant P, up to T3 = {}", q(p["T3"], T))
    s3 = _gas_state(g, p["T3"], s2["P"], 3)
    log.step("3→4 turbine, P4 = P3/rp")
    s4, t4s = _work_flow(g, s3, p["rp"], p["eta_t"], True, "3", "4")
    w_c, w_t = s2["h"] - s1["h"], s3["h"] - s4["h"]
    q_in, q_out = s3["h"] - s2["h"], s4["h"] - s1["h"]
    if log.on():
        log.step("Energy balance (steady flow: q − w = Δh)")
        log.line("w_c = h2 − h1 = {} − {} = {}", q(s2["h"], E), q(s1["h"], E), q(w_c, E))
        log.line("w_t = h3 − h4 = {} − {} = {}", q(s3["h"], E), q(s4["h"], E), q(w_t, E))
        log.line("q_in = h3 − h2 = {} − {} = {}", q(s3["h"], E), q(s2["h"], E), q(q_in, E))
        log.line("q_out = h4 − h1 = {} − {} = {}", q(s4["h"], E), q(s1["h"], E), q(q_out, E))
        log.line("w_net = w_t − w_c = {}   (= q_in − q_out = {})", q(w_t - w_c, E), q(q_in - q_out, E))
        log.line("η = w_net/q_in = {}   bwr = w_c/w_t = {}", f"{(w_t - w_c) / q_in:.6g}", f"{w_c / w_t:.6g}")
        if g.cold and p["eta_c"] == 1 and p["eta_t"] == 1:
            log.line("Check (cold, ideal Brayton): η = 1 − 1/rp^((k−1)/k) = {}",
                     f"{1 - p['rp'] ** ((1 - g.k) / g.k):.6g}")
        log.line("η_Carnot = 1 − T1/T3 = {}", f"{1 - s1['T'] / s3['T']:.6g}")
    out = _flat([s1, s2, s3, s4]) | dict(
        rp=p["rp"], w_c=w_c, w_t=w_t, w_net=w_t - w_c, q_in=q_in, q_out=q_out, eta=(w_t - w_c) / q_in,
        bwr=w_c / w_t, T2s=t2s, T4s=t4s, eta_carnot=1 - s1["T"] / s3["T"], eta_c=p["eta_c"], eta_t=p["eta_t"],
    )
    _flow_rate(out, p)
    return out


def _flow_rate(out, p):
    m = p.get("mdot")
    if m is not None:
        out.update(mdot=m, Wdot=m * out["w_net"], Qdot_in=m * out["q_in"], Qdot_out=m * out["q_out"])
        log.line("Ẇ = ṁ·w_net = {} × {} = {}", q(m, MASS_FLOW), q(out["w_net"], E), q(out["Wdot"], POWER))
        log.line("Q̇_in = ṁ·q_in = {}   Q̇_out = ṁ·q_out = {}", q(out["Qdot_in"], POWER), q(out["Qdot_out"], POWER))


def rankine(f, p):
    _log_basis(f)
    log.step("State 1: saturated liquid at the condenser pressure P1 = {}", q(p["P1"], P))
    s1 = f.sat(p["P1"], 0)
    s1["x"] = 0.0
    _log_state(1, s1, "x = 0")
    log.step("1→2 pump to P2 = P3 = {}", q(p["P3"], P))
    h2s = f.h_ps(p["P3"], s1["s"])
    log.line("Isentropic: h2s = h(P2, s = s1) = {}", q(h2s, E))
    log.line("Check, incompressible liquid: h2s − h1 ≈ v1(P2 − P1) = {}", q(s1["v"] * (p["P3"] - p["P1"]), E))
    h2 = s1["h"] + (h2s - s1["h"]) / p["eta_p"]
    log.line("h2 = h1 + (h2s − h1)/η_p = {} + ({} − {}) / {} = {}", q(s1["h"], E), q(h2s, E), q(s1["h"], E),
             f"{p['eta_p']:.6g}", q(h2, E))
    s2 = f.state_ph(p["P3"], h2)
    _log_state(2, s2, "from P2, h2")
    log.step("2→3 boiler at constant P to state 3")
    s3 = f.state_ph(p["P3"], p["h3"])
    _log_state(3, s3, "from P3, h3")
    log.step("3→4 turbine to P4 = P1")
    h4s = f.h_ps(p["P1"], s3["s"])
    x4s = f.quality(p["P1"], h4s)
    log.line("Isentropic: s4s = s3 = {}  ->  h4s = h(P1, s3) = {}, x4s = {}", q(s3["s"], S), q(h4s, E), _quality_text(x4s))
    if 0 <= x4s <= 1:
        sf, sg = f.sat(p["P1"], 0), f.sat(p["P1"], 1)
        log.line("  x4s = (s3 − sf)/(sg − sf) = ({} − {})/({} − {}) = {}", q(s3["s"], S), q(sf["s"], S), q(sg["s"], S),
                 q(sf["s"], S), f"{(s3['s'] - sf['s']) / (sg['s'] - sf['s']):.6g}")
        log.line("  h4s = hf + x4s·hfg = {} + {} × {} = {}", q(sf["h"], E), f"{x4s:.6g}", q(sg["h"] - sf["h"], E), q(h4s, E))
    h4 = s3["h"] - p["eta_t"] * (s3["h"] - h4s)
    log.line("h4 = h3 − η_t(h3 − h4s) = {} − {} × ({} − {}) = {}", q(s3["h"], E), f"{p['eta_t']:.6g}", q(s3["h"], E),
             q(h4s, E), q(h4, E))
    s4 = f.state_ph(p["P1"], h4)
    _log_state(4, s4, "from P1, h4")
    w_p, w_t = s2["h"] - s1["h"], s3["h"] - s4["h"]
    q_in, q_out = s3["h"] - s2["h"], s4["h"] - s1["h"]
    if log.on():
        log.step("Energy balance (steady flow: q − w = Δh)")
        log.line("w_p = h2 − h1 = {}   w_t = h3 − h4 = {}", q(w_p, E), q(w_t, E))
        log.line("q_in = h3 − h2 = {}   q_out = h4 − h1 = {}", q(q_in, E), q(q_out, E))
        log.line("w_net = w_t − w_p = {}   (= q_in − q_out = {})", q(w_t - w_p, E), q(q_in - q_out, E))
        log.line("η = w_net/q_in = {}   bwr = w_p/w_t = {}", f"{(w_t - w_p) / q_in:.6g}", f"{w_p / w_t:.6g}")
        log.line("η_Carnot = 1 − T1/T3 = 1 − {}/{} = {}", K(s1["T"]), K(s3["T"]), f"{1 - s1['T'] / s3['T']:.6g}")
    out = _flat([s1, s2, s3, s4]) | dict(
        w_p=w_p, w_t=w_t, w_net=w_t - w_p, q_in=q_in, q_out=q_out, eta=(w_t - w_p) / q_in, bwr=w_p / w_t,
        x4s=f.quality(p["P1"], h4s), eta_carnot=1 - s1["T"] / s3["T"], eta_t=p["eta_t"], eta_p=p["eta_p"],
    )
    _flow_rate(out, p)
    return out


# Single steps

def _ds(s1, s2):
    d = s2["s"] - s1["s"]
    return 0.0 if abs(d) < 1e-9 * (abs(s1["s"]) + abs(s2["s"]) + 1) else d


def step_closed(g, p, expand):
    _log_basis(g)
    log.step("State 1")
    s1 = _gas_state(g, p["T1"], p["P1"], 1)
    log.step("1→2 adiabatic {} (closed system: w = −Δu)", "expansion" if expand else "compression")
    s2, t2s = _work_closed(g, s1, p["r"], p["eta"], expand)
    w = s1["u"] - s2["u"] if expand else s2["u"] - s1["u"]
    log.step("Work and entropy")
    log.line("w = {} = {}", "u1 − u2" if expand else "u2 − u1", q(w, E))
    log.line("Δs = s2 − s1 = {} − {} = {}", q(s2["s"], S), q(s1["s"], S), q(_ds(s1, s2), S))
    return _flat([s1, s2]) | dict(r=p["r"], w=w, T2s=t2s, eta=p["eta"], ds=_ds(s1, s2))


def step_flow(g, p, expand):
    _log_basis(g)
    log.step("State 1")
    s1 = _gas_state(g, p["T1"], p["P1"], 1)
    log.step("1→2 {} (steady flow, adiabatic: w = −Δh)", "turbine" if expand else "compressor")
    s2, t2s = _work_flow(g, s1, p["rp"], p["eta"], expand)
    w = s1["h"] - s2["h"] if expand else s2["h"] - s1["h"]
    log.step("Work and entropy")
    log.line("w = {} = {}", "h1 − h2" if expand else "h2 − h1", q(w, E))
    log.line("Δs = s2 − s1 = {}", q(_ds(s1, s2), S))
    out = _flat([s1, s2]) | dict(rp=p["rp"], w=w, T2s=t2s, eta=p["eta"], ds=_ds(s1, s2))
    if p.get("mdot") is not None:
        out.update(mdot=p["mdot"], Wdot=p["mdot"] * w)
        log.line("Ẇ = ṁ·w = {} × {} = {}", q(p["mdot"], MASS_FLOW), q(w, E), q(p["mdot"] * w, POWER))
    return out


def step_heat_v(g, p):
    _log_basis(g)
    log.step("State 1")
    s1 = _gas_state(g, p["T1"], p["P1"], 1)
    log.step("1→2 heating at constant v (no boundary work): q = u2 − u1")
    log.line("v2 = v1, P2 = RT2/v2 = {}", q(g.R * p["T2"] / s1["v"], P))
    s2 = _gas_state(g, p["T2"], g.R * p["T2"] / s1["v"], 2)
    log.line("q = u2 − u1 = {} − {} = {}   P2/P1 = {}", q(s2["u"], E), q(s1["u"], E), q(s2["u"] - s1["u"], E),
             f"{s2['P'] / s1['P']:.6g}")
    log.line("Δs = s2 − s1 = {}", q(_ds(s1, s2), S))
    return _flat([s1, s2]) | dict(q=s2["u"] - s1["u"], w=0.0, ratio=s2["P"] / s1["P"], ds=_ds(s1, s2))


def step_heat_p(g, p):
    _log_basis(g)
    log.step("State 1")
    s1 = _gas_state(g, p["T1"], p["P1"], 1)
    log.step("1→2 heating at constant P: q = h2 − h1, boundary work P(v2 − v1)")
    s2 = _gas_state(g, p["T2"], p["P1"], 2)
    log.line("q = h2 − h1 = {} − {} = {}", q(s2["h"], E), q(s1["h"], E), q(s2["h"] - s1["h"], E))
    log.line("w = P(v2 − v1) = {} × ({} − {}) = {}", q(p["P1"], P), q(s2["v"], V), q(s1["v"], V),
             q(p["P1"] * (s2["v"] - s1["v"]), E))
    log.line("v2/v1 = {}   Δs = {}", f"{s2['v'] / s1['v']:.6g}", q(_ds(s1, s2), S))
    return _flat([s1, s2]) | dict(q=s2["h"] - s1["h"], w=p["P1"] * (s2["v"] - s1["v"]),
                                  rc=s2["v"] / s1["v"], ds=_ds(s1, s2))


def step_pump(f, p):
    _log_basis(f)
    log.step("State 1")
    h1 = p["h1"] if p["h1"] is not None else f.sat(p["P1"], 0)["h"]
    if p["h1"] is None:
        log.line("h1 not given: saturated liquid at P1, h1 = hf(P1) = {}", q(h1, E))
    s1 = f.state_ph(p["P1"], h1)
    _log_state(1, s1, "from P1, h1")
    log.step("1→2 pump to P2 = {}", q(p["P2"], P))
    h2s = f.h_ps(p["P2"], s1["s"])
    log.line("Isentropic: h2s = h(P2, s1) = {}   (≈ h1 + v1ΔP = {})", q(h2s, E), q(h1 + s1["v"] * (p["P2"] - p["P1"]), E))
    log.line("h2 = h1 + (h2s − h1)/η = {} + ({} − {}) / {} = {}", q(h1, E), q(h2s, E), q(h1, E), f"{p['eta']:.6g}",
             q(h1 + (h2s - h1) / p["eta"], E))
    s2 = f.state_ph(p["P2"], h1 + (h2s - h1) / p["eta"])
    _log_state(2, s2, "from P2, h2")
    log.line("w = h2 − h1 = {}", q(s2["h"] - h1, E))
    return _steam_step(p, s1, s2, w=s2["h"] - h1, eta=p["eta"], T2s=f.state_ph(p["P2"], h2s)["T"])


def step_turbine(f, p):
    _log_basis(f)
    log.step("State 1")
    s1 = f.state_ph(p["P1"], p["h1"])
    _log_state(1, s1, "from P1, h1")
    log.step("1→2 turbine to P2 = {}", q(p["P2"], P))
    h2s = f.h_ps(p["P2"], s1["s"])
    log.line("Isentropic: s2s = s1  ->  h2s = h(P2, s1) = {}, x2s = {}", q(h2s, E), _quality_text(f.quality(p["P2"], h2s)))
    h2 = s1["h"] - p["eta"] * (s1["h"] - h2s)
    log.line("h2 = h1 − η(h1 − h2s) = {} − {} × ({} − {}) = {}", q(s1["h"], E), f"{p['eta']:.6g}", q(s1["h"], E),
             q(h2s, E), q(h2, E))
    s2 = f.state_ph(p["P2"], h2)
    _log_state(2, s2, "from P2, h2")
    log.line("w = h1 − h2 = {}", q(s1["h"] - h2, E))
    return _steam_step(p, s1, s2, w=s1["h"] - s2["h"], eta=p["eta"], x2s=f.quality(p["P2"], h2s))


def step_boiler(f, p):
    _log_basis(f)
    log.step("States at P = {} (constant pressure)", q(p["P1"], P))
    s1, s2 = f.state_ph(p["P1"], p["h1"]), f.state_ph(p["P1"], p["h2"])
    _log_state(1, s1, "from P, h1")
    _log_state(2, s2, "from P, h2")
    log.line("q = h2 − h1 = {} − {} = {}", q(s2["h"], E), q(s1["h"], E), q(s2["h"] - s1["h"], E))
    return _steam_step(p, s1, s2, q=s2["h"] - s1["h"])


def step_condenser(f, p):
    _log_basis(f)
    log.step("States at P = {} (constant pressure)", q(p["P1"], P))
    h2 = p["h2"] if p["h2"] is not None else f.sat(p["P1"], 0)["h"]
    if p["h2"] is None:
        log.line("h2 not given: saturated liquid at P, h2 = hf(P) = {}", q(h2, E))
    s1, s2 = f.state_ph(p["P1"], p["h1"]), f.state_ph(p["P1"], h2)
    _log_state(1, s1, "from P, h1")
    _log_state(2, s2, "from P, h2")
    log.line("q_out = h1 − h2 = {} − {} = {}", q(s1["h"], E), q(h2, E), q(s1["h"] - h2, E))
    return _steam_step(p, s1, s2, q=s1["h"] - s2["h"])


def _steam_step(p, s1, s2, **extra):
    out = _flat([s1, s2]) | extra | dict(ds=_ds(s1, s2))
    log.line("Δs = s2 − s1 = {}", q(_ds(s1, s2), S))
    if p.get("mdot") is not None:
        key = "w" if "w" in extra else "q"
        out.update(mdot=p["mdot"], **{key.upper() + "dot": p["mdot"] * extra[key]})
        log.line("{}dot = ṁ·{} = {} × {} = {}", key.upper(), key, q(p["mdot"], MASS_FLOW), q(extra[key], E),
                 q(p["mdot"] * extra[key], POWER))
    return out


# Model definitions

def _T(key="T1", guess=300.0):
    return Var(key, key, T, guess, 100.0, 6000.0)


def _P(key="P1", default=REQUIRED, guess=1e5):
    return Var(key, key, P, guess, 100.0, 1e9, default)


def _eta(key, label):
    return Var(key, label, D, 0.85, 0.05, 1.0, 1.0)


MDOT = Var("mdot", "Mass flow rate ṁ", MASS_FLOW, 1.0, 1e-6, 1e6, None)
POWER_V = Var("Wdot", "Net power Ẇ", POWER)
ENGINE_RATE = [
    Var("Vd", "Displacement volume Vd = V1 − V2", VOLUME, 1e-3, 1e-8, 10.0, None),
    Var("m", "Mass of gas per cycle", MASS, 1e-3, 1e-9, 100.0, None),
    Var("rpm", "Engine speed (rpm)", D, 3000.0, 1.0, 1e6, None),
    Var("strokes", "Strokes per cycle (4 or 2)", D, default=4.0, solve=False),
]
ENGINE_OUT = [
    Var("q_in", "Heat added q_in", E), Var("q_out", "Heat rejected q_out", E),
    Var("w_c", "Compression work w_c (1→2)", E), Var("w_e", "Expansion work w_e", E),
    Var("w_net", "Net work w_net", E), Var("eta", "Thermal efficiency η", D),
    Var("mep", "Mean effective pressure MEP", P), Var("eta_carnot", "Carnot efficiency (T1, T3)", D),
    Var("T2s", "T2s (isentropic)", T), Var("T4s", "T4s (isentropic)", T),
    Var("W_cycle", "Net work per cycle", ENERGY), POWER_V,
]
FLOW_OUT = [
    Var("w_net", "Net work w_net", E), Var("q_in", "Heat added q_in", E), Var("q_out", "Heat rejected q_out", E),
    Var("eta", "Thermal efficiency η", D), Var("bwr", "Back work ratio", D),
    Var("eta_carnot", "Carnot efficiency (T1, T3)", D), POWER_V,
    Var("Qdot_in", "Heat rate in Q̇_in", POWER), Var("Qdot_out", "Heat rate out Q̇_out", POWER),
]

MODELS = {
    "otto": Model("otto", "Otto cycle", "gas", 4, [
        _T(), _P(), Var("r", "Compression ratio r = v1/v2", D, 8.0, 1.01, 100.0), _T("T3", 1800.0),
        *ENGINE_RATE, _eta("eta_c", "Compression isentropic efficiency"),
        _eta("eta_e", "Expansion isentropic efficiency"),
    ], ENGINE_OUT, otto, [
        ("1→2", "compression", "s"), ("2→3", "heat addition", "v"), ("3→4", "expansion", "s"),
        ("4→1", "heat rejection", "v")]),
    "diesel": Model("diesel", "Diesel cycle", "gas", 4, [
        _T(), _P(), Var("r", "Compression ratio r = v1/v2", D, 18.0, 1.01, 100.0), _T("T3", 2000.0),
        *ENGINE_RATE, _eta("eta_c", "Compression isentropic efficiency"),
        _eta("eta_e", "Expansion isentropic efficiency"),
    ], [Var("rc", "Cutoff ratio rc = v3/v2", D)] + ENGINE_OUT, diesel, [
        ("1→2", "compression", "s"), ("2→3", "heat addition", "P"), ("3→4", "expansion", "s"),
        ("4→1", "heat rejection", "v")]),
    "brayton": Model("brayton", "Brayton cycle", "gas", 4, [
        _T(), Var("rp", "Pressure ratio rp = P2/P1", D, 10.0, 1.01, 200.0), _T("T3", 1300.0),
        MDOT, _P(default=1e5), _eta("eta_c", "Compressor isentropic efficiency"),
        _eta("eta_t", "Turbine isentropic efficiency"),
    ], [Var("w_c", "Compressor work w_c", E), Var("w_t", "Turbine work w_t", E)] + FLOW_OUT + [
        Var("T2s", "T2s (isentropic)", T), Var("T4s", "T4s (isentropic)", T)], brayton, [
        ("1→2", "compressor", "s"), ("2→3", "combustor (heat in)", "P"), ("3→4", "turbine", "s"),
        ("4→1", "heat rejection", "P")]),
    "rankine": Model("rankine", "Rankine cycle", "fluid", 4, [
        Var("P1", "P1", P, 10e3, 100.0, 1e8), Var("P3", "P3", P, 5e6, 100.0, 1e8),
        Var("h3", "h3", E, 3.3e6, 1e5, 6e6), MDOT,
        _eta("eta_t", "Turbine isentropic efficiency"), _eta("eta_p", "Pump isentropic efficiency"),
    ], [Var("w_p", "Pump work w_p", E), Var("w_t", "Turbine work w_t", E)] + FLOW_OUT + [
        Var("x4s", "x4s (isentropic)", D)], rankine, [
        ("1→2", "pump", "s"), ("2→3", "boiler (heat in)", "P"), ("3→4", "turbine", "s"),
        ("4→1", "condenser (heat out)", "P")]),
}

_W = Var("w", "Work", E)
_DS = Var("ds", "Entropy change s2 − s1", S)
STEPS = {
    "compression": Model("compression", "Adiabatic compression in a piston (closed)", "gas", 2, [
        _T(), _P(), Var("r", "Compression ratio r = v1/v2", D, 8.0, 1.0001, 100.0), _eta("eta", "Isentropic efficiency"),
    ], [Var("w", "Work in w = u2 − u1", E), Var("T2s", "T2s (isentropic)", T), _DS],
        lambda g, p: step_closed(g, p, False), [("1→2", "compression", "s")]),
    "expansion": Model("expansion", "Adiabatic expansion in a piston (closed)", "gas", 2, [
        _T(guess=1800.0), _P(guess=5e6), Var("r", "Expansion ratio r = v2/v1", D, 8.0, 1.0001, 100.0),
        _eta("eta", "Isentropic efficiency"),
    ], [Var("w", "Work out w = u1 − u2", E), Var("T2s", "T2s (isentropic)", T), _DS],
        lambda g, p: step_closed(g, p, True), [("1→2", "expansion", "s")]),
    "heat-v": Model("heat-v", "Heat transfer at constant volume (closed)", "gas", 2, [
        _T(), _P(), _T("T2", 1500.0),
    ], [Var("q", "Heat in q = u2 − u1", E), Var("ratio", "Pressure ratio P2/P1", D), _DS],
        step_heat_v, [("1→2", "heat transfer", "v")]),
    "heat-p": Model("heat-p", "Heat transfer at constant pressure", "gas", 2, [
        _T(), _P(), _T("T2", 1500.0),
    ], [Var("q", "Heat in q = h2 − h1", E), Var("w", "Boundary work P(v2 − v1) (closed)", E),
        Var("rc", "Volume ratio v2/v1", D), _DS], step_heat_p, [("1→2", "heat transfer", "P")]),
    "gas-compressor": Model("gas-compressor", "Gas compressor (steady flow)", "gas", 2, [
        _T(), _P(), Var("rp", "Pressure ratio P2/P1", D, 10.0, 1.0001, 200.0), MDOT,
        _eta("eta", "Isentropic efficiency"),
    ], [Var("w", "Work in w = h2 − h1", E), Var("T2s", "T2s (isentropic)", T), _DS,
        Var("Wdot", "Power Ẇ", POWER)], lambda g, p: step_flow(g, p, False), [("1→2", "compressor", "s")]),
    "gas-turbine": Model("gas-turbine", "Gas turbine (steady flow)", "gas", 2, [
        _T(guess=1300.0), _P(guess=1e6), Var("rp", "Pressure ratio P1/P2", D, 10.0, 1.0001, 200.0), MDOT,
        _eta("eta", "Isentropic efficiency"),
    ], [Var("w", "Work out w = h1 − h2", E), Var("T2s", "T2s (isentropic)", T), _DS,
        Var("Wdot", "Power Ẇ", POWER)], lambda g, p: step_flow(g, p, True), [("1→2", "turbine", "s")]),
    "pump": Model("pump", "Pump (steady flow)", "fluid", 2, [
        Var("P1", "P1", P, 10e3, 100.0, 1e8), Var("P2", "P2", P, 5e6, 100.0, 1e8), MDOT,
        _eta("eta", "Isentropic efficiency"), Var("h1", "h1", E, 2e5, -1e5, 5e6, None),
    ], [Var("w", "Work in w = h2 − h1", E), Var("T2s", "T2s (isentropic)", T), _DS, Var("Wdot", "Power Ẇ", POWER)],
        step_pump, [("1→2", "pump", "s")]),
    "steam-turbine": Model("steam-turbine", "Steam turbine (steady flow)", "fluid", 2, [
        Var("P1", "P1", P, 5e6, 100.0, 1e8), Var("h1", "h1", E, 3.3e6, 1e5, 6e6),
        Var("P2", "P2", P, 10e3, 100.0, 1e8), MDOT, _eta("eta", "Isentropic efficiency"),
    ], [Var("w", "Work out w = h1 − h2", E), Var("x2s", "x2s (isentropic)", D), _DS, Var("Wdot", "Power Ẇ", POWER)],
        step_turbine, [("1→2", "turbine", "s")]),
    "boiler": Model("boiler", "Boiler or heater (constant P)", "fluid", 2, [
        Var("P1", "P", P, 5e6, 100.0, 1e8), Var("h1", "h1", E, 2e5, -1e5, 6e6),
        Var("h2", "h2", E, 3.3e6, -1e5, 6e6), MDOT,
    ], [Var("q", "Heat in q = h2 − h1", E), _DS, Var("Qdot", "Heat rate Q̇", POWER)],
        step_boiler, [("1→2", "boiler", "P")]),
    "condenser": Model("condenser", "Condenser or cooler (constant P)", "fluid", 2, [
        Var("P1", "P", P, 10e3, 100.0, 1e8), Var("h1", "h1", E, 2.3e6, -1e5, 6e6), MDOT,
        Var("h2", "h2", E, 2e5, -1e5, 6e6, None),
    ], [Var("q", "Heat out q = h1 − h2", E), _DS, Var("Qdot", "Heat rate Q̇", POWER)],
        step_condenser, [("1→2", "condenser", "P")]),
}
STEP_DESCRIPTIONS = {
    "compression": "piston compression (Otto/Diesel 1→2)",
    "expansion": "piston expansion (Otto/Diesel 3→4)",
    "heat-v": "heat at constant v (Otto 2→3, 4→1)",
    "heat-p": "heat at constant P (Diesel 2→3)",
    "gas-compressor": "gas compressor (Brayton 1→2)",
    "gas-turbine": "gas turbine (Brayton 3→4)",
    "pump": "pump (Rankine 1→2)",
    "boiler": "boiler (Rankine 2→3)",
    "steam-turbine": "steam turbine (Rankine 3→4)",
    "condenser": "condenser (Rankine 4→1)",
}


def all_vars(model):
    seen, out = set(), []
    for v in state_vars(model.fluid, model.states) + model.primaries + model.outputs:
        if v.key not in seen:
            seen.add(v.key)
            out.append(v)
    return out


# Solver

_SCALE = {T: 1.0, P: 1e3, V: 1e-3, E: 1e3, S: 10.0, D: 1e-3, MASS_FLOW: 1e-3, POWER: 1.0, VOLUME: 1e-7,
          MASS: 1e-7, ENERGY: 1.0}


def solve(model, fluid, given):
    prim = {v.key: v for v in model.primaries}
    known = {k: v for k, v in given.items() if v is not None}
    fixed = {k: v for k, v in known.items() if k in prim}
    eqs = {k: v for k, v in known.items() if k not in prim}
    missing = [v for v in model.primaries if v.key not in fixed]
    required = [v for v in missing if v.default is REQUIRED]
    optional = [v for v in missing if v.default is not REQUIRED and v.solve]
    extra = len(eqs) - len(required)
    if extra < 0:
        names = ", ".join(v.key for v in required)
        raise ValueError(f"Not enough data: give {-extra} more value(s). The cycle is set by {names} "
                         "or any other values that fix them.")
    kinds = {v.key: v.kind for v in all_vars(model)}
    for n in range(min(extra, len(optional)) + 1):
        for combo in itertools.combinations(optional, n):
            unknown = required + list(combo)
            base = fixed | {v.key: v.default for v in missing if v not in combo and v.default is not REQUIRED}
            sols = _roots(model, fluid, base, unknown, eqs, kinds)
            if sols:
                assumed = {v.key: v.default for v in optional
                           if v not in combo and v.default is not None and v.kind != D}
                return sols, assumed
    raise ValueError("No solution: the data are inconsistent, out of range or not enough to fix the cycle.")


def _full_rank(jac, n):
    sv = np.linalg.svd(np.atleast_2d(jac), compute_uv=False)
    return len(sv) >= n and sv[0] > 0 and sv[n - 1] > 1e-8 * sv[0]


def _is_log(v):
    return v.lo > 0 and v.kind != E and not (v.kind == D and v.hi <= 1)


def _roots(model, fluid, base, unknown, eqs, kinds):
    def params(x):
        p = dict(base)
        for v, xi in zip(unknown, x):
            p[v.key] = math.exp(xi) if _is_log(v) else xi
        return p

    def resid(x):
        try:
            out = model.evaluate(fluid, params(x))
            r = [(out[k] - t) / (abs(t) + _SCALE.get(kinds[k], 1.0)) for k, t in eqs.items()]
        except Exception:
            return np.full(len(eqs), 1e3)
        r = np.array(r, dtype=float)
        return np.where(np.isfinite(r), r, 1e3)

    if not unknown:
        r = resid([])
        return [model.evaluate(fluid, base) | {"_params": dict(base)}] if np.all(np.abs(r) < 1e-6) else []
    tr = [(lambda z: math.log(z)) if _is_log(v) else (lambda z: z) for v in unknown]
    lo = [f(v.lo) for f, v in zip(tr, unknown)]
    hi = [f(v.hi) for f, v in zip(tr, unknown)]
    first = [f(v.guess) for f, v in zip(tr, unknown)]
    rng = np.random.default_rng(0)
    starts = [first] + [list(rng.uniform(lo, hi) * 0.5 + np.array(first) * 0.5) for _ in range(5)]
    found, xs = [], []
    for x0 in starts:
        x0 = np.clip(x0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                max_nfev=400)
        except Exception:
            continue
        if np.max(np.abs(sol.fun)) > 1e-7 or not _full_rank(sol.jac, len(unknown)):
            continue
        p = params(sol.x)
        vals = np.array([p[v.key] for v in unknown])
        if any(np.all(np.abs(vals - o) <= 1e-5 * np.abs(o) + 1e-12) for o in xs):
            continue
        xs.append(vals)
        found.append(model.evaluate(fluid, p) | {"_params": p, "_x": unknown})
    return found


# Results

HIDDEN_IF_IDEAL = {"T2s": ("eta_c", "eta"), "T4s": ("eta_e", "eta_t"), "x4s": ("eta_t",), "x2s": ("eta",)}


def run(model_id, sub, k=None, cp=None, steps=False, **given):
    model = (STEPS if steps else MODELS)[model_id]
    set_molar_mass(sub.MW)
    if model.fluid == "gas":
        fluid = Gas(sub, k, cp)
    else:
        if k is not None or cp is not None:
            raise ValueError("k and cp apply only to gas cycles.")
        fluid = Fluid(sub)
    given = {key: val for key, val in given.items() if val is not None}
    unknown = set(given) - {v.key for v in all_vars(model)}
    if unknown:
        raise ValueError(f"Unknown inputs for {model_id}: {', '.join(sorted(unknown))}")
    with log.muted():
        sols, assumed = solve(model, fluid, given)
    params, solved = sols[0].get("_params"), sols[0].get("_x", [])
    sols = [{k: v for k, v in s_.items() if not k.startswith("_")} for s_ in sols]
    _explain_solve(model, given, assumed, solved, sols)
    model.evaluate(fluid, params)
    out = sols[0]
    res = Result(f"{model.title} - {sub.name}")
    res.equations = equations(model, fluid, out)
    res.tables.append(("States", *_state_table(model, out, given)))
    extra = [v for v in model.primaries + model.outputs
             if v.key in out and not v.key[-1].isdigit() and _shown(v, out, given)]
    if steps:
        res.section(model.processes[0][1].capitalize(), [_row(v, out, given) for v in extra])
        res.diagram = step_diagram(model, fluid, out)
    else:
        res.tables.append(("Processes", ["Process", "", f"q\n{_unit(E)}", f"w\n{_unit(E)}"],
                           _process_rows(model, out)))
        res.section("Cycle", [_row(v, out, given) for v in extra])
        res.diagram = diagram(model, fluid, out)
        res.plot = lambda: _plot(plot_panels(model, fluid, out), res.title)
        res.plot_name = f"{model.id}_{sub.name}"
    for alt in sols[1:]:
        order = model.primaries + all_vars(model)
        diff = list({v.key: v for v in order if v.key in alt and v.key not in given
                     and _differs(alt[v.key], out[v.key])}.values())[:3]
        res.notes.append("Another solution also fits the data: "
                         + ", ".join(f"{v.key} = {_fmt(alt[v.key], v.kind)}" for v in diff) + ".")
    if assumed:
        res.notes.append("Not given, assumed: " + ", ".join(
            f"{key} = {_fmt(val, _kind(model, key))}" for key, val in assumed.items()))
    res.notes.append(_basis(fluid))
    if model.id == "rankine":
        res.notes.append("State 1 is saturated liquid at P1.")
    if model.id == "pump" and not any(key.endswith("1") and key != "P1" for key in given):
        res.notes.append("Pump inlet taken as saturated liquid at P1.")
    if model.id == "condenser" and not any(key.endswith("2") for key in given):
        res.notes.append("Condenser outlet taken as saturated liquid.")
    res.notes.append("Values in orange are the data you gave.")
    res.data = out
    return res


def _log_basis(fluid):
    if not log.on():
        return
    log.step("Working fluid")
    if isinstance(fluid, Fluid):
        log.line("{} from its reference EOS (CoolProp), per kg; h and s on the steam-table reference", fluid.sub.name)
    elif fluid.cold:
        log.line("Cold air-standard: cp = {}, cv = {}, R = cp − cv = {}, k = cp/cv = {}",
                 f"{fluid.cp:.6g} J/(kg·K)", f"{fluid.cv:.6g} J/(kg·K)", f"{fluid.R:.6g} J/(kg·K)", f"{fluid.k:.6g}")
    else:
        log.line("Ideal gas {} with Cp(T) (variable specific heats), per kg: R = R_u/M = 8.31446/{} = {} J/(kg·K)",
                 fluid.sub.name, f"{fluid.sub.MW / 1000:.6g}", f"{fluid.R:.6g}")
        log.line("h(T) and s°(T) tabulated from ∫Cp dT and ∫Cp/T dT (like the air tables)")


def _explain_solve(model, given, assumed, solved, sols):
    if not log.on():
        return
    log.step("How the problem is set up")
    log.line("The {} is fixed by: {}", model.title.lower(), ", ".join(v.key for v in model.primaries if v.default is REQUIRED))
    if assumed:
        log.line("Not given, assumed: {}", ", ".join(f"{k} = {_fmt(v, _kind(model, k))}" for k, v in assumed.items()))
    derived = [k for k in given if k not in {v.key for v in model.primaries}]
    if solved:
        log.line("Unknown inputs: {}", ", ".join(v.key for v in solved))
        log.line("Equations: {}", ", ".join(f"{k}(unknowns) = {_fmt(given[k], _kind(model, k))}" for k in derived))
        log.line("Solved with least squares from several starting points ({} solution(s) found)", len(sols))
        for i, s_ in enumerate(sols, 1):
            log.line("  solution {}: {}", i, ", ".join(f"{v.key} = {_fmt(s_.get(v.key), v.kind)}" for v in solved))
        log.line("The steps below use solution 1")
    else:
        log.line("All inputs given: direct calculation")


def _shown(v, out, given):
    if v.key in given or v.key == "strokes":
        return v.key != "strokes" or "rpm" in out
    if v.key.startswith("eta_") and v.key != "eta_carnot" or (v.key == "eta" and "w" in out):
        return out[v.key] != 1.0
    if v.key in HIDDEN_IF_IDEAL:
        return any(out.get(e, 1.0) != 1.0 for e in HIDDEN_IF_IDEAL[v.key] if e != "eta" or "w" in out)
    return True


def _basis(fluid):
    if isinstance(fluid, Fluid):
        return f"{fluid.sub.name} from its reference EOS; h and s on the usual steam-table reference."
    if fluid.cold:
        return (f"Cold air-standard: constant cp = {fluid.cp / 1e3:.4g} kJ/(kg·K), "
                f"cv = {fluid.cv / 1e3:.4g} kJ/(kg·K), k = {fluid.k:.4g}.")
    ref = "air-table reference" if fluid.sub.name == "air" else "h = s = 0 at 298.15 K, 1 atm"
    return f"Ideal gas with Cp(T), {ref}."


def _differs(a, b):
    return isinstance(a, float) and isinstance(b, float) and abs(a - b) > 1e-6 * (abs(b) + 1e-9)


def _kind(model, key):
    return next((v.kind for v in all_vars(model) if v.key == key), D)


def _fmt(value, kind, unit=None):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if kind == D:
        return f"{value:.6g}"
    return fmt(value, kind, unit)


def _row(v, out, given):
    return (v.label, _fmt(out[v.key], v.kind, TYPED.get(v.key)), v.key in given)


def _quality_text(x):
    if x is None or math.isnan(x):
        return "supercritical"
    if x > 1:
        return "superheated"
    if x < 0:
        return "liquid"
    return f"{x:.4g}"


def _unit(kind):
    return unit_label(display.unit_for(kind))


def _num(value, kind):
    return f"{from_si(value, display.unit_for(kind), kind):.6g}"


def _state_table(model, out, given):
    keys = STATE_KEYS[model.fluid].split()
    headers = ["State"] + [k if k == "x" else f"{k}\n{_unit(STATE_LABELS[k][1])}" for k in keys]
    rows = []
    for i in range(1, model.states + 1):
        cells = [(str(i), False)]
        for k in keys:
            key = f"{k}{i}"
            text = _quality_text(out.get(key)) if k == "x" else _num(out[key], STATE_LABELS[k][1])
            cells.append((text, key in given))
        rows.append(cells)
    return headers, rows


def _process_rows(model, out):
    rows = []
    eff = {"1→2": out.get("eta_c", out.get("eta_p")), "3→4": out.get("eta_e", out.get("eta_t"))}
    e_key = "u" if model.id in ("otto", "diesel") else "h"
    for step, name, kind in model.processes:
        a, b = int(step[0]), int(step[-1])
        if kind == "s":
            e = eff.get(step)
            label = "isentropic" if e in (None, 1.0) else f"adiabatic, η = {e:.4g}"
        else:
            label = f"{kind} = const"
        q, w = _q_w(out, a, b, kind, e_key)
        rows.append([(f"{step} {name}", False), (label, False), (_num(q, E), False), (_num(w, E), False)])
    return rows


def _q_w(out, a, b, kind, e_key):
    d = out[f"{e_key}{b}"] - out[f"{e_key}{a}"]
    if kind == "s":
        return 0.0, -d
    if kind == "v":
        return d, 0.0
    if e_key == "u":
        return out[f"h{b}"] - out[f"h{a}"], out[f"P{a}"] * (out[f"v{b}"] - out[f"v{a}"])
    return d, 0.0


# Equations

def equations(model, fluid, out):
    cold = getattr(fluid, "cold", False)
    m = model.id
    lines = []
    if m in ("otto", "diesel", "compression", "expansion"):
        if cold:
            lines.append("Isentropic: T2/T1 = (v1/v2)^(k−1)     P2/P1 = (v1/v2)^k")
        else:
            lines.append("Isentropic: s°(T2) − s°(T1) = R ln(P2/P1)   ⇔   ∫cv/T dT = R ln(v1/v2)")
        if m == "compression" or (m in ("otto", "diesel") and out["eta_c"] != 1):
            lines.append("Adiabatic: u2 = u1 + (u2s − u1)/ηc")
        if m == "expansion":
            lines.append("Adiabatic: u2 = u1 − η(u1 − u2s)")
        if m in ("otto", "diesel") and out["eta_e"] != 1:
            lines.append("Adiabatic: u4 = u3 − ηe(u3 − u4s)")
    if m in ("brayton", "gas-compressor", "gas-turbine"):
        lines.append("Isentropic: T2/T1 = (P2/P1)^((k−1)/k)" if cold else "Isentropic: s°(T2) = s°(T1) + R ln(P2/P1)")
        if m == "gas-compressor" or (m == "brayton" and out["eta_c"] != 1):
            lines.append("Adiabatic: h2 = h1 + (h2s − h1)/ηc")
        if m == "gas-turbine":
            lines.append("Adiabatic: h2 = h1 − η(h1 − h2s)")
        if m == "brayton" and out["eta_t"] != 1:
            lines.append("Adiabatic: h4 = h3 − ηt(h3 − h4s)")
    if m == "otto":
        lines += ["q_in = u3 − u2     q_out = u4 − u1     w_net = q_in − q_out",
                  "η = w_net/q_in     MEP = w_net/(v1 − v2)     r = v1/v2"]
        if cold:
            lines.append("Ideal: η = 1 − 1/r^(k−1)")
    if m == "diesel":
        lines += ["q_in = h3 − h2 (P = const)     q_out = u4 − u1     w_net = q_in − q_out",
                  "η = w_net/q_in     MEP = w_net/(v1 − v2)     r = v1/v2     rc = v3/v2"]
        if cold:
            lines.append("Ideal: η = 1 − (rc^k − 1)/[k·r^(k−1)·(rc − 1)]")
    if m in ("otto", "diesel"):
        lines.append("Vd = m(v1 − v2)     Ẇ = m·w_net·rpm/60/(strokes/2)")
    if m == "brayton":
        lines += ["w_c = h2 − h1     w_t = h3 − h4     q_in = h3 − h2     q_out = h4 − h1",
                  "η = w_net/q_in     bwr = w_c/w_t     Ẇ = ṁ·w_net"]
        if cold:
            lines.append("Ideal: η = 1 − 1/rp^((k−1)/k)")
    if m == "rankine":
        lines += ["1: saturated liquid at P1     s2s = s1     s4s = s3",
                  "w_p = h2 − h1 = (h2s − h1)/ηp ≈ v1(P2 − P1)/ηp     w_t = h3 − h4 = ηt(h3 − h4s)",
                  "q_in = h3 − h2     q_out = h4 − h1     η = w_net/q_in     x = (h − hf)/hfg"]
    if m == "heat-v":
        lines += ["q = u2 − u1 = " + ("cv(T2 − T1)" if cold else "∫cv dT"), "P2/P1 = T2/T1   (v = const)"]
    if m == "heat-p":
        lines += ["q = h2 − h1 = " + ("cp(T2 − T1)" if cold else "∫cp dT"),
                  "w = P(v2 − v1)   (closed)     v2/v1 = T2/T1"]
    if m == "pump":
        lines += ["s2s = s1     w = h2 − h1 = (h2s − h1)/η     w_s ≈ v1(P2 − P1)"]
    if m == "steam-turbine":
        lines += ["s2s = s1     w = h1 − h2 = η(h1 − h2s)     x = (h − hf)/hfg"]
    if m in ("boiler", "condenser"):
        lines += ["P = const     q = Δh     x = (h − hf)/hfg"]
    if m in ("gas-compressor", "gas-turbine", "pump", "steam-turbine", "boiler", "condenser"):
        lines.append("Ẇ = ṁ·w     Q̇ = ṁ·q")
    lines.append("v = RT/P" if model.fluid == "gas" else "Properties from (P, h) and (P, s)")
    if model.id in MODELS:
        lines.append("Process table: q > 0 is heat in, w > 0 is work out")
    return lines


# Diagram data

def _plot(panels, title):
    from thermo_utils import plots
    return plots.figure(panels, title)


def diagram(model, fluid, out, pv=None):
    pv = model.id in ("otto", "diesel") if pv is None else pv
    curves, n = [], model.states
    for step, name, kind in model.processes:
        a, b = int(step[0]), int(step[-1])
        pts = _pv_path(out, a, b, kind) if pv else _ts_path(model, fluid, out, a, b, kind)
        curves.append({"pts": pts, "style": "accent", "name": f"{step} {name}"})
    dome = None
    if model.fluid == "fluid":
        t_min = min(out[f"T{i}"] for i in range(1, n + 1)) - 15 if pv else 0
        dome = fluid.dome(pv=pv, t_min=t_min)
    if pv:
        pts = [(out[f"v{i}"], out[f"P{i}"]) for i in range(1, n + 1)]
        axes = (("v", V), ("P", P))
    else:
        pts = [(out[f"s{i}"], out[f"T{i}"]) for i in range(1, n + 1)]
        axes = (("s", S), ("T", T))
    return dict(curves=curves, points=pts, axes=axes, dome=dome, ylog=pv, arrows=True,
                equipment=model.id if model.id in ("brayton", "rankine") else None)


def plot_panels(model, fluid, out):
    """P-v and T-s of a cycle; the enclosed area is the net work when every step is reversible."""
    reversible = all(out.get(k, 1.0) == 1.0 for k in ("eta_c", "eta_e", "eta_t", "eta_p"))
    panels = []
    for pv in (True, False):
        d = diagram(model, fluid, out, pv)
        d["title"] = "P-v diagram" if pv else "T-s diagram"
        d["xlog"] = pv and model.fluid == "fluid"
        d["legend"] = [("accent", "cycle")] + ([("dim", "saturation dome")] if d["dome"] else [])
        if reversible:
            d["fill"] = "area = net work" if pv else "area = net heat = net work"
        panels.append(d)
    return panels


STEP_EQUIPMENT = {
    "compression": "piston", "expansion": "piston", "heat-v": "piston", "heat-p": "heater",
    "gas-compressor": "compressor", "gas-turbine": "turbine", "pump": "pump", "steam-turbine": "turbine",
    "boiler": "heater", "condenser": "cooler",
}


def step_diagram(model, fluid, out):
    """T-s of one step; dashed 1→2s when the step is not isentropic."""
    curve = _ts_path(model, fluid, out, 1, 2, model.processes[0][2])
    pts, labels, curves = [(out["s1"], out["T1"]), (out["s2"], out["T2"])], ["1", "2"], [curve]
    legend = [("accent", "actual path")]
    if model.processes[0][2] == "s" and out.get("eta", 1.0) < 1.0:
        if model.fluid == "gas":
            t2s = out["T2s"]
        else:
            t2s = fluid.state_ph(out["P2"], fluid.h_ps(out["P2"], out["s1"]))["T"]
        pts.append((out["s1"], t2s))
        labels.append("2s")
        curves.append({"pts": [pts[0], pts[2]], "style": "accent-dashed"})
        legend.append(("accent-dashed", "isentropic 1→2s"))
    dome = fluid.dome() if model.fluid == "fluid" else None
    if dome:
        legend.append(("dim", "saturation dome"))
    return dict(curves=curves, points=pts, labels=labels, axes=(("s", S), ("T", T)), dome=dome,
                legend=legend, equipment=STEP_EQUIPMENT[model.id], arrows=True)


def _pv_path(out, a, b, kind):
    va, vb, pa, pb = out[f"v{a}"], out[f"v{b}"], out[f"P{a}"], out[f"P{b}"]
    if kind != "s":
        return [(va, pa), (vb, pb)]
    n = math.log(pb / pa) / math.log(va / vb)
    return [(v, pa * (va / v) ** n) for v in np.linspace(va, vb, 60)]


def _ts_path(model, fluid, out, a, b, kind):
    sa, sb, ta, tb = out[f"s{a}"], out[f"s{b}"], out[f"T{a}"], out[f"T{b}"]
    if kind == "v" and model.fluid == "gas":
        v = out[f"v{a}"]
        return [(fluid.s(t, fluid.R * t / v), t) for t in np.linspace(ta, tb, 60)]
    if kind != "P":
        return [(sa, ta), (sb, tb)]
    p = out[f"P{a}"]
    if model.fluid == "gas":
        return [(fluid.s(t, p), t) for t in np.linspace(ta, tb, 60)]
    pts = []
    for h in np.linspace(out[f"h{a}"], out[f"h{b}"], 80):
        st = fluid.state_ph(p, h)
        pts.append((st["s"], st["T"]))
    return pts


# Menu forms: main inputs only

def _states(desc, keys="TP"):
    return [(f"{k}{i}", f"{d}") for i, d in desc.items() for k in keys]


ENGINE_STATES = {1: "start of compression", 2: "end of compression", 3: "end of heat addition (peak)",
                 4: "end of expansion"}
ENGINE_REAL = [(None, "Real engine (optional)"), ("eta_c", "compression isentropic efficiency"),
               ("eta_e", "expansion isentropic efficiency"), ("Vd", "displacement volume"),
               ("rpm", "engine speed"), ("Wdot", "net power")]
FLOW_REAL = [("mdot", "mass flow rate"), ("Wdot", "power")]
FORMS = {
    "otto": [(None, "States")] + _states(ENGINE_STATES) + [
        (None, "Cycle"), ("r", "compression ratio v1/v2"), ("q_in", "heat added"), ("w_net", "net work"),
        ("eta", "thermal efficiency"), ("mep", "mean effective pressure")] + ENGINE_REAL,
    "diesel": [(None, "States")] + _states(ENGINE_STATES) + [
        (None, "Cycle"), ("r", "compression ratio v1/v2"), ("rc", "cutoff ratio v3/v2"), ("q_in", "heat added"),
        ("w_net", "net work"), ("eta", "thermal efficiency"), ("mep", "mean effective pressure")] + ENGINE_REAL,
    "brayton": [
        (None, "States"), ("T1", "compressor inlet"), ("P1", "compressor inlet (default 1 bar)"),
        ("T2", "compressor outlet"), ("P2", "compressor outlet (= P3)"), ("T3", "turbine inlet"),
        ("T4", "turbine outlet"),
        (None, "Cycle"), ("rp", "pressure ratio P2/P1"), ("q_in", "heat added"), ("w_net", "net work"),
        ("eta", "thermal efficiency"), ("bwr", "back work ratio w_c/w_t"),
        (None, "Real plant (optional)"), ("eta_c", "compressor isentropic efficiency"),
        ("eta_t", "turbine isentropic efficiency"), ("mdot", "mass flow rate"), ("Wdot", "net power")],
    "rankine": [
        (None, "States"), ("P1", "condenser pressure"), ("T1", "condenser temperature (saturation)"),
        ("P3", "boiler pressure"), ("T3", "turbine inlet temperature"),
        ("x3", "turbine inlet quality (1 = saturated vapor)"), ("x4", "turbine outlet quality"),
        (None, "Cycle"), ("q_in", "heat added"), ("w_net", "net work"), ("eta", "thermal efficiency"),
        (None, "Real plant (optional)"), ("eta_t", "turbine isentropic efficiency"),
        ("eta_p", "pump isentropic efficiency"), ("mdot", "mass flow rate"), ("Wdot", "net power")],
    "compression": _states({1: "start", 2: "end"}) + [
        ("r", "compression ratio v1/v2"), ("w", "work in"), ("eta", "isentropic efficiency")],
    "expansion": _states({1: "start", 2: "end"}) + [
        ("r", "expansion ratio v2/v1"), ("w", "work out"), ("eta", "isentropic efficiency")],
    "heat-v": _states({1: "start", 2: "end"}) + [("q", "heat in (negative if out)")],
    "heat-p": [("T1", "start"), ("P1", "pressure"), ("T2", "end"), ("rc", "volume ratio v2/v1"),
               ("q", "heat in (negative if out)")],
    "gas-compressor": _states({1: "inlet", 2: "outlet"}) + [
        ("rp", "pressure ratio P2/P1"), ("w", "work in"), ("eta", "isentropic efficiency")] + FLOW_REAL,
    "gas-turbine": _states({1: "inlet", 2: "outlet"}) + [
        ("rp", "pressure ratio P1/P2"), ("w", "work out"), ("eta", "isentropic efficiency")] + FLOW_REAL,
    "pump": [("P1", "inlet"), ("T1", "inlet (blank = saturated liquid)"), ("P2", "outlet"), ("w", "work in"),
             ("eta", "isentropic efficiency")] + FLOW_REAL,
    "steam-turbine": [("P1", "inlet"), ("T1", "inlet"), ("x1", "inlet quality"), ("P2", "outlet"),
                      ("x2", "outlet quality"), ("w", "work out"), ("eta", "isentropic efficiency")] + FLOW_REAL,
    "boiler": [("P1", "pressure"), ("T1", "inlet"), ("T2", "outlet"), ("x2", "outlet quality"),
               ("q", "heat in"), ("mdot", "mass flow rate"), ("Qdot", "heat rate")],
    "condenser": [("P1", "pressure"), ("T1", "inlet"), ("x1", "inlet quality"),
                  ("T2", "outlet (blank = saturated liquid)"), ("q", "heat out"), ("mdot", "mass flow rate"),
                  ("Qdot", "heat rate")],
}
