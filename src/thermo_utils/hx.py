"""Heat exchanger: energy balance, LMTD, effectiveness-NTU and entropy generation, with real properties."""

import math

import numpy as np
from scipy.optimize import brentq

from thermo_utils import pure, substances
from thermo_utils import steplog as log
from thermo_utils.result import Result, row
from thermo_utils.steplog import q
from thermo_utils.units import num
from thermo_utils.units import (
    MASS_FLOW, POWER, PRESSURE as P, SPECIFIC_ENERGY as E, SPECIFIC_ENTROPY as S, TEMPERATURE as T, display,
    from_si, label,
)

ARRANGEMENTS = {"counter": "counter-flow", "parallel": "parallel-flow"}
P_DEFAULT = 101325.0


class _Stream:
    def __init__(self, key, sub, P_, ends, m, model):
        self.key, self.sub, self.m, self.model = key, sub, m, model
        self.M = sub.MW / 1000
        self.hot = key == "h"
        self.name = "Hot" if self.hot else "Cold"
        self.assumed_P = False
        self.P = P_ if P_ is not None else self._pressure(ends)
        self.st = {i: self._end(*ends[i]) for i in (1, 2)}

    def _pressure(self, ends):
        for t, x in ends.values():
            if t is not None and x is not None:
                return pure.saturation(self.sub, T=t, model=self.model)[1]
        self.assumed_P = True
        return P_DEFAULT

    def _end(self, t, x):
        if x is not None:
            if not 0 <= x <= 1:
                raise ValueError(f"{self.name} stream: the quality x must be between 0 and 1.")
            return pure.two_phase_state(self.sub, P=self.P, x=x, model=self.model)
        if t is not None:
            return pure.state(self.sub, t, self.P, self.model)
        return None

    def h(self, i):
        return self.st[i].H / self.M

    def s(self, i):
        return self.st[i].S / self.M

    def at_h(self, h):
        with log.muted():
            return pure.state_PH(self.sub, self.P, h * self.M, self.model)

    def duty(self):
        return self.m * abs(self.h(2) - self.h(1))

    def known(self):
        return self.m is not None and all(self.st.values())

    def missing(self):
        return [k for k, v in (("inlet", self.st[1]), ("outlet", self.st[2]), ("flow", self.m)) if v is None]

    def fill(self, Q):
        miss = self.missing()
        if len(miss) > 1:
            raise ValueError(f"{self.name} stream: {' and '.join(miss)} unknown; give one more value.")
        sign = -1 if self.hot else 1
        if miss == ["flow"]:
            self.m = Q / abs(self.h(2) - self.h(1))
            log.line("{}: ṁ = Q̇/|h2 − h1| = {}", self.name, q(self.m, MASS_FLOW))
        elif miss == ["outlet"]:
            self.st[2] = self.at_h(self.h(1) + sign * Q / self.m)
            log.line("{}: h2 = h1 {} Q̇/ṁ = {} -> T2 = {}", self.name, "−" if self.hot else "+", q(self.h(2), E),
                     q(self.st[2].T, T))
        elif miss == ["inlet"]:
            self.st[1] = self.at_h(self.h(2) - sign * Q / self.m)
            log.line("{}: h1 = h2 {} Q̇/ṁ = {} -> T1 = {}", self.name, "+" if self.hot else "−", q(self.h(1), E),
                     q(self.st[1].T, T))


# Temperature differences

def _ends(hot, cold, arrangement):
    th1, th2, tc1, tc2 = hot.st[1].T, hot.st[2].T, cold.st[1].T, cold.st[2].T
    if arrangement == "counter":
        return th1 - tc2, th2 - tc1
    return th1 - tc1, th2 - tc2


def _lmtd(d1, d2):
    if d1 <= 0 or d2 <= 0:
        raise ValueError(f"Temperature cross: the hot stream is not hotter than the cold one at both ends "
                         f"(ΔT = {d1:.4g} K and {d2:.4g} K).")
    return d1 if abs(d1 - d2) < 1e-9 * max(d1, d2) else (d1 - d2) / math.log(d1 / d2)


def _profile(hot, cold, Q, arrangement, n=31):
    """Hot and cold T along the exchanger, as functions of the heat already given by the hot stream."""
    xs = np.linspace(0.0, Q, n)
    th = [hot.at_h(hot.h(1) - x / hot.m).T for x in xs]
    if arrangement == "counter":
        tc = [cold.at_h(cold.h(2) - x / cold.m).T for x in xs]
    else:
        tc = [cold.at_h(cold.h(1) + x / cold.m).T for x in xs]
    return xs, th, tc


def _q_max(hot, cold):
    th1, tc1 = hot.st[1].T, cold.st[1].T
    with log.muted():
        qh = hot.m * (hot.h(1) - pure.state(hot.sub, tc1, hot.P, hot.model).H / hot.M)
        qc = cold.m * (pure.state(cold.sub, th1, cold.P, cold.model).H / cold.M - cold.h(1))
    return min(qh, qc)


def _rating(hot, cold, UA, arrangement):
    """Outlets from U·A and the inlets: solve U·A·LMTD(Q) = Q."""
    q_max = _q_max(hot, cold)

    def f(Q):
        hot.st[2], cold.st[2] = hot.at_h(hot.h(1) - Q / hot.m), cold.at_h(cold.h(1) + Q / cold.m)
        try:
            return UA * _lmtd(*_ends(hot, cold, arrangement)) - Q
        except ValueError:
            return -Q
    log.step("Rating: find Q̇ with U·A·LMTD(Q̇) = Q̇ (outlets from the energy balance), 0 < Q̇ < Q̇max = {}",
             q(q_max, POWER))
    Q = brentq(f, q_max * 1e-9, q_max * (1 - 1e-9), xtol=1e-6)
    f(Q)
    log.line("Q̇ = {}", q(Q, POWER))
    return Q


# Calculation

def heat_exchanger(hot=None, cold=None, flow="counter", model="auto", Ph=None, Th1=None, Th2=None, xh1=None,
                   xh2=None, mh=None, Pc=None, Tc1=None, Tc2=None, xc1=None, xc2=None, mc=None, Q=None, U=None,
                   A=None):
    hot_sub, cold_sub = hot or substances.get("water"), cold or substances.get("water")
    log.step("Streams ({}, no heat loss)", ARRANGEMENTS[flow])
    h = _Stream("h", hot_sub, Ph, {1: (Th1, xh1), 2: (Th2, xh2)}, mh, pure.resolve_model(hot_sub, model))
    c = _Stream("c", cold_sub, Pc, {1: (Tc1, xc1), 2: (Tc2, xc2)}, mc, pure.resolve_model(cold_sub, model))
    for s_ in (h, c):
        log.line("{} ({}): P = {}{}", s_.name, s_.sub.name, q(s_.P, P), " (assumed)" if s_.assumed_P else "")
    UA_given = U * A if U is not None and A is not None else None
    log.step("Heat duty Q̇")
    full = [s_ for s_ in (h, c) if s_.known()]
    source = None
    if Q is not None:
        source = "given"
    elif full:
        Q, source = full[0].duty(), f"{full[0].name.lower()} stream"
        log.line("From the {} stream: Q̇ = ṁ|h2 − h1| = {}", full[0].name.lower(), q(Q, POWER))
    elif UA_given is not None and all(s_.st[1] and s_.st[2] for s_ in (h, c)):
        Q, source = UA_given * _lmtd(*_ends(h, c, flow)), "U·A·LMTD"
        log.line("All four temperatures known: Q̇ = U·A·LMTD = {}", q(Q, POWER))
    elif UA_given is not None and h.st[1] and c.st[1] and h.m is not None and c.m is not None:
        Q, source = _rating(h, c, UA_given, flow), "rating"
    else:
        raise ValueError("Not enough data: give Q̇, or one stream completely (inlet, outlet, flow), "
                         "or U and A with the four temperatures or with both inlets and flows.")
    for s_ in (h, c):
        if not s_.known():
            s_.fill(Q)
    for s_ in (h, c):
        if abs(s_.duty() - Q) > 1e-3 * Q:
            raise ValueError(f"Energy balance not closed: the {s_.name.lower()} stream exchanges "
                             f"{s_.duty() / 1e3:.5g} kW, but Q̇ = {Q / 1e3:.5g} kW. Remove one value.")
    if h.h(2) >= h.h(1) or c.h(2) <= c.h(1):
        raise ValueError("The hot stream must cool down and the cold stream must heat up.")
    d1, d2 = _ends(h, c, flow)
    lmtd = _lmtd(d1, d2)
    UA = Q / lmtd
    if UA_given is not None and source not in ("U·A·LMTD", "rating") and abs(UA_given - UA) > 1e-3 * UA:
        raise ValueError(f"U·A = {UA_given:.5g} W/K does not match the duty: Q̇/LMTD = {UA:.5g} W/K. "
                         "Give U or A, not both.")
    xs, th, tc = _profile(h, c, Q, flow)
    dmin = min(a - b for a, b in zip(th, tc))
    if dmin <= 0:
        raise ValueError(f"Temperature cross inside the exchanger (minimum ΔT = {dmin:.4g} K): "
                         "the duty is not possible.")
    ch = Q / (h.st[1].T - h.st[2].T) if h.st[1].T - h.st[2].T > 1e-6 else math.inf
    cc = Q / (c.st[2].T - c.st[1].T) if c.st[2].T - c.st[1].T > 1e-6 else math.inf
    cmin, cmax = min(ch, cc), max(ch, cc)
    q_max = cmin * (h.st[1].T - c.st[1].T)
    s_gen = h.m * (h.s(2) - h.s(1)) + c.m * (c.s(2) - c.s(1))
    if log.on():
        log.step("LMTD ({})", ARRANGEMENTS[flow])
        log.line("ΔT1 = {} K, ΔT2 = {} K  ->  LMTD = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2) = {} K", num(d1), num(d2), num(lmtd))
        log.line("UA = Q̇/LMTD = {} W/K", num(UA))
        log.step("Effectiveness-NTU")
        log.line("Mean heat capacity rates C = Q̇/ΔT: Ch = {} W/K, Cc = {} W/K", num(ch), num(cc))
        log.line("Q̇max = Cmin(Th1 − Tc1) = {}   ε = Q̇/Q̇max = {}   NTU = UA/Cmin = {}", q(q_max, POWER),
                 num(Q / q_max), num(UA / cmin))
        log.step("Entropy generation")
        log.line("Ṡgen = ṁh(s2 − s1)h + ṁc(s2 − s1)c = {} W/K", num(s_gen))
    res = Result(f"Heat exchanger ({ARRANGEMENTS[flow]}) - {h.sub.name} / {c.sub.name}")
    res.equations = [
        "Q̇ = ṁh(h1 − h2)hot = ṁc(h2 − h1)cold     (real properties, phase change included)",
        "Counter-flow: ΔT1 = Th1 − Tc2, ΔT2 = Th2 − Tc1     Parallel: ΔT1 = Th1 − Tc1, ΔT2 = Th2 − Tc2",
        "LMTD = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2)     Q̇ = U·A·LMTD",
        "C = Q̇/ΔT (mean)     ε = Q̇/Q̇max     Q̇max = Cmin(Th1 − Tc1)     NTU = UA/Cmin     Cr = Cmin/Cmax",
        "Ṡgen = ṁh(s2 − s1)hot + ṁc(s2 − s1)cold ≥ 0",
    ]
    given = {k for k, v in dict(Th1=Th1, Th2=Th2, xh1=xh1, xh2=xh2, Tc1=Tc1, Tc2=Tc2, xc1=xc1, xc2=xc2).items()
             if v is not None}
    _table(res, h, c, given)
    rows = [
        row("Heat duty Q̇", Q, POWER), row("Hot flow ṁh", h.m, MASS_FLOW), row("Cold flow ṁc", c.m, MASS_FLOW),
        row("ΔT1, ΔT2 (ends)", f"{num(d1)} K, {num(d2)} K"), row("LMTD", f"{num(lmtd)} K"),
        row("Minimum ΔT (pinch)", f"{num(dmin)} K"), row("UA = Q̇/LMTD", f"{num(UA / 1e3)} kW/K"),
    ]
    given_rows = {"Heat duty Q̇": Q is not None and source == "given", "Hot flow ṁh": mh is not None,
                  "Cold flow ṁc": mc is not None}
    rows = [r + (given_rows.get(r[0], False),) for r in rows]
    if U is not None or A is not None:
        rows.append(("Overall coefficient U", f"{num(U if U is not None else UA / A)} W/(m²·K)", U is not None))
        rows.append(("Area A", f"{num(A if A is not None else UA / U)} m²", A is not None))
    rows += [
        row("Effectiveness ε = Q̇/Q̇max", num(Q / q_max)), row("Q̇max = Cmin(Th1 − Tc1)", q_max, POWER),
        row("NTU = UA/Cmin", num(UA / cmin)), row("Cr = Cmin/Cmax", num(cmin / cmax if cmax < math.inf else 0.0)),
        row("Entropy generation Ṡgen", f"{num(s_gen / 1e3)} kW/K"),
    ]
    res.section("Heat exchanger", rows)
    if dmin < min(d1, d2) - 0.5:
        res.notes.append(f"The smallest ΔT ({num(dmin)} K) is inside the exchanger, not at an end (phase change): "
                         "the LMTD from the end temperatures is only an estimate.")
    if math.inf in (ch, cc):
        res.notes.append("One stream changes phase at constant temperature: Cmax → ∞ and Cr = 0.")
    for s_ in (h, c):
        if s_.assumed_P:
            res.notes.append(f"{s_.name} stream pressure not given: 1 atm assumed (give P{s_.key} if not).")
    if source == "rating":
        res.notes.append("Outlets found from U·A and the inlets (rating).")
    res.diagram = _diagram(xs, th, tc, flow)
    res.plot = lambda: _plot(res.diagram, res.title)
    res.plot_name = "heat_exchanger"
    res.data = dict(Q=Q, mh=h.m, mc=c.m, LMTD=lmtd, UA=UA, eps=Q / q_max, NTU=UA / cmin, Sgen=s_gen,
                    Th2=h.st[2].T, Tc2=c.st[2].T, Th1=h.st[1].T, Tc1=c.st[1].T)
    return res


def _table(res, h, c, given):
    tu, eu, su = (label(display.unit_for(k)) for k in (T, E, S))
    headers = ["Stream", f"T\n{tu}", f"P\n{label(display.unit_for(P))}", "Phase", f"h\n{eu}", f"s\n{su}"]
    rows = []
    for s_ in (h, c):
        for i, where in ((1, "in"), (2, "out")):
            st = s_.st[i]
            k = f"T{s_.key}{i}"
            rows.append([(f"{s_.name} {where}", False), (_n(st.T, T), k in given or f"x{s_.key}{i}" in given),
                         (_n(st.P, P), False), (st.phase, f"x{s_.key}{i}" in given), (_n(s_.h(i), E), False),
                         (_n(s_.s(i), S), False)])
    res.tables.append(("Streams (h and s per kg)", headers, rows))


def _n(value, kind):
    return num(from_si(value, display.unit_for(kind), kind), si=value)


def _diagram(xs, th, tc, flow):
    hot = [(x, t) for x, t in zip(xs, th)]
    cold = [(x, t) for x, t in zip(xs, tc)]
    return dict(curves=[{"pts": hot, "style": "accent", "name": "hot"}, {"pts": cold, "style": "alt", "name": "cold"}],
                points=[hot[0], hot[-1], cold[0], cold[-1]], labels=["h1", "h2"] + (["c2", "c1"] if flow == "counter" else ["c1", "c2"]),
                axes=(("Q̇", POWER), ("T", T)), legend=[("accent", "hot stream"), ("alt", "cold stream")],
                title="Temperature profile")


def _plot(diagram, title):
    from thermo_utils import plots
    return plots.figure([diagram], title)


FORM = [
    (None, "Hot stream (T or quality x at each end)"), ("Th1", "inlet temperature"), ("xh1", "inlet quality"),
    ("Th2", "outlet temperature"), ("xh2", "outlet quality"), ("mh", "mass flow"), ("Ph", "pressure (default 1 atm)"),
    (None, "Cold stream"), ("Tc1", "inlet temperature"), ("xc1", "inlet quality"), ("Tc2", "outlet temperature"),
    ("xc2", "outlet quality"), ("mc", "mass flow"), ("Pc", "pressure (default 1 atm)"),
    (None, "Exchanger (optional)"), ("Q", "heat duty"), ("U", "overall coefficient, W/(m²·K)"), ("A", "area, m²"),
]
