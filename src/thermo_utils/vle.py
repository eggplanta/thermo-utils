"""Vapor-liquid equilibrium: gamma-phi and phi-phi (cubic EOS)."""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from thermo_utils import steplog as log
from thermo_utils import substances
from thermo_utils.activity import GEModel
from thermo_utils.steplog import q
from thermo_utils.units import num
from thermo_utils.units import MOLAR_VOLUME as V_, PRESSURE as P_, TEMPERATURE as T_, R


def _c(v):
    return ", ".join(num(x) for x in v)


def log_virial_B(subs, T):
    if not log.on():
        return
    B = cross_virial(subs, T)
    log.line("Bij from Pitzer with Tcij = √(Tci·Tcj), ωij = (ωi + ωj)/2, Zcij = (Zci + Zcj)/2,")
    log.line("  Vcij = [(Vci^⅓ + Vcj^⅓)/2]³, Pcij = Zcij·R·Tcij/Vcij; Bij = (RTcij/Pcij)(B0 + ωij·B1)")
    n = len(subs)
    log.table(["i", "j", "Bij"], [[subs[i].name, subs[j].name, q(B[i, j], V_)] for i in range(n) for j in range(i, n)])

METHODS = {
    "gamma-phi": "γ-Φ: activity model for the liquid + virial for the vapor",
    "modified-raoult": "Modified Raoult: yi P = xi γi Psat_i (ideal vapor)",
    "raoult": "Raoult's law: ideal liquid and vapor",
    "phi-phi": "φ-φ: Peng-Robinson for both phases (ChemSep kij), good for hydrocarbons/gases at high P",
}


# Virial mixture

def cross_virial(subs, T, kij=None):
    n = len(subs)
    B = np.zeros((n, n))
    for i, a in enumerate(subs):
        for j, b in enumerate(subs):
            k = kij[i][j] if kij else 0.0
            w = (a.omega + b.omega) / 2
            Tc = math.sqrt(a.Tc * b.Tc) * (1 - k)
            Zc = (a.Zc + b.Zc) / 2
            Vc = ((a.Vc ** (1 / 3) + b.Vc ** (1 / 3)) / 2) ** 3
            Pc = Zc * R * Tc / Vc
            Tr = T / Tc
            B0 = 0.083 - 0.422 / Tr**1.6
            B1 = 0.139 - 0.172 / Tr**4.2
            B[i, j] = R * Tc / Pc * (B0 + w * B1)
    return B


def virial_phis(subs, T, P, ys):
    """Fugacity coefficients of species in a gas mixture (virial)."""
    B = cross_virial(subs, T)
    n = len(subs)
    d = np.array([[2 * B[i, k] - B[i, i] - B[k, k] for i in range(n)] for k in range(n)])
    ys = np.asarray(ys)
    out = []
    for k in range(n):
        s = sum(ys[i] * ys[j] * (2 * d[i, k] - d[i, j]) for i in range(n) for j in range(n))
        out.append(math.exp(P / (R * T) * (B[k, k] + 0.5 * s)))
    Bmix = float(ys @ B @ ys)
    return out, Bmix


def big_phi(subs, T, P, ys, Psats):
    """Φi = φ̂i / φi_sat * Poynting ignored."""
    B = cross_virial(subs, T)
    n = len(subs)
    d = np.array([[2 * B[j, i] - B[j, j] - B[i, i] for j in range(n)] for i in range(n)])
    out = []
    for i in range(n):
        s = sum(ys[j] * ys[k] * (2 * d[i, j] - d[j, k]) for j in range(n) for k in range(n))
        out.append(math.exp((B[i, i] * (P - Psats[i]) + 0.5 * P * s) / (R * T)))
    return out


# Gamma-phi solver

@dataclass
class VLEResult:
    T: float
    P: float
    xs: list
    ys: list
    gammas: list = field(default_factory=list)
    phis: list = field(default_factory=list)
    Psats: list = field(default_factory=list)
    VF: float = None


class GammaPhi:
    def __init__(self, subs, method="gamma-phi", model="unifac", params=None):
        self.subs = subs
        self.method = method
        self.ge = GEModel(subs, "ideal" if method == "raoult" else model, params)

    def psats(self, T):
        return [s.Psat(T) for s in self.subs]

    def _log_psats(self, T, ps):
        for s, p in zip(self.subs, ps):
            src = "reference EOS" if s.coolprop and substances._in_coolprop_range(s.coolprop, T) else \
                getattr(s._correlations.VaporPressures[0], "method", "correlation")
            log.line("Psat {} = {}   ({})", s.name, q(p, P_), src)

    def _log_model(self, T, xs, ys=None):
        log.line("Method: {}", METHODS[self.method])
        if self.method == "raoult":
            log.line("Raoult: γi = 1, Φi = 1")
        else:
            self.ge.explain(T, xs)
        if self.method == "gamma-phi":
            log_virial_B(self.subs, T)
            log.line("Φi = exp{[Bii(P − Psat_i) + ½P ΣjΣk yj yk(2δji − δjk)]/RT}, δji = 2Bji − Bjj − Bii")
        else:
            log.line("Ideal vapor: Φi = 1")

    def _gam(self, T, xs):
        return self.ge.gammas(T, xs)

    def _Phi(self, T, P, ys, ps):
        if self.method != "gamma-phi":
            return [1.0] * len(ys)
        return big_phi(self.subs, T, P, ys, ps)

    def bubble_P(self, T, xs):
        ps, g = self.psats(T), self._gam(T, xs)
        Phi = [1.0] * len(xs)
        rows = []
        for it in range(100):
            P = sum(x * gi * p / f for x, gi, p, f in zip(xs, g, ps, Phi))
            ys = [x * gi * p / (f * P) for x, gi, p, f in zip(xs, g, ps, Phi)]
            new = self._Phi(T, P, ys, ps)
            rows.append([it + 1, q(P, P_), _c(Phi), _c(ys)])
            if max(abs(a - b) for a, b in zip(new, Phi)) < 1e-12:
                break
            Phi = new
        if log.on():
            log.step("Bubble P at T = {}, x = {}", q(T, T_), _c(xs))
            self._log_psats(T, ps)
            self._log_model(T, xs)
            log.line("P = Σ xi γi Psat_i/Φi,  yi = xi γi Psat_i/(Φi P); start with Φ = 1, update Φ with the new y and P")
            log.table(["it", "P", "Φ used", "y"], rows)
            terms = [x * gi * p / f for x, gi, p, f in zip(xs, g, ps, Phi)]
            log.line("Final: P = {} = {}", " + ".join(f"{num(x)}·{num(gi)}·{num(p)}/{num(f)}"
                                                     for x, gi, p, f in zip(xs, g, ps, Phi)) + " Pa", q(sum(terms), P_))
            log.line("y = {}   K = y/x = {}", _c(ys), _c([y / x if x else float("nan") for x, y in zip(xs, ys)]))
        return VLEResult(T, P, list(xs), ys, g, Phi, ps)

    def dew_P(self, T, ys):
        ps = self.psats(T)
        Phi, g = [1.0] * len(ys), [1.0] * len(ys)
        rows = []
        for it in range(200):
            P = 1 / sum(y * f / (gi * p) for y, f, gi, p in zip(ys, Phi, g, ps))
            xs = [y * f * P / (gi * p) for y, f, gi, p in zip(ys, Phi, g, ps)]
            s = sum(xs)
            xs = [x / s for x in xs]
            rows.append([it + 1, q(P, P_), _c(g), _c(Phi), _c(xs)])
            g_new, Phi_new = self._gam(T, xs), self._Phi(T, P, ys, ps)
            if max(abs(a - b) for a, b in zip(g_new + Phi_new, g + Phi)) < 1e-12:
                break
            g, Phi = g_new, Phi_new
        if log.on():
            log.step("Dew P at T = {}, y = {}", q(T, T_), _c(ys))
            self._log_psats(T, ps)
            self._log_model(T, xs)
            log.line("P = 1/Σ(yi Φi/(γi Psat_i)),  xi = yi Φi P/(γi Psat_i); start with γ = Φ = 1, update with the new x")
            log.table(["it", "P", "γ used", "Φ used", "x"], rows)
            log.line("Final: P = {}, x = {}, K = y/x = {}", q(P, P_), _c(xs), _c([y / x for x, y in zip(xs, ys)]))
        return VLEResult(T, P, xs, list(ys), g, Phi, ps)

    def _T_bracket(self, P):
        Ts = [s.Tsat(P) for s in self.subs]
        return min(Ts) - 40, max(Ts) + 40

    def bubble_T(self, P, xs):
        return self._solve_T(P, xs, self.bubble_P, "Bubble")

    def dew_T(self, P, ys):
        return self._solve_T(P, ys, self.dew_P, "Dew")

    def _solve_T(self, P, zs, at_T, name):
        lo, hi = self._T_bracket(P)
        tries = []

        def f(t):
            p = at_T(t, zs).P
            tries.append([len(tries) + 1, q(t, T_), q(p, P_)])
            return p - P

        with log.muted():
            T = brentq(f, lo, hi, xtol=1e-10)
        if log.on():
            log.step("{} T at P = {}", name, q(P, P_))
            log.line("Tsat of each pure species at P: {}", ", ".join(f"{s.name} {num(s.Tsat(P))} K" for s in self.subs))
            log.line("Find T where {} P(T) = P: Brent's method between {} and {}", name.lower(), q(lo, T_), q(hi, T_))
            log.table(["it", "T", f"{name.lower()} P(T)"], tries)
            log.line("T = {}; full calculation at this T:", q(T, T_))
        return at_T(T, zs)

    def flash(self, T, P, zs):
        with log.muted():
            bub, dew = self.bubble_P(T, zs).P, self.dew_P(T, zs).P
        if P >= bub:
            log.step("Flash at T = {}, P = {}", q(T, T_), q(P, P_))
            log.line("P ≥ bubble P(z) = {}: all liquid, V/F = 0", q(bub, P_))
            return VLEResult(T, P, list(zs), [None] * len(zs), VF=0.0)
        if P <= dew:
            log.step("Flash at T = {}, P = {}", q(T, T_), q(P, P_))
            log.line("P ≤ dew P(z) = {}: all vapor, V/F = 1", q(dew, P_))
            return VLEResult(T, P, [None] * len(zs), list(zs), VF=1.0)
        xs, ys = list(zs), list(zs)
        ps = self.psats(T)
        V = (bub - P) / (bub - dew)
        rows = []
        if log.on():
            log.step("Flash at T = {}, P = {}, z = {}", q(T, T_), q(P, P_), _c(zs))
            log.line("Bubble P(z) = {}, dew P(z) = {}: dew P < P < bubble P, so two phases", q(bub, P_), q(dew, P_))
            self._log_psats(T, ps)
            log.line("Ki = γi Psat_i/(Φi P); Rachford-Rice: Σ zi(Ki − 1)/(1 + V(Ki − 1)) = 0 for V")
            log.line("xi = zi/(1 + V(Ki − 1)), yi = Ki xi; γ and Φ updated with the new x, y until nothing changes")
        for it in range(300):
            g, Phi = self._gam(T, xs), self._Phi(T, P, ys, ps)
            K = [gi * p / (f * P) for gi, p, f in zip(g, ps, Phi)]
            rr = lambda v: sum(z * (k - 1) / (1 + v * (k - 1)) for z, k in zip(zs, K))
            V = brentq(rr, 1e-12, 1 - 1e-12)
            new_x = [z / (1 + V * (k - 1)) for z, k in zip(zs, K)]
            new_y = [k * x for k, x in zip(K, new_x)]
            rows.append([it + 1, _c(K), num(V), _c(new_x), _c(new_y)])
            if max(abs(a - b) for a, b in zip(new_x + new_y, xs + ys)) < 1e-12:
                xs, ys = new_x, new_y
                break
            xs, ys = new_x, new_y
        if log.on():
            log.table(["it", "K", "V/F", "x", "y"], rows)
            self._log_model(T, xs)
            log.line("Result: V/F = {}, x = {}, y = {}", num(V), _c(xs), _c(ys))
            log.line("Check, mass balance: zi = (1 − V)xi + V·yi -> {}", _c([(1 - V) * x + V * y for x, y in zip(xs, ys)]))
        return VLEResult(T, P, xs, ys, g, Phi, ps, V)


# Phi-phi solver (thermo FlashVL with PR)

class PhiPhi:
    def __init__(self, subs):
        from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL, PRMIX
        from thermo.interaction_parameters import IPDB
        self.subs = subs
        cas = [s.cas for s in subs]
        if substances.AIR_CAS in cas:
            raise ValueError("Air is pseudo-pure and cannot enter a φ-φ flash: use nitrogen and oxygen.")
        consts, corrs = ChemicalConstantsPackage.from_IDs(cas)
        kij = IPDB.get_ip_asymmetric_matrix("ChemSep PR", cas, "kij")
        kw = dict(Tcs=consts.Tcs, Pcs=consts.Pcs, omegas=consts.omegas, kijs=kij)
        self.kij = kij
        self.gas = CEOSGas(PRMIX, kw, HeatCapacityGases=corrs.HeatCapacityGases)
        self.liq = CEOSLiquid(PRMIX, kw, HeatCapacityGases=corrs.HeatCapacityGases)
        self.flasher = FlashVL(consts, corrs, liquid=self.liq, gas=self.gas)

    def _wrap(self, r, xs=None, ys=None):
        gas = r.gas.zs if r.gas is not None else ys
        liq = r.liquid0.zs if r.liquid0 is not None else xs
        phis = r.gas.phis() if r.gas is not None else []
        if log.on():
            log.step("φ-φ equilibrium with Peng-Robinson for both phases")
            log.line("Pure PR: a_i = 0.45724 R²Tc²/Pc·α_i(T), b_i = 0.07780 RTc/Pc")
            log.line("Mixing rules: aα = ΣiΣj zi zj √(aαi aαj)(1 − kij), b = Σ zi bi")
            n = len(self.subs)
            log.table(["i", "j", "kij (ChemSep)"], [[self.subs[i].name, self.subs[j].name, self.kij[i][j]]
                                                    for i in range(n) for j in range(i + 1, n)])
            log.line("Equilibrium: xi φ̂i_liq = yi φ̂i_vap, solved by thermo's FlashVL (successive substitution + Newton)")
            log.line("Result: T = {}, P = {}, V/F = {}", q(r.T, T_), q(r.P, P_), num(r.VF))
            if liq:
                log.line("x = {}", _c(liq))
            if gas:
                log.line("y = {}", _c(gas))
            if r.liquid0 is not None:
                log.line("φ̂ liquid = {}   Z liquid = {}", _c(r.liquid0.phis()), num(r.liquid0.Z()))
            if r.gas is not None:
                log.line("φ̂ vapor = {}   Z vapor = {}", _c(phis), num(r.gas.Z()))
            if liq and gas:
                log.line("K = y/x = {}", _c([y / x for x, y in zip(liq, gas)]))
        return VLEResult(r.T, r.P, list(liq) if liq else None, list(gas) if gas else None, [], phis, [], r.VF)

    def bubble_P(self, T, xs):
        return self._wrap(self.flasher.flash(T=T, VF=0, zs=xs), xs=xs)

    def dew_P(self, T, ys):
        return self._wrap(self.flasher.flash(T=T, VF=1, zs=ys), ys=ys)

    def bubble_T(self, P, xs):
        return self._wrap(self.flasher.flash(P=P, VF=0, zs=xs), xs=xs)

    def dew_T(self, P, ys):
        return self._wrap(self.flasher.flash(P=P, VF=1, zs=ys), ys=ys)

    def flash(self, T, P, zs):
        r = self.flasher.flash(T=T, P=P, zs=zs)
        res = self._wrap(r)
        if r.VF in (0, 1) or r.gas is None or r.liquid0 is None:
            res.xs = list(zs) if r.VF == 0 else [None] * len(zs)
            res.ys = list(zs) if r.VF == 1 else [None] * len(zs)
        return res


def solver(subs, method="gamma-phi", model="unifac", params=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Options: {', '.join(METHODS)}")
    if method == "phi-phi":
        return PhiPhi(subs)
    return GammaPhi(subs, method, model, params)
