"""Vapor-liquid equilibrium: gamma-phi and phi-phi (cubic EOS)."""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from thermo_utils.activity import GEModel
from thermo_utils.units import R

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

    def _gam(self, T, xs):
        return self.ge.gammas(T, xs)

    def _Phi(self, T, P, ys, ps):
        if self.method != "gamma-phi":
            return [1.0] * len(ys)
        return big_phi(self.subs, T, P, ys, ps)

    def bubble_P(self, T, xs):
        ps, g = self.psats(T), self._gam(T, xs)
        Phi = [1.0] * len(xs)
        for _ in range(100):
            P = sum(x * gi * p / f for x, gi, p, f in zip(xs, g, ps, Phi))
            ys = [x * gi * p / (f * P) for x, gi, p, f in zip(xs, g, ps, Phi)]
            new = self._Phi(T, P, ys, ps)
            if max(abs(a - b) for a, b in zip(new, Phi)) < 1e-12:
                break
            Phi = new
        return VLEResult(T, P, list(xs), ys, g, Phi, ps)

    def dew_P(self, T, ys):
        ps = self.psats(T)
        Phi, g = [1.0] * len(ys), [1.0] * len(ys)
        for _ in range(200):
            P = 1 / sum(y * f / (gi * p) for y, f, gi, p in zip(ys, Phi, g, ps))
            xs = [y * f * P / (gi * p) for y, f, gi, p in zip(ys, Phi, g, ps)]
            s = sum(xs)
            xs = [x / s for x in xs]
            g_new, Phi_new = self._gam(T, xs), self._Phi(T, P, ys, ps)
            if max(abs(a - b) for a, b in zip(g_new + Phi_new, g + Phi)) < 1e-12:
                break
            g, Phi = g_new, Phi_new
        return VLEResult(T, P, xs, list(ys), g, Phi, ps)

    def _T_bracket(self, P):
        Ts = [s.Tsat(P) for s in self.subs]
        return min(Ts) - 40, max(Ts) + 40

    def bubble_T(self, P, xs):
        lo, hi = self._T_bracket(P)
        T = brentq(lambda t: self.bubble_P(t, xs).P - P, lo, hi, xtol=1e-10)
        return self.bubble_P(T, xs)

    def dew_T(self, P, ys):
        lo, hi = self._T_bracket(P)
        T = brentq(lambda t: self.dew_P(t, ys).P - P, lo, hi, xtol=1e-10)
        return self.dew_P(T, ys)

    def flash(self, T, P, zs):
        bub, dew = self.bubble_P(T, zs).P, self.dew_P(T, zs).P
        if P >= bub:
            return VLEResult(T, P, list(zs), [None] * len(zs), VF=0.0)
        if P <= dew:
            return VLEResult(T, P, [None] * len(zs), list(zs), VF=1.0)
        xs, ys = list(zs), list(zs)
        ps = self.psats(T)
        V = (bub - P) / (bub - dew)
        for _ in range(300):
            g, Phi = self._gam(T, xs), self._Phi(T, P, ys, ps)
            K = [gi * p / (f * P) for gi, p, f in zip(g, ps, Phi)]
            rr = lambda v: sum(z * (k - 1) / (1 + v * (k - 1)) for z, k in zip(zs, K))
            V = brentq(rr, 1e-12, 1 - 1e-12)
            new_x = [z / (1 + V * (k - 1)) for z, k in zip(zs, K)]
            new_y = [k * x for k, x in zip(K, new_x)]
            if max(abs(a - b) for a, b in zip(new_x + new_y, xs + ys)) < 1e-12:
                xs, ys = new_x, new_y
                break
            xs, ys = new_x, new_y
        return VLEResult(T, P, xs, ys, g, Phi, ps, V)


# Phi-phi solver (thermo FlashVL with PR)

class PhiPhi:
    def __init__(self, subs):
        from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, FlashVL, PRMIX
        from thermo.interaction_parameters import IPDB
        self.subs = subs
        cas = [s.cas for s in subs]
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
