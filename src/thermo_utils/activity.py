"""Excess Gibbs energy models."""

import math

import numpy as np
from thermo import NRTL, UNIQUAC, IdealSolution
from thermo.interaction_parameters import IPDB
from thermo.unifac import DOUFIP2016, DOUFSG, UFIP, UFSG, UNIFAC

from thermo_utils.units import R

MODELS = {
    "unifac": "UNIFAC (predictive, original)",
    "unifac-do": "Modified UNIFAC (Dortmund)",
    "nrtl": "NRTL (ChemSep parameters or yours)",
    "uniquac": "UNIQUAC (ChemSep parameters)",
    "wilson": "Wilson (binary, your Λ12 and Λ21)",
    "margules": "Margules two-parameter (binary, your A12 and A21)",
    "vanlaar": "van Laar (binary, your A'12 and A'21)",
    "ideal": "Ideal solution (γ = 1, Raoult)",
}


class GEModel:
    """Activity coefficients for a fixed set of components."""

    def __init__(self, subs, name="unifac", params=None):
        self.subs = subs
        self.name = name
        self.params = list(params or [])
        self.notes = []
        n = len(subs)
        if name in ("margules", "vanlaar", "wilson") and n != 2:
            raise ValueError(f"{name} is implemented for binary mixtures.")
        if name in ("margules", "vanlaar", "wilson") and len(self.params) != 2:
            raise ValueError(f"{name} needs two parameters, e.g. params=1.2,0.8")
        if name == "nrtl" and not self.params:
            cas = [s.cas for s in subs]
            self._b = IPDB.get_ip_asymmetric_matrix("ChemSep NRTL", cas, "bij")
            self._alpha = IPDB.get_ip_asymmetric_matrix("ChemSep NRTL", cas, "alphaij")
            if not any(any(r) for r in self._b):
                raise ValueError("No NRTL parameters in the databank for this mixture; give b12,b21,alpha or use UNIFAC.")
            self.notes.append("NRTL parameters from the ChemSep databank (τij = bij/T).")
        elif name == "nrtl":
            if n != 2 or len(self.params) not in (2, 3):
                raise ValueError("Give NRTL params as b12,b21[,alpha] in K (binary).")
            b12, b21 = self.params[:2]
            a = self.params[2] if len(self.params) == 3 else 0.3
            self._b = [[0, b12], [b21, 0]]
            self._alpha = [[0, a], [a, 0]]
        if name == "uniquac":
            cas = [s.cas for s in subs]
            self._b = IPDB.get_ip_asymmetric_matrix("ChemSep UNIQUAC", cas, "bij")
            if not any(any(r) for r in self._b):
                raise ValueError("No UNIQUAC parameters in the databank for this mixture; use UNIFAC.")
            self.notes.append("UNIQUAC parameters from the ChemSep databank.")
        if name.startswith("unifac"):
            for s in subs:
                if not s.unifac_groups:
                    raise ValueError(f"No UNIFAC groups for {s.name}.")

    def _thermo(self, T, xs):
        n = len(xs)
        if self.name == "nrtl":
            return NRTL(T=T, xs=xs, tau_bs=self._b, alpha_cs=self._alpha)
        if self.name == "uniquac":
            return UNIQUAC(T=T, xs=xs, rs=[s.r for s in self.subs], qs=[s.q for s in self.subs], tau_bs=self._b)
        if self.name == "unifac":
            return UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=[s.unifac_groups for s in self.subs],
                                         version=0, interaction_data=UFIP, subgroups=UFSG)
        if self.name == "unifac-do":
            groups = [s._constants.UNIFAC_Dortmund_groups[0] for s in self.subs]
            return UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=groups, version=1,
                                         interaction_data=DOUFIP2016, subgroups=DOUFSG)
        return IdealSolution(T=T, xs=xs) if n else None

    def ln_gammas(self, T, xs):
        xs = [max(float(x), 0.0) for x in xs]
        if self.name == "margules":
            A12, A21 = self.params
            x1, x2 = xs
            return [x2**2 * (A12 + 2 * (A21 - A12) * x1), x1**2 * (A21 + 2 * (A12 - A21) * x2)]
        if self.name == "vanlaar":
            A12, A21 = self.params
            x1, x2 = xs
            d = A12 * x1 + A21 * x2
            return [A12 * (A21 * x2 / d) ** 2, A21 * (A12 * x1 / d) ** 2]
        if self.name == "wilson":
            L12, L21 = self.params
            x1, x2 = xs
            s = L12 / (x1 + x2 * L12) - L21 / (x2 + x1 * L21)
            return [-math.log(x1 + x2 * L12) + x2 * s, -math.log(x2 + x1 * L21) - x1 * s]
        return [math.log(g) for g in self._thermo(T, _safe(xs)).gammas()]

    def gammas(self, T, xs):
        return [math.exp(v) for v in self.ln_gammas(T, xs)]

    def GE(self, T, xs):
        return R * T * sum(x * lg for x, lg in zip(xs, self.ln_gammas(T, xs)))

    def HE(self, T, xs, dT=0.01):
        g = lambda t: self.GE(t, xs) / t
        return -T**2 * (g(T + dT) - g(T - dT)) / (2 * dT)

    def SE(self, T, xs):
        return (self.HE(T, xs) - self.GE(T, xs)) / T


def _safe(xs):
    xs = [max(x, 1e-12) for x in xs]
    s = sum(xs)
    return [x / s for x in xs]


def binary_curve(model, T, n=101):
    x1 = np.linspace(0, 1, n)
    ln_g = np.array([model.ln_gammas(T, [max(x, 1e-9), max(1 - x, 1e-9)]) for x in x1])
    return x1, ln_g
