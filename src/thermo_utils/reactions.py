"""Reaction parsing and standard-state reaction properties ."""

import math
import re
from dataclasses import dataclass

from scipy.integrate import quad

from thermo_utils import substances
from thermo_utils.units import R

T0 = 298.15
P0 = 1e5

# Standard Gibbs energy of formation of liquids, J/mol
GF_LIQUID = {
    "7732-18-5": -237129.0, "67-56-1": -166270.0, "64-17-5": -174780.0, "64-19-7": -389900.0,
    "141-78-6": -332200.0,
}

_ARROWS = ("<=>", "<->", "->", "=>", "→", "⇌", "=")
_PHASE = re.compile(r"\((g|l|s|aq)\)\s*$", re.I)
_COEF = re.compile(r"^\s*(\d+(?:[.,]\d+)?|\d+/\d+)\s*(?:\s|\*|(?=[A-Z]))(.+)$")


@dataclass
class Species:
    sub: object
    nu: float
    phase: str

    @property
    def label(self):
        return f"{self.sub.formula}({self.phase})"

    def Hf(self):
        s = self.sub
        if self.phase == "g":
            substances.require(s, "Hf")
            return s.Hf
        if self.phase == "l":
            if s.phase_std == "l" and s.Hf_std is not None:
                return s.Hf_std
            return s.Hf - s.Hvap(T0)
        if s.phase_std == "s" and s.Hf_std is not None:
            return s.Hf_std
        raise ValueError(f"No solid-phase formation data for {s.name}")

    def Gf(self):
        s = self.sub
        substances.require(s, "Gf")
        if self.phase == "g":
            return s.Gf
        if self.phase == "l":
            if s.cas in GF_LIQUID:
                return GF_LIQUID[s.cas]
            return s.Gf + R * T0 * math.log(s.Psat(T0) / P0)
        raise ValueError(f"No solid-phase Gibbs energy data for {s.name}")

    def Cp(self, T):
        return self.sub.Cp_ig(T) if self.phase == "g" else self.sub.Cp_liq(T)


@dataclass
class Reaction:
    species: list
    text: str

    @property
    def dnu(self):
        return sum(sp.nu for sp in self.species if sp.phase == "g")

    def dH0(self):
        return sum(sp.nu * sp.Hf() for sp in self.species)

    def dG0(self):
        return sum(sp.nu * sp.Gf() for sp in self.species)

    def dCp(self, T):
        return sum(sp.nu * sp.Cp(T) for sp in self.species)

    def dH(self, T):
        return self.dH0() + quad(self.dCp, T0, T, limit=200)[0]

    def dS(self, T):
        dS0 = (self.dH0() - self.dG0()) / T0
        return dS0 + quad(lambda t: self.dCp(t) / t, T0, T, limit=200)[0]

    def dG(self, T):
        return self.dH(T) - T * self.dS(T)

    def K(self, T):
        return math.exp(-self.dG(T) / (R * T))

    def pretty(self):
        def side(items):
            return " + ".join(f"{_num(abs(sp.nu))}{sp.label}" for sp in items)
        reac = [sp for sp in self.species if sp.nu < 0]
        prod = [sp for sp in self.species if sp.nu > 0]
        return f"{side(reac)} -> {side(prod)}"


def _num(x):
    return "" if x == 1 else f"{x:g} "


def _parse_side(text, sign, default_phase):
    out = []
    for token in text.split(" + ") if " + " in text else text.split("+"):
        token = token.strip()
        if not token:
            continue
        phase = default_phase
        m = _PHASE.search(token)
        if m:
            phase = m.group(1).lower()
            token = token[: m.start()].strip()
        nu = 1.0
        m = _COEF.match(token)
        if m and not re.match(r"^\d+-", token):
            raw = m.group(1).replace(",", ".")
            nu = float(raw.split("/")[0]) / float(raw.split("/")[1]) if "/" in raw else float(raw)
            token = m.group(2).strip()
        sub = substances.get(token)
        if sub.cas == substances.AIR_CAS:
            raise ValueError("Air is a mixture: write O2 and N2 (about 3.76 mol N2 per mol O2).")
        out.append(Species(sub, sign * nu, phase))
    return out


def parse(text, default_phase="g"):
    """Parse 'CH4 + 2 O2 -> CO2 + 2 H2O(l)'."""
    for arrow in _ARROWS:
        if arrow in text:
            left, right = text.split(arrow, 1)
            break
    else:
        raise ValueError("Use '->' or '=' between reactants and products.")
    species = _parse_side(left, -1, default_phase) + _parse_side(right, 1, default_phase)
    _check_balance(species)
    return Reaction(species, text)


def _check_balance(species):
    from chemicals.elements import simple_formula_parser
    total = {}
    for sp in species:
        for el, n in simple_formula_parser(sp.sub.formula).items():
            total[el] = total.get(el, 0) + sp.nu * n
    bad = {el: v for el, v in total.items() if abs(v) > 1e-9}
    if bad:
        detail = ", ".join(f"{el}: {v:+g}" for el, v in bad.items())
        raise ValueError(f"Reaction is not balanced ({detail}).")
