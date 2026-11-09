from dataclasses import dataclass, field
from functools import lru_cache

import CoolProp.CoolProp as CP
from chemicals.identifiers import search_chemical
from thermo import ChemicalConstantsPackage

FORMULA_ALIASES = {
    "c2h5oh": "ethanol", "ch3ch2oh": "ethanol", "ch3oh": "methanol", "ch3cooh": "acetic acid",
    "hcooh": "formic acid", "ch3coch3": "acetone", "c2h4": "ethylene", "c3h6": "propylene",
    "c6h5ch3": "toluene", "ch3cooc2h5": "ethyl acetate", "ch3och3": "dimethyl ether",
}


@lru_cache(maxsize=1)
def _coolprop_by_cas():
    table = {}
    for name in CP.get_global_param_string("FluidsList").split(","):
        try:
            table[CP.get_fluid_param_string(name, "CAS")] = name
        except ValueError:
            pass
    return table


@dataclass(eq=False)
class Substance:
    name: str
    cas: str
    formula: str
    MW: float
    Tc: float
    Pc: float
    Vc: float
    Zc: float
    omega: float
    Tb: float
    Tm: float
    Hfus: float
    Hf: float
    Gf: float
    S0: float
    Hf_std: float
    phase_std: str
    dipole: float
    r: float
    q: float
    unifac_groups: dict
    coolprop: str | None
    _constants: object = field(repr=False, default=None)
    _correlations: object = field(repr=False, default=None)

    @property
    def Cp_ig_model(self):
        return self._correlations.HeatCapacityGases[0]

    @property
    def Cp_liq_model(self):
        return self._correlations.HeatCapacityLiquids[0]

    def Cp_ig(self, T):
        return self.Cp_ig_model(T)

    def H_ig(self, T1, T2):
        return self.Cp_ig_model.T_dependent_property_integral(T1, T2)

    def S_ig(self, T1, T2):
        return self.Cp_ig_model.T_dependent_property_integral_over_T(T1, T2)

    def Cp_liq(self, T):
        return self.Cp_liq_model(T)

    def H_liq(self, T1, T2):
        return self.Cp_liq_model.T_dependent_property_integral(T1, T2)

    def S_liq(self, T1, T2):
        return self.Cp_liq_model.T_dependent_property_integral_over_T(T1, T2)

    def Psat(self, T):
        if self.coolprop and _in_coolprop_range(self.coolprop, T):
            return CP.PropsSI("P", "T", T, "Q", 0, self.coolprop)
        return self._correlations.VaporPressures[0](T)

    def Tsat(self, P):
        if self.coolprop:
            try:
                return CP.PropsSI("T", "P", P, "Q", 0, self.coolprop)
            except ValueError:
                pass
        return self._correlations.VaporPressures[0].solve_property(P)

    def Hvap(self, T):
        if self.coolprop and _in_coolprop_range(self.coolprop, T):
            h_v = CP.PropsSI("Hmolar", "T", T, "Q", 1, self.coolprop)
            h_l = CP.PropsSI("Hmolar", "T", T, "Q", 0, self.coolprop)
            return h_v - h_l
        return self._correlations.EnthalpyVaporizations[0](T)

    def Vliq(self, T, P=101325.0):
        return self._correlations.VolumeLiquids[0](T, P)

    @property
    def data_available(self):
        return {k: getattr(self, k) is not None for k in ("Tc", "Pc", "omega", "Hf", "Gf")}


def _in_coolprop_range(fluid, T):
    try:
        t_min = CP.PropsSI("Tmin", fluid)
        t_c = CP.PropsSI("Tcrit", fluid)
        return t_min <= T < t_c
    except ValueError:
        return False


def resolve_id(query):
    query = query.strip()
    return FORMULA_ALIASES.get(query.lower(), query)


@lru_cache(maxsize=512)
def get(query):
    """Find a substance by name, CAS or formula."""
    ident = resolve_id(query)
    try:
        meta = search_chemical(ident)
    except ValueError:
        raise ValueError(f"Substance '{query}' not found. Try the English name or the CAS number.") from None
    return _build(meta.CASs)


@lru_cache(maxsize=256)
def _build(cas):
    meta = search_chemical(cas)
    consts, corrs = ChemicalConstantsPackage.from_IDs([meta.CASs])
    first = lambda attr: (getattr(consts, attr, None) or [None])[0]
    groups = first("UNIFAC_groups") or {}
    return Substance(
        name=meta.common_name, cas=meta.CASs, formula=meta.formula, MW=first("MWs"),
        Tc=first("Tcs"), Pc=first("Pcs"), Vc=first("Vcs"), Zc=first("Zcs"), omega=first("omegas"),
        Tb=first("Tbs"), Tm=first("Tms"), Hfus=first("Hfus_Tms"), Hf=first("Hfgs"), Gf=first("Gfgs"),
        S0=first("S0gs"), Hf_std=first("Hf_STPs"), phase_std=first("phase_STPs"), dipole=first("dipoles"),
        r=first("UNIFAC_Rs"), q=first("UNIFAC_Qs"), unifac_groups=dict(groups),
        coolprop=_coolprop_by_cas().get(meta.CASs), _constants=consts, _correlations=corrs,
    )


def require(sub, *attrs):
    missing = [a for a in attrs if getattr(sub, a) is None]
    if missing:
        raise ValueError(f"No data for {', '.join(missing)} of {sub.name}")
