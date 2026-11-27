import math
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace

from scipy.integrate import quad
from chemicals.identifiers import search_chemical
from thermo import ChemicalConstantsPackage


class _Lazy:
    """Imports a module on first attribute access."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        import importlib
        return getattr(importlib.import_module(self._name), attr)


CoolProp = _Lazy("CoolProp")
CP = _Lazy("CoolProp.CoolProp")

FORMULA_ALIASES = {
    "c2h5oh": "ethanol", "ch3ch2oh": "ethanol", "ch3oh": "methanol", "ch3cooh": "acetic acid",
    "hcooh": "formic acid", "ch3coch3": "acetone", "c2h4": "ethylene", "c3h6": "propylene",
    "c6h5ch3": "toluene", "ch3cooc2h5": "ethyl acetate", "ch3och3": "dimethyl ether",
    # usual compound for isomeric formulas
    "c2h6o": "ethanol", "c2h4o2": "acetic acid", "c2h4o": "acetaldehyde", "c3h6o": "acetone",
    "c4h8o2": "ethyl acetate", "c2h6o2": "ethylene glycol", "c3h8o2": "propylene glycol", "c3h4": "propyne",
    "c3h4o": "acrolein", "c3h4o2": "acrylic acid", "c3h7no": "dimethylformamide", "c2h6os": "dimethyl sulfoxide",
    "c4h6": "1,3-butadiene", "c4h6o2": "vinyl acetate", "c5h8o2": "methyl methacrylate", "c6h6": "benzene",
    "c7h8": "toluene", "c6h12": "cyclohexane", "c7h14": "methylcyclohexane", "c6h10o": "cyclohexanone",
    "c6h12o": "cyclohexanol", "c6h7n": "aniline", "c6h15n": "triethylamine", "c6h5no2": "nitrobenzene",
    "c7h6o2": "benzoic acid", "c7h6o3": "salicylic acid", "c8h8": "styrene", "c9h12": "cumene",
    "c12h10": "biphenyl", "c14h10": "anthracene", "c10h8": "naphthalene", "c12h22o11": "sucrose",
    "c6h12o6": "glucose", "c3h6o3": "lactic acid", "c2h2f4": "1,1,1,2-tetrafluoroethane", "c2h7n": "ethylamine",
    "h2": "hydrogen", "c4h10": "butane", "c5h12": "pentane", "c6h14": "hexane", "c7h16": "heptane",
    "c8h18": "octane", "c10h22": "decane",
}
NOTICES = {}


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
    note: str | None = None

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


# Air (pseudo-pure fluid)

AIR_CAS = "132259-10-0"
AIR_NAMES = ("air", "dry air", AIR_CAS)
AIR_COMPOSITION = {"nitrogen": 0.7812, "oxygen": 0.2096, "argon": 0.0092}
R_GAS = 8.314462618


class _IdealGasCp:
    """Ideal-gas Cp, H and S of a CoolProp fluid (Cp0 of its reference EOS)."""

    def __init__(self, fluid):
        self.AS = CoolProp.AbstractState("HEOS", fluid)

    def _at(self, T):
        self.AS.update(CoolProp.DmolarT_INPUTS, 1e-6, T)
        return self.AS

    def __call__(self, T):
        return self._at(T).cp0molar()

    def _h(self, T):
        a = self._at(T)
        return a.hmolar() - a.hmolar_residual()

    def _s(self, T):
        a = self._at(T)
        return a.smolar() - a.smolar_residual()

    def T_dependent_property_integral(self, T1, T2):
        return self._h(T2) - self._h(T1)

    def T_dependent_property_integral_over_T(self, T1, T2):
        return self._s(T2) - self._s(T1) + R_GAS * math.log(T2 / T1)


class _SaturatedLiquidCp:
    def __init__(self, fluid):
        self.fluid = fluid

    def __call__(self, T):
        return CP.PropsSI("Cpmolar", "T", T, "Q", 0, self.fluid)

    def T_dependent_property_integral(self, T1, T2):
        return quad(self, T1, T2)[0]

    def T_dependent_property_integral_over_T(self, T1, T2):
        return quad(lambda t: self(t) / t, T1, T2)[0]


@dataclass(eq=False)
class _PseudoPure(Substance):
    def Psat(self, T):
        self._check_range(T)
        return CP.PropsSI("P", "T", T, "Q", 0, self.coolprop)

    def Tsat(self, P):
        return CP.PropsSI("T", "P", P, "Q", 0, self.coolprop)

    def Hvap(self, T):
        self._check_range(T)
        return (CP.PropsSI("Hmolar", "T", T, "Q", 1, self.coolprop)
                - CP.PropsSI("Hmolar", "T", T, "Q", 0, self.coolprop))

    def Vliq(self, T, P=101325.0):
        self._check_range(T)
        return 1 / CP.PropsSI("Dmolar", "T", T, "Q", 0, self.coolprop)

    def _check_range(self, T):
        if not _in_coolprop_range(self.coolprop, T):
            raise ValueError(f"{self.name}: no liquid at {T:.5g} K (triple point {self.Tm:.5g} K, "
                             f"critical point {self.Tc:.5g} K).")


@lru_cache(maxsize=1)
def air():
    fluid = "Air"
    Tc, Pc = CP.PropsSI("Tcrit", fluid), CP.PropsSI("pcrit", fluid)
    Vc = 1 / CP.PropsSI("rhomolar_critical", fluid)
    empty = [None]
    corrs = SimpleNamespace(
        HeatCapacityGases=[_IdealGasCp(fluid)], HeatCapacityLiquids=[_SaturatedLiquidCp(fluid)],
        VaporPressures=empty, EnthalpyVaporizations=empty, VolumeLiquids=empty, SurfaceTensions=empty,
    )
    consts = SimpleNamespace(Parachors=empty, UNIFAC_Dortmund_groups=[{}], Hcs=empty, Hcs_lower=empty)
    return _PseudoPure(
        name="air", cas=AIR_CAS, formula="N2/O2/Ar", MW=CP.PropsSI("molar_mass", fluid) * 1000,
        Tc=Tc, Pc=Pc, Vc=Vc, Zc=Pc * Vc / (R_GAS * Tc), omega=CP.PropsSI("acentric", fluid),
        Tb=CP.PropsSI("T", "P", 101325, "Q", 0, fluid), Tm=CP.PropsSI("Ttriple", fluid), Hfus=None,
        Hf=0.0, Gf=None, S0=None, Hf_std=None, phase_std="g", dipole=0.0, r=None, q=None, unifac_groups={},
        coolprop=fluid, _constants=consts, _correlations=corrs,
        note=("Air is treated as a pseudo-pure fluid (N2 0.7812, O2 0.2096, Ar 0.0092; reference EOS of "
              "Lemmon et al., 2000). Below the critical point, liquid and vapor values are approximate: "
              "the bubble and dew points of real air differ by about 3 K."),
    )


def resolve_id(query):
    query = query.strip()
    return FORMULA_ALIASES.get(query.lower(), query)


def get(query):
    """Find a substance by name, CAS or formula."""
    sub, notice = _lookup(query)
    if notice:
        NOTICES[query.strip()] = notice
    return sub


@lru_cache(maxsize=512)
def _lookup(query):
    ident = resolve_id(query)
    if ident.lower() in AIR_NAMES:
        return air(), None
    try:
        meta = search_chemical(ident)
    except ValueError:
        raise ValueError(f"Substance '{query}' not found. Try the English name or the CAS number.") from None
    return _build(meta.CASs), _isomer_notice(ident, meta)


def _isomer_notice(ident, meta):
    if ident != meta.formula:
        return None
    from chemicals.identifiers import pubchem_db
    others = sorted({m.common_name for m in pubchem_db.CAS_index.values()
                     if m.formula == meta.formula and m.CASs != meta.CASs}, key=len)
    if not others:
        return None
    return (f"{ident} matches {len(others) + 1} compounds; assumed {meta.common_name}. "
            f"Type the name for another one (e.g. {', '.join(others[:3])}).")


@lru_cache(maxsize=256)
def _build(cas):
    meta = search_chemical(cas)
    consts, corrs = ChemicalConstantsPackage.from_IDs([meta.CASs])
    first = lambda attr: (getattr(consts, attr, None) or [None])[0]
    groups = first("UNIFAC_groups") or {}
    coolprop = _coolprop_by_cas().get(meta.CASs)
    return Substance(
        name=coolprop if coolprop and coolprop[0] == "R" and coolprop[1:2].isdigit() else meta.common_name, cas=meta.CASs, formula=meta.formula, MW=first("MWs"),
        Tc=first("Tcs"), Pc=first("Pcs"), Vc=first("Vcs"), Zc=first("Zcs"), omega=first("omegas"),
        Tb=first("Tbs"), Tm=first("Tms"), Hfus=first("Hfus_Tms"), Hf=first("Hfgs"), Gf=first("Gfgs"),
        S0=first("S0gs"), Hf_std=first("Hf_STPs"), phase_std=first("phase_STPs"), dipole=first("dipoles"),
        r=first("UNIFAC_Rs"), q=first("UNIFAC_Qs"), unifac_groups=dict(groups),
        coolprop=coolprop, _constants=consts, _correlations=corrs,
    )


def require(sub, *attrs):
    missing = [a for a in attrs if getattr(sub, a) is None]
    if missing:
        raise ValueError(f"No data for {', '.join(missing)} of {sub.name}")
