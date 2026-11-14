"""Registry of calculations: one entry drives both the menu and the command line."""

from dataclasses import dataclass, field
from typing import Callable

from thermo_utils import (
    activity, chemical_equilibrium, cycles, devices, heat, mixtures, properties, substances, surface, vle,
)
from thermo_utils.pure import MODELS
from thermo_utils.units import (
    CONCENTRATION, DIMENSIONLESS, LENGTH, MASS_FLOW, MOLAR_ENERGY, MOLAR_ENTROPY, MOLAR_FLOW, PRESSURE,
    SPECIFIC_ENTROPY, TEMPERATURE, MOLAR_VOLUME,
    _NUMBER, display, parse_quantity, set_molar_mass,
)

SUBSTANCE = "substance"
SUBSTANCES = "substances"
QUANTITY = "quantity"
NUMBER = "number"
INTEGER = "integer"
CHOICE = "choice"
TEXT = "text"
LIST = "list"
FLOW = "flow"


@dataclass
class Field:
    name: str
    label: str
    type: str
    qkind: str = None
    optional: bool = False
    default: object = None
    choices: dict = None
    help: str = ""


@dataclass
class Calc:
    id: str
    group: str
    title: str
    fn: Callable
    fields: list = field(default_factory=list)
    help: str = ""


def parse_value(f, text):
    text = text.strip()
    if text == "":
        if f.optional or f.default is not None:
            return f.default
        raise ValueError(f"{f.label} is required")
    if f.type == SUBSTANCE:
        sub = substances.get(text)
        set_molar_mass(sub.MW)
        return sub
    if f.type == SUBSTANCES:
        subs = [substances.get(t) for t in _split(text)]
        set_molar_mass(subs[0].MW if len(subs) == 1 else None)
        return subs
    if f.type == QUANTITY:
        value = parse_quantity(text, f.qkind)
        unit = _NUMBER.match(text.replace(",", ".")).group(2)
        if unit:
            display.adopt(f.qkind, unit, name=f.name)
        return value
    if f.type == NUMBER:
        return float(text.replace(",", ".")) if "/" not in text else _ratio(text)
    if f.type == INTEGER:
        return int(text)
    if f.type == LIST:
        return [float(t.replace(",", ".")) for t in _split(text)]
    if f.type == CHOICE:
        key = text.lower()
        if key not in f.choices:
            raise ValueError(f"Options for {f.label}: {', '.join(f.choices)}")
        return key
    if f.type == FLOW:
        return _parse_flow(text)
    return text


def _split(text):
    sep = ";" if ";" in text else ","
    return [t.strip() for t in text.split(sep) if t.strip()]


def _ratio(text):
    a, b = text.split("/")
    return float(a) / float(b)


def _parse_flow(text):
    from thermo_utils.units import _NUMBER, _norm, _FACTORS, to_si
    m = _NUMBER.match(text.replace(",", "."))
    if not m:
        raise ValueError("Flow must look like '10 kg/s' or '5 kmol/h'")
    value, unit = float(m.group(1)), _norm(m.group(2) or "kg/s")
    if unit in _FACTORS[MASS_FLOW]:
        return ("mass", to_si(value, unit, MASS_FLOW))
    if unit in _FACTORS[MOLAR_FLOW]:
        return ("molar", to_si(value, unit, MOLAR_FLOW))
    raise ValueError(f"Unknown flow unit '{unit}'")


MODEL_FIELD = Field("model", "Model", CHOICE, default="auto", choices=MODELS)
FLOW_FIELD = Field("flow", "Flow rate (e.g. 2 kg/s, 10 kmol/h)", FLOW, optional=True,
                   help="Optional. Gives the power in kW.")


def _q(name, label, qkind, optional=False, default=None, help=""):
    return Field(name, label, QUANTITY, qkind, optional, default, help=help)


def _sub(name="sub", label="Substance"):
    return Field(name, label, SUBSTANCE, help="Name, formula or CAS")


PURE = "Pure fluids"
HEAT = "Heat effects"
DEVICES = "Flow devices"
MIX = "Mixtures"
PHASE = "Phase equilibrium"
REACT = "Chemical equilibrium"
SURF = "Surface phenomena"
CYCLES = "Power cycles"

CALCS = [
    Calc("info", PURE, "Substance data (Tc, Pc, ω, Cp, ΔHf...)", properties.substance_info, [_sub()]),
    Calc("pvt", PURE, "PVT: solve P, V or T (any EOS) + model comparison", properties.pvt, [
        _sub(), _q("T_", "T", TEMPERATURE, True), _q("P_", "P", PRESSURE, True),
        _q("V_", "V (molar)", MOLAR_VOLUME, True), MODEL_FIELD,
    ], "Leave empty the variable you want to calculate."),
    Calc("state", PURE, "State properties (V, Z, H, S, U, G, A, Cp, φ...)", properties.state_properties, [
        _sub(), _q("T_", "T", TEMPERATURE, True), _q("P_", "P", PRESSURE, True),
        Field("x", "Vapor fraction (two-phase only)", NUMBER, optional=True), MODEL_FIELD,
    ], "Give T and P, or a vapor fraction with T or P."),
    Calc("delta", PURE, "Property change ΔH, ΔS, ΔU, ΔG, ΔA between two states", properties.property_change, [
        _sub(),
        _q("T1", "T1", TEMPERATURE, True), _q("P1", "P1", PRESSURE, True), _q("V1", "V1", MOLAR_VOLUME, True),
        Field("x1", "Vapor fraction 1", NUMBER, optional=True),
        _q("T2", "T2", TEMPERATURE, True), _q("P2", "P2", PRESSURE, True), _q("V2", "V2", MOLAR_VOLUME, True),
        Field("x2", "Vapor fraction 2", NUMBER, optional=True),
    ] + [_q(k, f"Δ{p}", MOLAR_ENTROPY if k == "dS" else MOLAR_ENERGY, True) for k, p in properties.DELTAS.items()]
      + [MODEL_FIELD],
         "Give two of T, P, V for each state, or three in total plus one of ΔU, ΔH, ΔS, ΔG, ΔA."),
    Calc("saturation", PURE, "Saturation: Psat/Tsat, ΔHvap, saturated volumes", properties.saturation, [
        _sub(), _q("T_", "T", TEMPERATURE, True), _q("P_", "P", PRESSURE, True), MODEL_FIELD,
    ], "Give T or P."),
    Calc("sensible", HEAT, "Sensible heat ∫Cp dT and mean heat capacities", heat.sensible_heat, [
        _sub(), _q("T1", "T1", TEMPERATURE), _q("T2", "T2", TEMPERATURE),
        Field("phase", "Phase", CHOICE, default="gas", choices={"gas": "ideal gas", "liquid": "liquid"}),
    ]),
    Calc("latent", HEAT, "Latent heats (databank, Riedel, Watson, fusion)", heat.latent_heat, [
        _sub(), _q("T_", "T (default: normal boiling point)", TEMPERATURE, True),
    ]),
    Calc("flame", HEAT, "Adiabatic flame temperature", heat.adiabatic_flame, [
        _sub("fuel", "Fuel"), Field("excess_air", "Excess air fraction (0.2 = 20 %)", NUMBER, default=0.0),
        _q("T_in", "Inlet temperature", TEMPERATURE, default=298.15),
    ]),
    Calc("device", DEVICES, "Compressor, pump, turbine or valve (one step)", devices.flow_device, [
        _sub(), Field("kind", "Device", CHOICE, choices={
            "compressor": "compressor", "pump": "pump", "turbine": "turbine/expander", "valve": "throttle valve"}),
        _q("T1", "Inlet T", TEMPERATURE), _q("P1", "Inlet P", PRESSURE), _q("P2", "Outlet P", PRESSURE),
        Field("efficiency", "Isentropic efficiency (0-1)", NUMBER, default=1.0), MODEL_FIELD, FLOW_FIELD,
    ]),
]


GE_FIELD = Field("model", "Activity model", CHOICE, default="unifac", choices=activity.MODELS)
PARAMS_FIELD = Field("params", "Model parameters (Margules A12,A21 / van Laar / Wilson Λ12,Λ21 / NRTL b12,b21,α)",
                     LIST, optional=True, help="Only for models that need them.")
SUBS_FIELD = Field("subs", "Substances (comma separated)", SUBSTANCES)
COMP_HELP = "Mole fractions in the same order; the last one may be omitted."


def _comp(name="zs", label="Mole fractions"):
    return Field(name, f"{label} (e.g. 0.3,0.7)", LIST, help=COMP_HELP)


CALCS += [
    Calc("fugacity-mix", MIX, "Fugacity and φ̂ of species in a mixture (PR, SRK or virial)", mixtures.mixture_fugacity, [
        SUBS_FIELD, _q("T_", "T", TEMPERATURE), _q("P_", "P", PRESSURE), _comp("ys"),
        Field("model", "Model", CHOICE, default="pr", choices={"pr": "Peng-Robinson", "srk": "SRK", "virial": "virial"}),
    ]),
    Calc("activity", MIX, "Activity coefficients, G^E, H^E, mixing properties", mixtures.activity_coefficients, [
        SUBS_FIELD, _q("T_", "T", TEMPERATURE), _comp("xs"), GE_FIELD, PARAMS_FIELD,
    ]),
    Calc("vle", PHASE, "VLE: bubble/dew P or T, flash (γ-φ, modified Raoult, φ-φ)", mixtures.vle_calc, [
        SUBS_FIELD, Field("kind", "Calculation", CHOICE, choices={
            "bubble-p": "bubble P (give T, x)", "dew-p": "dew P (give T, y)", "bubble-t": "bubble T (give P, x)",
            "dew-t": "dew T (give P, y)", "flash": "flash (give T, P, z)"}),
        _q("T_", "T", TEMPERATURE, True), _q("P_", "P", PRESSURE, True), _comp("zs", "Composition x, y or z"),
        Field("method", "Method", CHOICE, default="gamma-phi", choices=vle.METHODS), GE_FIELD, PARAMS_FIELD,
    ]),
    Calc("diagram", PHASE, "Binary Pxy / Txy / xy diagram and azeotrope", mixtures.binary_diagram, [
        SUBS_FIELD, _q("T_", "T (for Pxy)", TEMPERATURE, True), _q("P_", "P (for Txy)", PRESSURE, True),
        Field("method", "Method", CHOICE, default="gamma-phi", choices=vle.METHODS), GE_FIELD, PARAMS_FIELD,
    ], "Give T for a Pxy diagram or P for a Txy diagram."),
    Calc("fit", MIX, "Fit Margules / van Laar to P-x-y data (CSV)", mixtures.fit_parameters, [
        SUBS_FIELD, _q("T_", "T of the data", TEMPERATURE),
        Field("path", "CSV file with columns P, x1, y1", TEXT),
        Field("model", "Model", CHOICE, default="margules", choices={"margules": "Margules", "vanlaar": "van Laar"}),
    ], "Header like 'P (kPa),x1,y1' sets the pressure unit."),
    Calc("lle", PHASE, "Liquid-liquid equilibrium (binary) + binodal curve", mixtures.lle, [
        SUBS_FIELD, _q("T_", "T", TEMPERATURE), GE_FIELD, PARAMS_FIELD,
    ]),
    Calc("sle", PHASE, "Solid-liquid equilibrium: eutectic diagram (binary)", mixtures.sle, [
        SUBS_FIELD, Field("model", "Activity model", CHOICE, default="ideal", choices=activity.MODELS), PARAMS_FIELD,
    ]),
    Calc("solubility", PHASE, "Solubility of a solid in a liquid", mixtures.solubility, [
        _sub("solute", "Solid solute"), _q("T_", "T", TEMPERATURE),
        Field("solvent", "Solvent (optional)", SUBSTANCE, optional=True),
        Field("model", "Activity model", CHOICE, default="ideal", choices=activity.MODELS),
    ]),
    Calc("osmotic", PHASE, "Osmotic pressure", mixtures.osmotic, [
        _sub("solvent", "Solvent"), _sub("solute", "Solute"), _q("T_", "T", TEMPERATURE),
        Field("x_solute", "Solute mole fraction", NUMBER),
        Field("model", "Activity model", CHOICE, default="ideal", choices=activity.MODELS),
    ]),
    Calc("k", REACT, "Equilibrium constant K(T), ΔG°, ΔH° of a reaction", chemical_equilibrium.equilibrium_constant, [
        Field("reaction", "Reaction (e.g. N2 + 3 H2 -> 2 NH3)", TEXT), _q("T_", "T", TEMPERATURE),
    ]),
    Calc("equilibrium", REACT, "Equilibrium conversion of one reaction", chemical_equilibrium.single_reaction, [
        Field("reaction", "Reaction (e.g. CO + H2O -> CO2 + H2)", TEXT), _q("T_", "T", TEMPERATURE),
        _q("P_", "P", PRESSURE, default=1e5), Field("feed", "Feed in mol (e.g. CO=1, H2O=2, N2=0.5)", TEXT),
        Field("phase_model", "Phase model", CHOICE, default="ideal", choices={
            "ideal": "ideal gas", "virial": "gas with virial φ̂", "ideal-liquid": "ideal liquid",
            "unifac": "liquid with UNIFAC γ"}),
    ], "Species in the feed that are not in the reaction are inerts."),
    Calc("gibbs", REACT, "Multi-reaction equilibrium (Gibbs energy minimization)", chemical_equilibrium.gibbs_minimization, [
        Field("species", "All species that may be present (comma separated)", SUBSTANCES),
        _q("T_", "T", TEMPERATURE), _q("P_", "P", PRESSURE, default=1e5),
        Field("feed", "Feed in mol (e.g. CH4=2, H2O=3)", TEXT),
    ], "Ideal-gas mixture. No need to write the reactions."),
]

CALCS += [
    Calc("surface-tension", SURF, "Surface tension σ(T), surface entropy and enthalpy", surface.surface_tension, [
        _sub(), _q("T_", "T", TEMPERATURE),
    ]),
    Calc("kelvin", SURF, "Kelvin equation: vapor pressure over a curved surface", surface.kelvin, [
        _sub(), _q("T_", "T", TEMPERATURE), _q("r", "Radius of curvature", LENGTH),
        Field("kind", "Surface", CHOICE, default="drop", choices={
            "drop": "drop (convex)", "bubble": "bubble or capillary (concave)"}),
    ]),
    Calc("adsorption", SURF, "Adsorption isotherms: Langmuir, Freundlich, BET (CSV)", surface.adsorption, [
        Field("path", "CSV file with columns P, q", TEXT),
        Field("model", "Model", CHOICE, default="all", choices={
            "all": "all", "langmuir": "Langmuir", "freundlich": "Freundlich", "bet": "BET"}),
        Field("adsorbate", "Adsorbate (for BET P0 and area)", SUBSTANCE, optional=True),
        _q("T_", "T (for BET)", TEMPERATURE, True), _q("P0", "P0 (optional, BET)", PRESSURE, True),
        Field("cross_section", "Area per molecule in nm² (optional, BET)", NUMBER, optional=True),
    ], "Header like 'P (kPa),q (mmol/g)' sets the units."),
    Calc("gibbs-adsorption", SURF, "Gibbs adsorption isotherm (surface excess from σ vs c)", surface.gibbs_adsorption, [
        Field("path", "CSV file with columns c, σ", TEXT), _q("T_", "T", TEMPERATURE),
        Field("n", "n (1 nonionic, 2 ionic 1:1 without salt)", INTEGER, default=1),
    ], "Header like 'c (mmol/L),sigma (mN/m)' sets the units."),
    Calc("micelles", SURF, "Micellization: CMC and ΔG°mic", surface.micelles, [
        _q("T_", "T", TEMPERATURE), _q("cmc", "CMC (or leave empty and give a CSV)", CONCENTRATION, True),
        Field("beta", "Counterion binding β (0 for nonionic)", NUMBER, default=0.0),
        _q("T2", "Second T (optional, for ΔH)", TEMPERATURE, True),
        _q("cmc2", "CMC at the second T (optional)", CONCENTRATION, True),
        Field("path", "CSV file with columns c, σ (optional)", TEXT, optional=True),
    ]),
]


# Power cycles

GAS_FIELDS = [
    Field("k", "k = cp/cv (constant Cp)", NUMBER, optional=True, help="Gives the cold air-standard."),
    _q("cp", "cp (constant Cp)", SPECIFIC_ENTROPY, True),
]


def _var_field(v):
    if v.kind == DIMENSIONLESS:
        return Field(v.key, v.label, NUMBER, optional=True)
    return _q(v.key, v.label, v.kind, True)


def _cycle_fields(models):
    fields, seen = [Field("sub", "Working fluid", SUBSTANCE, optional=True, help="Name, formula or CAS")], set()
    if any(m.fluid == "gas" for m in models):
        fields += GAS_FIELDS
    for m in models:
        for v in cycles.all_vars(m):
            if v.key not in seen:
                seen.add(v.key)
                fields.append(_var_field(v))
    return fields


def _cycle_fn(model_id):
    def fn(sub=None, k=None, cp=None, **given):
        model = cycles.MODELS[model_id]
        return cycles.run(model_id, sub or substances.get("air" if model.fluid == "gas" else "water"),
                          k, cp, **given)
    return fn


def _step_fn(kind, sub=None, k=None, cp=None, **given):
    model = cycles.STEPS[kind]
    return cycles.run(kind, sub or substances.get("air" if model.fluid == "gas" else "water"), k, cp,
                      steps=True, **given)


CYCLE_HELP = "Give any values that fix the cycle; the rest is solved."
CALCS += [
    Calc(m.id, CYCLES, f"{m.title} (any data in, the rest out)", _cycle_fn(m.id), _cycle_fields([m]), CYCLE_HELP)
    for m in cycles.MODELS.values()
] + [
    Calc("step", CYCLES, "One cycle step: compression, heating, turbine, pump...", _step_fn,
         [Field("kind", "Step", CHOICE, choices=cycles.STEP_DESCRIPTIONS)]
         + _cycle_fields(list(cycles.STEPS.values())), "Give any values that fix the step; the rest is solved."),
]


def by_id(calc_id):
    for c in CALCS:
        if c.id == calc_id:
            return c
    raise KeyError(calc_id)


def groups():
    out = {g: [] for g in (PURE, HEAT, DEVICES, CYCLES, MIX, PHASE, REACT, SURF)}
    for c in CALCS:
        out.setdefault(c.group, []).append(c)
    return out
