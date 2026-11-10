"""Mixtures: fugacity, activity, excess properties and phase equilibria."""

import csv
import math

import numpy as np
from scipy.optimize import brentq, fsolve

from thermo_utils import activity, vle
from thermo_utils.result import Result, row
from thermo_utils.units import (
    MOLAR_ENERGY as E, MOLAR_ENTROPY as S, MOLAR_VOLUME as V, PRESSURE as P, TEMPERATURE as T,
    R, display, fmt, from_si,
)


def _norm_comp(subs, xs, label="composition"):
    if xs is None:
        raise ValueError(f"Give the {label}.")
    if len(xs) == len(subs) - 1:
        xs = list(xs) + [1 - sum(xs)]
    if len(xs) != len(subs):
        raise ValueError(f"The {label} needs {len(subs)} mole fractions.")
    if any(x < 0 for x in xs):
        raise ValueError("Mole fractions must be positive.")
    s = sum(xs)
    return [x / s for x in xs]


def _names(subs):
    return [s.name for s in subs]


def _comp_rows(subs, xs, label):
    return [row(f"{label} {s.name}", f"{x:.6f}") for s, x in zip(subs, xs)]


def _tu():
    return display.temperature


def _pu():
    return display.pressure


# Mixture fugacity

def mixture_fugacity(subs, T_, P_, ys, model="pr"):
    ys = _norm_comp(subs, ys)
    res = Result("Fugacity in a mixture")
    if model == "virial":
        phis, Bmix = vle.virial_phis(subs, T_, P_, ys)
        Z = 1 + Bmix * P_ / (R * T_)
        rows = res.section("Virial mixture (Pitzer)", [
            row("B mixture", Bmix, V), row("Z", Z), row("V", Z * R * T_ / P_, V),
        ])
    else:
        from thermo import CEOSGas, CEOSLiquid, ChemicalConstantsPackage, PRMIX, SRKMIX
        from thermo.interaction_parameters import IPDB
        cas = [s.cas for s in subs]
        c, p = ChemicalConstantsPackage.from_IDs(cas)
        kij = IPDB.get_ip_asymmetric_matrix("ChemSep PR", cas, "kij")
        kw = dict(Tcs=c.Tcs, Pcs=c.Pcs, omegas=c.omegas, kijs=kij)
        eos = PRMIX if model == "pr" else SRKMIX
        g = CEOSGas(eos, kw, HeatCapacityGases=p.HeatCapacityGases).to(T=T_, P=P_, zs=ys)
        l = CEOSLiquid(eos, kw, HeatCapacityGases=p.HeatCapacityGases).to(T=T_, P=P_, zs=ys)
        st = min((g, l), key=lambda ph: ph.G_dep())
        phis = st.phis()
        rows = res.section(f"{model.upper()} with van der Waals mixing rules (ChemSep kij)", [
            row("Root", "vapor-like" if st is g else "liquid-like"), row("Z", st.Z()), row("V", st.V(), V),
            row("H residual", st.H_dep(), E), row("S residual", st.S_dep(), S),
        ])
    for s, y, ph in zip(subs, ys, phis):
        rows.append(row(f"φ̂ {s.name}", f"{ph:.6f}   f̂ = {fmt(ph * y * P_, P)}"))
    res.section("Composition", _comp_rows(subs, ys, "y"))
    lewis = []
    for s, y, ph in zip(subs, ys, phis):
        try:
            from thermo_utils import pure
            phi_pure = pure.state(s, T_, P_, "pr" if model != "virial" else "virial").phi
            lewis.append(row(f"Lewis-Randall f̂ = y·φ_pure·P ({s.name})", fmt(y * phi_pure * P_, P)))
        except Exception:
            pass
    if lewis:
        res.section("Lewis-Randall rule (ideal solution) for comparison", lewis)
    return res


# Activity coefficients and excess properties

def activity_coefficients(subs, T_, xs, model="unifac", params=None):
    xs = _norm_comp(subs, xs)
    ge = activity.GEModel(subs, model, params)
    lg = ge.ln_gammas(T_, xs)
    GE, HE = ge.GE(T_, xs), ge.HE(T_, xs)
    res = Result(f"Activity coefficients ({activity.MODELS[model]})")
    res.section("Composition", _comp_rows(subs, xs, "x"))
    res.section("Activity coefficients", [
        row(f"γ {s.name}", f"{math.exp(l):.6f}   ln γ = {l:.6f}   a = {x * math.exp(l):.6f}")
        for s, x, l in zip(subs, xs, lg)
    ])
    dG_id = R * T_ * sum(x * math.log(x) for x in xs if x > 0)
    res.section("Excess and mixing properties", [
        row("G^E", GE, E), row("G^E / RT", GE / (R * T_)), row("H^E = ΔH mixing", HE, E),
        row("S^E", (HE - GE) / T_, S), row("ΔG mixing", dG_id + GE, E),
        row("ΔS mixing", -R * sum(x * math.log(x) for x in xs if x > 0) + (HE - GE) / T_, S),
    ] + [row(f"Partial excess Ḡ^E {s.name} = RT ln γ", R * T_ * l, E) for s, l in zip(subs, lg)])
    res.notes += ge.notes
    if len(subs) == 2:
        res.plot = lambda: _plot_activity(ge, T_)
        res.plot_name = f"gamma_{'_'.join(_names(subs))}"
    return res


def _plot_activity(ge, T_):
    import matplotlib.pyplot as plt
    x1, lg = activity.binary_curve(ge, T_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x1, lg[:, 0], label=f"ln γ {ge.subs[0].name}")
    ax.plot(x1, lg[:, 1], label=f"ln γ {ge.subs[1].name}")
    ge_rt = x1 * lg[:, 0] + (1 - x1) * lg[:, 1]
    ax.plot(x1, ge_rt, "k--", label="G^E/RT")
    ax.set_xlabel(f"x {ge.subs[0].name}"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"{ge.name} at {T_:.2f} K")
    fig.tight_layout()
    return fig


# VLE

def vle_calc(subs, kind, T_=None, P_=None, zs=None, method="gamma-phi", model="unifac", params=None):
    zs = _norm_comp(subs, zs)
    solver = vle.solver(subs, method, model, params)
    if kind in ("bubble-p", "dew-p", "flash") and T_ is None:
        raise ValueError("Give T.")
    if kind in ("bubble-t", "dew-t", "flash") and P_ is None:
        raise ValueError("Give P.")
    r = {"bubble-p": lambda: solver.bubble_P(T_, zs), "dew-p": lambda: solver.dew_P(T_, zs),
         "bubble-t": lambda: solver.bubble_T(P_, zs), "dew-t": lambda: solver.dew_T(P_, zs),
         "flash": lambda: solver.flash(T_, P_, zs)}[kind]()
    titles = {"bubble-p": "Bubble pressure", "dew-p": "Dew pressure", "bubble-t": "Bubble temperature",
              "dew-t": "Dew temperature", "flash": "Isothermal flash (T, P)"}
    res = Result(f"{titles[kind]} - {vle.METHODS[method].split(':')[0]}")
    rows = res.section("Result", [row("T", r.T, T), row("P", r.P, P)])
    if r.VF is not None:
        rows.append(row("Vapor fraction V/F", f"{r.VF:.6f}"))
    comp = res.section("Phase compositions")
    for i, s in enumerate(subs):
        x = f"{r.xs[i]:.6f}" if r.xs and r.xs[i] is not None else "-"
        y = f"{r.ys[i]:.6f}" if r.ys and r.ys[i] is not None else "-"
        k = f"   K = {r.ys[i] / r.xs[i]:.5g}" if x != "-" and y != "-" and r.xs[i] > 0 else ""
        comp.append(row(s.name, f"x = {x}   y = {y}{k}"))
    if r.gammas:
        res.section("Model values", [
            row(s.name, f"γ = {g:.5f}   Φ = {f:.5f}   Psat = {fmt(p, P)}")
            for s, g, f, p in zip(subs, r.gammas, r.phis, r.Psats)
        ])
    if method != "phi-phi" and model and method != "raoult":
        res.notes += solver.ge.notes
    if method == "phi-phi":
        res.notes.append("φ-φ with PR is best for nonpolar mixtures; use γ-φ for polar liquids.")
    if r.VF in (0.0, 1.0) and kind == "flash":
        res.notes.append("Single phase: P is outside the bubble-dew range at this T.")
    if len(subs) == 2:
        fixed_T = kind in ("bubble-p", "dew-p", "flash")
        res.plot = lambda: _plot_binary(solver, subs, r.T if fixed_T else None, None if fixed_T else r.P)
        res.plot_name = f"vle_{'_'.join(_names(subs))}"
    return res


def binary_diagram(subs, T_=None, P_=None, method="gamma-phi", model="unifac", params=None):
    if len(subs) != 2:
        raise ValueError("Phase diagrams need exactly two substances.")
    if (T_ is None) == (P_ is None):
        raise ValueError("Give T (Pxy diagram) or P (Txy diagram).")
    solver = vle.solver(subs, method, model, params)
    xs, ys, vals = _binary_curve(solver, T_, P_)
    res = Result(f"{'Pxy' if T_ else 'Txy'} diagram {subs[0].name} / {subs[1].name}")
    rows = res.section(f"Points (x, y of {subs[0].name})")
    kind = P if T_ else T
    for x, y, v in list(zip(xs, ys, vals))[::5]:
        rows.append(row(f"x = {x:.3f}", f"y = {y:.4f}   {'P' if T_ else 'T'} = {fmt(v, kind)}"))
    az = _azeotrope(xs, ys, vals)
    if az:
        res.section("Azeotrope", [row("x = y", f"{az[0]:.4f}"), row("P" if T_ else "T", fmt(az[1], kind))])
    else:
        res.notes.append("No azeotrope found.")
    res.plot = lambda: _plot_binary(solver, subs, T_, P_)
    res.plot_name = f"{'pxy' if T_ else 'txy'}_{'_'.join(_names(subs))}"
    return res


def _binary_curve(solver, T_, P_, n=51):
    xs, ys, vals = [], [], []
    for x in np.linspace(0, 1, n):
        try:
            r = solver.bubble_P(T_, [x, 1 - x]) if T_ else solver.bubble_T(P_, [x, 1 - x])
            xs.append(x); ys.append(r.ys[0]); vals.append(r.P if T_ else r.T)
        except Exception:
            pass
    return xs, ys, vals


def _azeotrope(xs, ys, vals):
    d = [y - x for x, y in zip(xs, ys)]
    for i in range(1, len(d) - 2):
        if d[i] * d[i + 1] < 0:
            f = d[i] / (d[i] - d[i + 1])
            return xs[i] + f * (xs[i + 1] - xs[i]), vals[i] + f * (vals[i + 1] - vals[i])
    return None


def _plot_binary(solver, subs, T_, P_):
    import matplotlib.pyplot as plt
    xs, ys, vals = _binary_curve(solver, T_, P_)
    kind, unit = (P, _pu()) if T_ else (T, _tu())
    v = [from_si(a, unit, kind) for a in vals]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(xs, v, label="bubble (x)")
    ax1.plot(ys, v, label="dew (y)")
    ax1.set_xlabel(f"x, y {subs[0].name}")
    ax1.set_ylabel(f"{'P' if T_ else 'T'} ({unit})")
    ax1.set_title(f"{'Pxy at ' + fmt(T_, T) if T_ else 'Txy at ' + fmt(P_, P)}")
    ax2.plot(xs, ys, label="equilibrium")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.set_xlabel(f"x {subs[0].name}"); ax2.set_ylabel(f"y {subs[0].name}"); ax2.set_title("x-y diagram")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(); ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


# Liquid-liquid equilibrium

def lle(subs, T_, model="unifac", params=None):
    if len(subs) != 2:
        raise ValueError("LLE is implemented for binary mixtures.")
    ge = activity.GEModel(subs, model, params)
    sol = _lle_solve(ge, T_)
    res = Result(f"Liquid-liquid equilibrium ({activity.MODELS[model]})")
    if sol is None:
        res.section("Result", [row("Phases", "single liquid phase (miscible) at this T")])
    else:
        a, b = sol
        g = ge.gammas(T_, [a, 1 - a])
        res.section("Coexisting liquid phases", [
            row(f"Phase α: x {subs[0].name}", f"{a:.6f}"), row(f"Phase β: x {subs[0].name}", f"{b:.6f}"),
            row(f"γ {subs[0].name} (α)", f"{g[0]:.5f}"), row(f"γ {subs[1].name} (α)", f"{g[1]:.5f}"),
        ])
    res.notes += ge.notes
    res.notes.append("Solved from xα·γα = xβ·γβ for both species.")
    res.plot = lambda: _plot_lle(ge, subs, T_)
    res.plot_name = f"lle_{'_'.join(_names(subs))}"
    return res


def _lle_solve(ge, T_):
    def eqs(v):
        a, b = v
        ga, gb = ge.gammas(T_, [a, 1 - a]), ge.gammas(T_, [b, 1 - b])
        return [a * ga[0] - b * gb[0], (1 - a) * ga[1] - (1 - b) * gb[1]]
    for guess in ([0.01, 0.99], [0.05, 0.9], [0.1, 0.8], [0.001, 0.999]):
        sol, info, ok, _ = fsolve(eqs, guess, full_output=True, xtol=1e-12)
        a, b = sol
        if ok == 1 and 0 < a < 1 and 0 < b < 1 and abs(a - b) > 1e-4 and max(map(abs, eqs(sol))) < 1e-9:
            return tuple(sorted((a, b)))
    return None


def _plot_lle(ge, subs, T_):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.linspace(1e-4, 1 - 1e-4, 300)
    dg = [ge.GE(T_, [a, 1 - a]) / (R * T_) + a * math.log(a) + (1 - a) * math.log(1 - a) for a in x]
    ax1.plot(x, dg)
    ax1.set_xlabel(f"x {subs[0].name}"); ax1.set_ylabel("ΔG mixing / RT"); ax1.set_title(f"Stability at {T_:.2f} K")
    pts = []
    for t in np.linspace(T_ - 60, T_ + 60, 41):
        s = _lle_solve(ge, t)
        if s:
            pts.append((s[0], s[1], t))
    if pts:
        a, b, t = zip(*pts)
        tu = [from_si(v, _tu(), T) for v in t]
        ax2.plot(a, tu, "o-", ms=3, label="phase α")
        ax2.plot(b, tu, "o-", ms=3, label="phase β")
        ax2.legend()
    ax2.set_xlabel(f"x {subs[0].name}"); ax2.set_ylabel(f"T ({_tu()})"); ax2.set_title("Binodal curve")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


# Solid-liquid equilibrium

def sle(subs, P_=101325.0, model="ideal", params=None):
    """Solubility / eutectic of a binary with immiscible solids."""
    if len(subs) != 2:
        raise ValueError("SLE is implemented for binary mixtures.")
    for s in subs:
        if not (s.Tm and s.Hfus):
            raise ValueError(f"No melting data for {s.name}")
    ge = activity.GEModel(subs, model, params)

    def liquidus(i, xi):
        xs = [xi, 1 - xi] if i == 0 else [1 - xi, xi]
        s = subs[i]
        f = lambda t: math.log(xi * ge.gammas(t, xs)[i]) + s.Hfus / R * (1 / t - 1 / s.Tm)
        return brentq(f, 0.2 * s.Tm, s.Tm * 1.0000001)

    def diff(x1):
        return liquidus(0, x1) - liquidus(1, 1 - x1)

    xe = brentq(diff, 1e-4, 1 - 1e-4)
    Te = liquidus(0, xe)
    res = Result(f"Solid-liquid equilibrium {subs[0].name} / {subs[1].name}")
    res.section("Eutectic point", [row(f"x {subs[0].name}", f"{xe:.5f}"), row("T eutectic", Te, T)])
    res.section("Pure-solid data", [
        row(f"{s.name}", f"Tm = {fmt(s.Tm, T)}   ΔHfus = {fmt(s.Hfus, E)}") for s in subs
    ])
    res.notes.append("ln(xi γi) = ΔHfus,i/R (1/Tm,i - 1/T), pure solids, ΔCp neglected.")
    res.notes += ge.notes

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        x1 = np.linspace(xe, 0.9999, 60)
        ax.plot(x1, [from_si(liquidus(0, a), _tu(), T) for a in x1], label=f"solid {subs[0].name}")
        x1 = np.linspace(1e-4, xe, 60)
        ax.plot(x1, [from_si(liquidus(1, 1 - a), _tu(), T) for a in x1], label=f"solid {subs[1].name}")
        ax.axhline(from_si(Te, _tu(), T), color="k", ls="--", lw=0.8)
        ax.set_xlabel(f"x {subs[0].name}"); ax.set_ylabel(f"T ({_tu()})"); ax.set_xlim(0, 1)
        ax.set_title("Solid-liquid diagram"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        return fig

    res.plot = plot
    res.plot_name = f"sle_{'_'.join(_names(subs))}"
    return res


def solubility(solute, T_, solvent=None, model="ideal"):
    subs = [solute] + ([solvent] if solvent else [])
    if not (solute.Tm and solute.Hfus):
        raise ValueError(f"No melting data for {solute.name}")
    ideal = math.exp(-solute.Hfus / R * (1 / T_ - 1 / solute.Tm))
    res = Result(f"Solubility of solid {solute.name}")
    rows = res.section("Result", [row("Ideal solubility x", f"{min(ideal, 1):.6g}")])
    if solvent and model != "ideal":
        ge = activity.GEModel(subs, model)
        f = lambda x: math.log(x * ge.gammas(T_, [x, 1 - x])[0]) - math.log(ideal)
        try:
            x = brentq(f, 1e-10, 1 - 1e-10)
            rows.append(row(f"Solubility with {model}", f"{x:.6g}"))
        except ValueError:
            rows.append(row(f"Solubility with {model}", "no solution"))
    if T_ >= solute.Tm:
        res.notes.append("T is above the melting point: the solute is liquid.")
    return res


# Osmotic pressure

def osmotic(solvent, solute, T_, x_solute, model="ideal"):
    subs = [solvent, solute]
    xs = [1 - x_solute, x_solute]
    ge = activity.GEModel(subs, model)
    g = ge.gammas(T_, xs)[0]
    Vl = solvent.Vliq(T_)
    Pi = -R * T_ / Vl * math.log(xs[0] * g)
    res = Result("Osmotic pressure")
    res.section("Π = -(RT/V_solvent) ln(x_solvent γ_solvent)", [
        row("Solvent molar volume", Vl, V), row("γ solvent", f"{g:.6f}"),
        row("Osmotic pressure Π", Pi, P),
        row("Dilute approximation Π ≈ x RT/V", x_solute * R * T_ / Vl, P),
    ])
    return res


# Margules / van Laar fit from P-x-y data

def fit_parameters(subs, T_, path, model="margules"):
    """Fit Margules or van Laar to isothermal P-x1-y1 data."""
    if len(subs) != 2:
        raise ValueError("The fit needs two substances.")
    data = _read_pxy(path)
    ps = [s.Psat(T_) for s in subs]
    ends = {x1: Pexp for Pexp, x1, _ in data if x1 in (0, 1)}
    ps = [ends.get(1, ps[0]), ends.get(0, ps[1])]
    pts = []
    for Pexp, x1, y1 in data:
        if 0 < x1 < 1:
            g1 = y1 * Pexp / (x1 * ps[0])
            g2 = (1 - y1) * Pexp / ((1 - x1) * ps[1])
            pts.append((x1, math.log(g1), math.log(g2), Pexp, y1))
    if len(pts) < 2:
        raise ValueError("Need at least two points with 0 < x1 < 1.")
    x = np.array([p[0] for p in pts])
    gert = np.array([a * l1 + (1 - a) * l2 for a, l1, l2, *_ in pts])
    y = gert / (x * (1 - x))
    if model == "margules":
        A = np.vstack([x, 1 - x]).T
        A21, A12 = np.linalg.lstsq(A, y, rcond=None)[0]
        params = (A12, A21)
    else:
        A = np.vstack([x, 1 - x]).T
        inv = 1 / y
        c1, c2 = np.linalg.lstsq(A, inv, rcond=None)[0]
        params = (1 / c2, 1 / c1)
    solver = vle.solver(subs, "modified-raoult", model, list(params))
    solver.psats = lambda T, ps=ps: ps
    dev = [(solver.bubble_P(T_, [p[0], 1 - p[0]]).P - p[3]) for p in pts]
    rms = math.sqrt(sum(d * d for d in dev) / len(dev))
    res = Result(f"{model.capitalize()} parameters from P-x-y data")
    res.section("Fitted parameters", [
        row("A12" if model == "margules" else "A'12", f"{params[0]:.6f}"),
        row("A21" if model == "margules" else "A'21", f"{params[1]:.6f}"),
        row("RMS deviation in P", rms, P), row("Points used", str(len(pts))),
        row("Psat used", ", ".join(f"{s.name}: {fmt(p, P)}" for s, p in zip(subs, ps))),
    ])
    res.notes.append("γi = yi P / (xi Psat_i) (modified Raoult); linear fit of G^E/(x1 x2 RT).")
    res.notes.append("Psat taken from the x1 = 0 and x1 = 1 rows when present, else from the databank.")

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        xx, yy, vv = _binary_curve(solver, T_, None)
        ax.plot(xx, [from_si(v, _pu(), P) for v in vv], label="P-x fit")
        ax.plot(yy, [from_si(v, _pu(), P) for v in vv], label="P-y fit")
        ax.plot([p[0] for p in pts], [from_si(p[3], _pu(), P) for p in pts], "o", label="x data")
        ax.plot([p[4] for p in pts], [from_si(p[3], _pu(), P) for p in pts], "s", label="y data")
        ax.set_xlabel(f"x, y {subs[0].name}"); ax.set_ylabel(f"P ({_pu()})"); ax.set_xlim(0, 1)
        ax.grid(alpha=0.3); ax.legend(); ax.set_title(f"{model} fit at {T_:.2f} K")
        fig.tight_layout()
        return fig

    res.plot = plot
    res.plot_name = f"fit_{model}"
    return res


def _read_pxy(path):
    """CSV columns P, x1, y1. P in the display pressure unit (header may say e.g. 'P (kPa)')."""
    from thermo_utils.units import to_si
    import re
    rows = []
    unit = display.pressure
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        for line in csv.reader(fh, dialect):
            if not line:
                continue
            try:
                vals = [float(v.replace(",", ".")) if dialect.delimiter != "," else float(v) for v in line[:3]]
            except ValueError:
                m = re.search(r"\(([^)]+)\)", line[0])
                if m:
                    unit = m.group(1)
                continue
            rows.append((to_si(vals[0], unit, P), vals[1], vals[2]))
    return rows

