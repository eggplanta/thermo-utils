"""Mixtures: fugacity, activity, excess properties and phase equilibria."""

import csv
import math

import numpy as np
from scipy.optimize import brentq, fsolve

from thermo_utils import activity, vle
from thermo_utils import steplog as log
from thermo_utils.result import Result, row
from thermo_utils.steplog import K, q
from thermo_utils.units import mark_given, num
from thermo_utils.units import (
    DIMENSIONLESS as D, MOLAR_ENERGY as E, MOLAR_ENTROPY as S, MOLAR_VOLUME as V, PRESSURE as P, TEMPERATURE as T,
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
    return [row(f"{label} {s.name}", num(x)) for s, x in zip(subs, xs)]


def _tu():
    return display.temperature


def _pu():
    return display.pressure


# Mixture fugacity

def mixture_fugacity(subs, T_, P_, ys, model="pr"):
    ys = _norm_comp(subs, ys)
    res = Result("Fugacity in a mixture")
    log.step("Composition: y = {}", vle._c(ys))
    if model == "virial":
        phis, Bmix = vle.virial_phis(subs, T_, P_, ys)
        Z = 1 + Bmix * P_ / (R * T_)
        if log.on():
            log.step("Virial mixture at T = {}, P = {}", q(T_, T), q(P_, P))
            vle.log_virial_B(subs, T_)
            log.line("B = ΣiΣj yi yj Bij = {}", q(Bmix, V))
            log.line("Z = 1 + BP/RT = 1 + {} · {} / ({} · {}) = {}", q(Bmix, V), q(P_, P), R, K(T_), q(Z))
            log.line("ln φ̂k = (P/RT)[Bkk + ½ΣiΣj yi yj(2δik − δij)], δik = 2Bik − Bii − Bkk")
            for s, ph in zip(subs, phis):
                log.line("  {}: ln φ̂ = {}  ->  φ̂ = {}", s.name, num(math.log(ph)), num(ph))
        rows = res.section("Virial mixture (Pitzer)", [
            row("B mixture", Bmix, V), row("Z", Z), row("V", Z * R * T_ / P_, V),
        ])
    else:
        from thermo import CEOSGas, CEOSLiquid, PRMIX, SRKMIX
        from thermo.interaction_parameters import IPDB
        kij = IPDB.get_ip_asymmetric_matrix("ChemSep PR", [s.cas for s in subs], "kij")
        kw = dict(Tcs=[s.Tc for s in subs], Pcs=[s.Pc for s in subs], omegas=[s.omega for s in subs], kijs=kij)
        eos = PRMIX if model == "pr" else SRKMIX
        g = CEOSGas(eos, kw).to(T=T_, P=P_, zs=ys)
        l = CEOSLiquid(eos, kw).to(T=T_, P=P_, zs=ys)
        st = min((g, l), key=lambda ph: ph.G_dep())
        phis = st.phis()
        if log.on():
            log.step("{} mixture at T = {}, P = {}", model.upper(), q(T_, T), q(P_, P))
            log.line("Mixing rules: aα = ΣiΣj yi yj √(aαi aαj)(1 − kij), b = Σ yi bi")
            n = len(subs)
            log.table(["i", "j", "kij"], [[subs[i].name, subs[j].name, kij[i][j]] for i in range(n) for j in range(i + 1, n)])
            eos_ = st.eos_mix
            log.table(["species", "aα_i", "b_i"], [[s.name, a_, q(b_, V)] for s, a_, b_ in zip(subs, eos_.a_alphas, eos_.bs)])
            log.line("Mixture: aα = {}, b = {}", num(eos_.a_alpha), q(eos_.b, V))
            log.line("A = aαP/(RT)² = {}, B = bP/RT = {}", num(eos_.a_alpha * P_ / (R * T_) ** 2), num(eos_.b * P_ / (R * T_)))
            log.line("Roots: vapor-like Z = {}, liquid-like Z = {}; stable root (lower G_R): {}",
                     num(g.Z()), num(l.Z()), "vapor-like" if st is g else "liquid-like")
            log.line("V = ZRT/P = {}; H_R = {}, S_R = {}", q(st.V(), V), q(st.H_dep(), E), q(st.S_dep(), S))
            log.line("ln φ̂i = (bi/b)(Z − 1) − ln(Z − B) − (A/(2√2 B))[2Σj yj aαij/aα − bi/b] ln[(Z + (1+√2)B)/(Z + (1−√2)B)] (PR)")
            for s, ph in zip(subs, phis):
                log.line("  {}: φ̂ = {}", s.name, num(ph))
        rows = res.section(f"{model.upper()} with van der Waals mixing rules (ChemSep kij)", [
            row("Root", "vapor-like" if st is g else "liquid-like"), row("Z", st.Z()), row("V", st.V(), V),
            row("H residual", st.H_dep(), E), row("S residual", st.S_dep(), S),
        ])
    log.step("Fugacities f̂i = yi φ̂i P")
    for s, y, ph in zip(subs, ys, phis):
        rows.append(row(f"φ̂ {s.name}", f"{num(ph)}   f̂ = {fmt(ph * y * P_, P)}"))
        log.line("{}: f̂ = {} × {} × {} = {}", s.name, num(y), num(ph), q(P_, P), q(ph * y * P_, P))
    res.section("Composition", _comp_rows(subs, ys, "y"))
    lewis = []
    for s, y, ph in zip(subs, ys, phis):
        try:
            from thermo_utils import pure
            with log.muted():
                phi_pure = pure.state(s, T_, P_, "pr" if model != "virial" else "virial").phi
            lewis.append(row(f"Lewis-Randall f̂ = y·φ_pure·P ({s.name})", fmt(y * phi_pure * P_, P)))
            if not lewis[1:]:
                log.step("Lewis-Randall rule (ideal solution): f̂i = yi φi(pure) P")
            log.line("{}: φ pure = {}  ->  f̂ = {}", s.name, num(phi_pure), q(y * phi_pure * P_, P))
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
    if log.on():
        log.step("Activity coefficients at T = {}, x = {}", q(T_, T), vle._c(xs))
        ge.explain(T_, xs)
        log.line("Activities ai = xi γi: {}", vle._c([x * math.exp(l) for x, l in zip(xs, lg)]))
        log.step("Excess properties")
        log.line("G^E/RT = Σ xi ln γi = {} = {}", " + ".join(f"{num(x)}·{num(l)}" for x, l in zip(xs, lg)), num(GE / (R * T_)))
        log.line("G^E = RT·(G^E/RT) = {}", q(GE, E))
        dT = 0.01
        g1, g2 = ge.GE(T_ + dT, xs) / (T_ + dT), ge.GE(T_ - dT, xs) / (T_ - dT)
        log.line("H^E = −T² d(G^E/T)/dT (central difference, ±{} K): G^E/T = {} and {} J/(mol·K)", dT, f"{g1:.8g}", f"{g2:.8g}")
        log.line("H^E = −({})² × ({} − {})/{} = {}", num(T_), f"{g1:.8g}", f"{g2:.8g}", 2 * dT, q(HE, E))
        log.line("S^E = (H^E − G^E)/T = {}", q((HE - GE) / T_, S))
        log.step("Mixing properties")
        sx = sum(x * math.log(x) for x in xs if x > 0)
        log.line("ΔG_mix = RT Σ xi ln xi + G^E = {} + {} = {}", q(R * T_ * sx, E), q(GE, E), q(R * T_ * sx + GE, E))
        log.line("ΔS_mix = −R Σ xi ln xi + S^E = {}", q(-R * sx + (HE - GE) / T_, S))
        log.line("ΔH_mix = H^E = {}", q(HE, E))
    res.section("Composition", _comp_rows(subs, xs, "x"))
    res.section("Activity coefficients", [
        row(f"γ {s.name}", f"{num(math.exp(l))}   ln γ = {num(l)}   a = {num(x * math.exp(l))}")
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
    if log.on() and r.xs and r.ys and None not in r.xs + r.ys:
        log.step("Result")
        log.line("T = {}, P = {}", q(r.T, T), q(r.P, P))
        for s, x, y in zip(subs, r.xs, r.ys):
            log.line("{}: x = {}, y = {}, K = {}", s.name, num(x), num(y), num(y / x) if x else "-")
        if len(subs) == 2 and r.xs[0] not in (0, 1) and r.ys[0] / r.xs[0] != r.ys[1] / r.xs[1]:
            log.line("Relative volatility α12 = K1/K2 = {}", num((r.ys[0] / r.xs[0]) / (r.ys[1] / r.xs[1])))
    titles = {"bubble-p": "Bubble pressure", "dew-p": "Dew pressure", "bubble-t": "Bubble temperature",
              "dew-t": "Dew temperature", "flash": "Isothermal flash (T, P)"}
    res = Result(f"{titles[kind]} - {vle.METHODS[method].split(':')[0]}")
    rows = res.section("Result", [row("T", r.T, T), row("P", r.P, P)])
    if r.VF is not None:
        rows.append(row("Vapor fraction V/F", num(r.VF)))
    comp = res.section("Phase compositions")
    for i, s in enumerate(subs):
        x = num(r.xs[i]) if r.xs and r.xs[i] is not None else "-"
        y = num(r.ys[i]) if r.ys and r.ys[i] is not None else "-"
        k = f"   K = {num(r.ys[i] / r.xs[i])}" if x != "-" and y != "-" and r.xs[i] > 0 else ""
        comp.append(row(s.name, f"x = {x}   y = {y}{k}"))
    if r.gammas:
        res.section("Model values", [
            row(s.name, f"γ = {num(g)}   Φ = {num(f)}   Psat = {fmt(p, P)}")
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
        T_d, P_d = (r.T, None) if fixed_T else (None, r.P)
        val = r.P if fixed_T else r.T
        pts, labels = [], []
        for comp, lab in ((r.xs, "x"), (r.ys, "y")):
            if comp and comp[0] is not None:
                pts.append((comp[0], val))
                labels.append(lab)
        if kind == "flash":
            pts.append((zs[0], val))
            labels.append("z")
        res.diagram = _binary_diagram(subs, T_d, P_d, *_binary_curve(solver, T_d, P_d, 26), pts, labels)
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
        rows.append(row(f"x = {x:.3f}", f"y = {num(y)}   {'P' if T_ else 'T'} = {fmt(v, kind)}"))
    az = _azeotrope(xs, ys, vals)
    if log.on():
        log.step("Example: one point of the curve, x1 = 0.5")
        solver.bubble_P(T_, [0.5, 0.5]) if T_ else solver.bubble_T(P_, [0.5, 0.5])
        log.step("{} curve: bubble {} at {} values of x1 from 0 to 1 (same method as above)", "Pxy" if T_ else "Txy",
                 "P" if T_ else "T", len(xs))
        log.table(["x1", "y1", "y1 − x1", "P" if T_ else "T"],
                  [[f"{x:.3f}", num(y), num(y - x), q(v, kind)] for x, y, v in zip(xs, ys, vals)])
        log.step("Azeotrope: where y1 − x1 changes sign (linear interpolation)")
        if az:
            log.line("x1 = y1 = {}, {} = {}", num(az[0]), "P" if T_ else "T", q(az[1], kind))
        else:
            log.line("y1 − x1 never changes sign: no azeotrope")
    if az:
        res.section("Azeotrope", [row("x = y", num(az[0])), row("P" if T_ else "T", fmt(az[1], kind))])
    else:
        res.notes.append("No azeotrope found.")
    res.plot = lambda: _plot_binary(solver, subs, T_, P_)
    res.plot_name = f"{'pxy' if T_ else 'txy'}_{'_'.join(_names(subs))}"
    res.diagram = _binary_diagram(subs, T_, P_, xs, ys, vals, [az] if az else [], ["az"] if az else [])
    if res.diagram and az:
        res.diagram["legend"].append(("point", "az azeotrope"))
    return res


def _binary_diagram(subs, T_, P_, xs, ys, vals, points, labels):
    """Pxy or Txy: bubble and dew curves, tie line between the marked x and y."""
    if not xs:
        return None
    curves = [list(zip(xs, vals)), {"pts": list(zip(ys, vals)), "style": "alt"}]
    legend = [("accent", "bubble (x)"), ("alt", "dew (y)")]
    if len(points) >= 2:
        curves.append({"pts": sorted(points[:2]), "style": "dim-dashed"})
        legend.append(("dim-dashed", "tie line"))
    title = f"{'Pxy at ' + fmt(T_, T) if T_ else 'Txy at ' + fmt(P_, P)}"
    return dict(title=title, curves=curves, points=points, labels=labels, xlim=(0, 1),
                axes=((f"x, y {subs[0].name}", D), ("P" if T_ else "T", P if T_ else T)), legend=legend)


@log.quiet
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
    with log.muted():
        sol = _lle_solve(ge, T_)
    res = Result(f"Liquid-liquid equilibrium ({activity.MODELS[model]})")
    if log.on():
        log.step("Equal activities in both liquids at T = {}", q(T_, T))
        log.line("x1α γ1α = x1β γ1β and x2α γ2α = x2β γ2β, unknowns x1α and x1β")
        log.line("Solved with fsolve from the guesses (0.01, 0.99), (0.05, 0.9), (0.1, 0.8), (0.001, 0.999)")
        x_ = np.linspace(0.02, 0.98, 9)
        log.line("Stability check, ΔG_mix/RT = Σ xi ln xi + G^E/RT (a split exists where the curve is not convex):")
        log.table(["x1", "ΔG_mix/RT"], [[f"{a:.3f}", num(ge.GE(T_, [a, 1 - a]) / (R * T_) + a * math.log(a) + (1 - a) * math.log(1 - a))] for a in x_])
        if sol is None:
            log.line("No solution with x1α ≠ x1β: a single liquid phase")
        else:
            a, b = sol
            for name, x1 in (("α", a), ("β", b)):
                log.step("Phase {}: x1 = {}", name, num(x1))
                ge.explain(T_, [x1, 1 - x1])
            ga, gb = ge.gammas(T_, [a, 1 - a]), ge.gammas(T_, [b, 1 - b])
            log.step("Check: activities equal in both phases")
            log.line("{}: {} × {} = {}   vs   {} × {} = {}", subs[0].name, num(a), num(ga[0]), num(a * ga[0]),
                     num(b), num(gb[0]), num(b * gb[0]))
            log.line("{}: {} × {} = {}   vs   {} × {} = {}", subs[1].name, num(1 - a), num(ga[1]),
                     num((1 - a) * ga[1]), num(1 - b), num(gb[1]), num((1 - b) * gb[1]))
    if sol is None:
        res.section("Result", [row("Phases", "single liquid phase (miscible) at this T")])
    else:
        a, b = sol
        g = ge.gammas(T_, [a, 1 - a])
        res.section("Coexisting liquid phases", [
            row(f"Phase α: x {subs[0].name}", num(a)), row(f"Phase β: x {subs[0].name}", num(b)),
            row(f"γ {subs[0].name} (α)", num(g[0])), row(f"γ {subs[1].name} (α)", num(g[1])),
        ])
    res.notes += ge.notes
    res.notes.append("Solved from xα·γα = xβ·γβ for both species.")
    res.plot = lambda: _plot_lle(ge, subs, T_)
    res.plot_name = f"lle_{'_'.join(_names(subs))}"
    res.diagram = _lle_diagram(ge, subs, T_, sol)
    return res


@log.quiet
def _lle_diagram(ge, subs, T_, sol):
    pts = [(s[0], s[1], t) for t in np.linspace(T_ - 60, T_ + 60, 21) if (s := _lle_solve(ge, t))]
    if not pts:
        return None
    a, b, ts = zip(*pts)
    points, labels, curves = [], [], [list(zip(a, ts)), list(zip(b, ts))]
    legend = [("accent", "binodal curve")]
    if sol:
        points, labels = [(sol[0], T_), (sol[1], T_)], ["α", "β"]
        curves.append({"pts": points, "style": "dim-dashed"})
        legend.append(("dim-dashed", "tie line"))
    return dict(title=f"Binodal curve, x {subs[0].name}", curves=curves, points=points, labels=labels,
                xlim=(0, 1), axes=((f"x {subs[0].name}", D), ("T", T)), legend=legend)


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
    if log.on():
        log.step("Liquidus of each solid (pure solid in equilibrium with the liquid)")
        for s in subs:
            log.line("{}: Tm = {}, ΔHfus = {}", s.name, q(s.Tm, T), q(s.Hfus, E))
        log.line("ln(xi γi) = −(ΔHfus,i/R)(1/T − 1/Tm,i), solved for T at each x (Brent's method)")
        log.table(["x1", f"T (solid {subs[0].name})", f"T (solid {subs[1].name})"],
                  [[f"{x:.2f}", q(liquidus(0, x), T) if x > 0 else "-", q(liquidus(1, 1 - x), T) if x < 1 else "-"]
                   for x in np.linspace(0, 1, 11)])
        log.step("Eutectic: both liquidus curves meet, T1(x) = T2(x) (Brent's method)")
        log.line("x1 = {}, T = {}", num(xe), q(Te, T))
        g = ge.gammas(Te, [xe, 1 - xe])
        for i, s in enumerate(subs):
            xi = xe if i == 0 else 1 - xe
            log.line("Check {}: ln(x γ) = ln({} × {}) = {};  −ΔHfus/R(1/T − 1/Tm) = {}", s.name, num(xi), num(g[i]),
                     num(math.log(xi * g[i])), num(-s.Hfus / R * (1 / Te - 1 / s.Tm)))
        ge.explain(Te, [xe, 1 - xe])
    res.section("Eutectic point", [row(f"x {subs[0].name}", num(xe)), row("T eutectic", Te, T)])
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
    a = np.linspace(xe, 0.9999, 25)
    b = np.linspace(1e-4, xe, 25)
    with log.muted():
        res.diagram = _sle_diagram(subs, xe, Te, a, b, liquidus)
    return res


def _sle_diagram(subs, xe, Te, a, b, liquidus):
    return dict(
        title="Solid-liquid diagram", xlim=(0, 1), points=[(xe, Te)], labels=["E"],
        curves=[[(x, liquidus(0, x)) for x in a], {"pts": [(x, liquidus(1, 1 - x)) for x in b], "style": "alt"},
                {"pts": [(0, Te), (1, Te)], "style": "dim-dashed"}],
        axes=((f"x {subs[0].name}", D), ("T", T)),
        legend=[("accent", f"solid {subs[0].name}"), ("alt", f"solid {subs[1].name}"), ("dim-dashed", "eutectic")])


def solubility(solute, T_, solvent=None, model="ideal", params=None):
    subs = [solute] + ([solvent] if solvent else [])
    if not (solute.Tm and solute.Hfus):
        raise ValueError(f"No melting data for {solute.name}")
    ideal = math.exp(-solute.Hfus / R * (1 / T_ - 1 / solute.Tm))
    res = Result(f"Solubility of solid {solute.name}")
    if log.on():
        log.step("Ideal solubility (γ = 1, pure solid, ΔCp neglected)")
        log.line("Tm = {}, ΔHfus = {}", q(solute.Tm, T), q(solute.Hfus, E))
        log.line("ln x = −(ΔHfus/R)(1/T − 1/Tm) = −({} / {})(1/{} − 1/{}) = {}", q(solute.Hfus, E), R,
                 f"{num(T_)} K", f"{num(solute.Tm)} K", num(math.log(ideal)))
        log.line("x = exp({}) = {}", num(math.log(ideal)), num(ideal))
    rows = res.section("Result", [row("Ideal solubility x", num(min(ideal, 1)))])
    if solvent and model != "ideal":
        ge = activity.GEModel(subs, model, params)
        f = lambda x: math.log(x * ge.gammas(T_, [x, 1 - x])[0]) - math.log(ideal)
        try:
            x = brentq(f, 1e-10, 1 - 1e-10)
            rows.append(row(f"Solubility with {model}", num(x)))
            log.step("Solubility with {} in {}", model, solvent.name)
            log.line("Solve x·γ(x) = x_ideal = {} for x (Brent's method)", num(ideal))
            ge.explain(T_, [x, 1 - x])
            log.line("x = {}; check x·γ = {}", num(x), num(x * ge.gammas(T_, [x, 1 - x])[0]))
        except ValueError:
            rows.append(row(f"Solubility with {model}", "no solution"))
    if T_ >= solute.Tm:
        res.notes.append("T is above the melting point: the solute is liquid.")
    ideal_x = lambda t: min(1.0, math.exp(-solute.Hfus / R * (1 / t - 1 / solute.Tm)))
    ts = np.linspace(min(T_, 0.6 * solute.Tm), solute.Tm, 30)
    res.diagram = dict(title=f"Solubility of {solute.name} (mole fraction)", curves=[[(t, ideal_x(t)) for t in ts]],
                       points=[(T_, ideal_x(T_))], labels=[""], ylim=(0, 1), xlim=(ts[0], ts[-1]),
                       axes=(("T", T), ("x", D)),
                       legend=[("accent", "ideal solubility"), ("point", "this T")])
    return res


# Osmotic pressure

def osmotic(solvent, solute, T_, x_solute, model="ideal", params=None):
    subs = [solvent, solute]
    xs = [1 - x_solute, x_solute]
    ge = activity.GEModel(subs, model, params)
    g = ge.gammas(T_, xs)[0]
    Vl = solvent.Vliq(T_)
    Pi = -R * T_ / Vl * math.log(xs[0] * g)
    res = Result("Osmotic pressure")
    if log.on():
        log.step("Solvent activity")
        ge.explain(T_, xs)
        log.line("a_solvent = x γ = {} × {} = {}", num(xs[0]), num(g), num(xs[0] * g))
        log.step("Osmotic pressure")
        log.line("V_solvent (liquid, databank) = {}", q(Vl, V))
        log.line("Π = −(RT/V) ln(a) = −({} · {} / {}) × ln({}) = {}", R, K(T_), q(Vl, V), num(xs[0] * g), q(Pi, P))
        log.line("Dilute limit (van 't Hoff): Π ≈ x_solute RT/V = {} × {} · {} / {} = {}", num(x_solute), R,
                 q(T_, T), q(Vl, V), q(x_solute * R * T_ / Vl, P))
    res.section("Π = -(RT/V_solvent) ln(x_solvent γ_solvent)", [
        row("Solvent molar volume", Vl, V), row("γ solvent", num(g)),
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
    if log.on():
        log.step("Experimental γ from each point (modified Raoult)")
        log.line("Psat used: {} = {}, {} = {}", subs[0].name, q(ps[0], P), subs[1].name, q(ps[1], P))
        log.line("γ1 = y1P/(x1Psat1), γ2 = y2P/(x2Psat2), G^E/RT = x1 ln γ1 + x2 ln γ2")
        log.table(["x1", "y1", "P", "ln γ1", "ln γ2", "G^E/RT", "G^E/(x1x2RT)"],
                  [[f"{a:.4f}", f"{y1:.4f}", q(Pexp, P), num(l1), num(l2), num(g_), num(yy)]
                   for (a, l1, l2, Pexp, y1), g_, yy in zip(pts, gert, y)])
    if model == "margules":
        A = np.vstack([x, 1 - x]).T
        A21, A12 = np.linalg.lstsq(A, y, rcond=None)[0]
        params = (A12, A21)
        log.step("Margules: G^E/(x1x2RT) = A21·x1 + A12·x2, linear least squares")
        log.line("A12 = {} (value at x1 = 0), A21 = {} (value at x1 = 1)", num(A12), num(A21))
    else:
        A = np.vstack([x, 1 - x]).T
        inv = 1 / y
        c1, c2 = np.linalg.lstsq(A, inv, rcond=None)[0]
        params = (1 / c2, 1 / c1)
        log.step("van Laar: x1x2RT/G^E = x1/A'21 + x2/A'12, linear least squares")
        log.line("1/A'21 = {}, 1/A'12 = {}  ->  A'12 = {}, A'21 = {}", num(c1), num(c2), num(params[0]), num(params[1]))
    solver = vle.solver(subs, "modified-raoult", model, list(params))
    solver.psats = lambda T, ps=ps: ps
    with log.muted():
        calc = [solver.bubble_P(T_, [p[0], 1 - p[0]]) for p in pts]
    dev = [(c.P - p[3]) for c, p in zip(calc, pts)]
    rms = math.sqrt(sum(d * d for d in dev) / len(dev))
    if log.on():
        log.step("Check: bubble P with the fitted model (modified Raoult) at each x1")
        log.table(["x1", "P data", "P model", "ΔP", "y1 data", "y1 model"],
                  [[f"{p[0]:.4f}", q(p[3], P), q(c.P, P), q(d, P), f"{p[4]:.4f}", num(c.ys[0])]
                   for p, c, d in zip(pts, calc, dev)])
        log.line("RMS deviation = √(Σ ΔP²/n) = {}", q(rms, P))
    res = Result(f"{model.capitalize()} parameters from P-x-y data")
    res.section("Fitted parameters", [
        row("A12" if model == "margules" else "A'12", num(params[0])),
        row("A21" if model == "margules" else "A'21", num(params[1])),
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
            for v in rows[-1]:
                mark_given(v)
    return rows

