"""Chemical reaction equilibrium."""

import math

import numpy as np
from chemicals.elements import simple_formula_parser
from scipy.optimize import brentq, minimize

from thermo_utils import activity, reactions, substances, vle
from thermo_utils import steplog as log
from thermo_utils.result import Result, row
from thermo_utils.steplog import q
from thermo_utils.units import (
    DIMENSIONLESS as D, MOLAR_ENERGY as E, MOLAR_ENTROPY as S, PRESSURE as P, TEMPERATURE as T, R, display, fmt, from_si,
)

P0 = 1e5


def equilibrium_constant(reaction, T_):
    rx = reactions.parse(reaction)
    dG, dH = rx.dG(T_), rx.dH(T_)
    K = math.exp(-dG / (R * T_))
    K_vh = math.exp(-rx.dG0() / (R * 298.15) - rx.dH0() / R * (1 / T_ - 1 / 298.15))
    res = Result("Equilibrium constant")
    rx.explain(T_)
    if log.on():
        log.step("Short-cut van 't Hoff (ΔH° constant)")
        log.line("ln K = ln K298 − (ΔH°298/R)(1/T − 1/298.15) = {} − ({} / {})({} − {}) = {}",
                 f"{-rx.dG0() / (R * 298.15):.6g}", q(rx.dH0(), E), R, f"{1 / T_:.6g}", f"{1 / 298.15:.6g}",
                 f"{math.log(K_vh):.6g}")
        log.line("K = {} (vs {} with ΔCp)", f"{K_vh:.6g}", f"{K:.6g}")
    res.section(rx.pretty(), [
        row("ΔG° at 298.15 K", rx.dG0(), E), row("ΔH° at 298.15 K", rx.dH0(), E),
        row(f"ΔG° at {fmt(T_, T)}", dG, E), row(f"ΔH° at {fmt(T_, T)}", dH, E),
        row(f"ΔS° at {fmt(T_, T)}", rx.dS(T_), S),
        row("K (rigorous, ΔCp included)", f"{K:.6g}"),
        row("ln K", f"{math.log(K):.6g}"),
        row("K (short-cut van 't Hoff, ΔH constant)", f"{K_vh:.6g}"),
    ])
    res.notes.append("Standard state: ideal gas at 1 bar (g) or pure liquid at 1 bar (l).")

    def plot():
        import matplotlib.pyplot as plt
        Ts = np.linspace(max(250, T_ - 400), T_ + 400, 120)
        lnK = [-rx.dG(t) / (R * t) for t in Ts]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(1 / Ts, lnK)
        ax1.plot(1 / T_, math.log(K), "ko")
        ax1.set_xlabel("1/T (1/K)"); ax1.set_ylabel("ln K"); ax1.set_title("van 't Hoff plot")
        ax2.plot([from_si(t, display.temperature, T) for t in Ts], [rx.dG(t) / 1000 for t in Ts])
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel(f"T ({display.temperature})"); ax2.set_ylabel("ΔG° (kJ/mol)"); ax2.set_title("ΔG° vs T")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    res.plot = plot
    res.plot_name = "lnK"
    ts = np.linspace(max(250, T_ - 400), T_ + 400, 40)
    curve = [(t, -rx.dG(t) / (R * t)) for t in ts]
    curves, legend = [curve], [("accent", "ln K(T)"), ("point", "this T")]
    lnks = [c[1] for c in curve]
    if min(lnks) < 0 < max(lnks):
        curves.append({"pts": [(ts[0], 0), (ts[-1], 0)], "style": "dim-dashed"})
        legend.append(("dim-dashed", "K = 1"))
    res.diagram = dict(curves=curves, points=[(T_, math.log(K))], labels=[""], axes=(("T", T), ("ln K", D)),
                       xlim=(ts[0], ts[-1]),
                       legend=legend)
    return res


# Single reaction

def _parse_feed(text):
    feed = {}
    if not text:
        return feed
    for part in text.replace(";", ",").split(","):
        if "=" not in part:
            raise ValueError("Feed must look like 'CH4=1, H2O=2, N2=0.5' (moles).")
        k, v = part.split("=", 1)
        feed[substances.get(k.strip())] = float(v)
    return feed


def _setup(rx, feed):
    names = [sp.sub for sp in rx.species]
    n0 = [feed.pop(s, 0.0) for s in names]
    nu = [sp.nu for sp in rx.species]
    inert = list(feed.items())
    return names, n0, nu, inert


def single_reaction(reaction, T_, P_, feed, phase_model="ideal", curve=True):
    """Equilibrium extent of one reaction."""
    rx = reactions.parse(reaction, "l" if phase_model in ("ideal-liquid", "unifac") else "g")
    feed = _parse_feed(feed)
    subs, n0, nu, inert = _setup(rx, dict(feed))
    if sum(n0) <= 0:
        raise ValueError("Give the feed in moles of at least one reactant.")
    K = rx.K(T_)
    liquid = all(sp.phase == "l" for sp in rx.species)
    lo = max([-n / v for n, v in zip(n0, nu) if v > 0] + [-1e300])
    hi = min([n / -v for n, v in zip(n0, nu) if v < 0] + [1e300])
    if hi > 1e299 or lo < -1e299:
        raise ValueError("The feed must allow the reaction to proceed in both directions or one with limits.")
    span = hi - lo
    eps_lo, eps_hi = lo + 1e-12 * span, hi - 1e-12 * span
    ge = activity.GEModel(subs + [s for s, _ in inert], "unifac") if phase_model == "unifac" else None

    def comp(e):
        n = [a + v * e for a, v in zip(n0, nu)] + [v for _, v in inert]
        tot = sum(n)
        return [x / tot for x in n]

    def ln_activity(e):
        ys = comp(e)
        if liquid:
            g = ge.gammas(T_, ys) if ge else [1.0] * len(ys)
            return [math.log(max(y, 1e-300) * gi) for y, gi in zip(ys, g)][: len(subs)]
        phis = [1.0] * len(ys)
        if phase_model == "virial":
            phis = vle.virial_phis(subs + [s for s, _ in inert], T_, P_, ys)[0]
        return [math.log(max(y, 1e-300) * f * P_ / P0) for y, f in zip(ys, phis)][: len(subs)]

    tries = []

    def f(e):
        r = sum(v * la for v, la in zip(nu, ln_activity(e))) - math.log(K)
        tries.append([len(tries) + 1, f"{e:.10g}", f"{r:.6g}"])
        return r

    rx.explain(T_)
    if log.on():
        log.step("Mole balance with the extent ε")
        for s, a, v in zip(subs, n0, nu):
            log.line("n {} = {} {} {}·ε", s.name, f"{a:g}", "+" if v > 0 else "−", f"{abs(v):g}")
        for s, v in inert:
            log.line("n {} = {} (inert)", s.name, f"{v:g}")
        log.line("Σν = {}; ε between {} and {} (no negative moles)", f"{sum(nu):g}", f"{lo + 0.0:.6g}", f"{hi + 0.0:.6g}")
        log.step("Equilibrium condition")
        log.line({"ideal": "Π (yi P/P°)^νi = K  ->  Σ νi ln(yi P/P°) − ln K = 0",
                  "virial": "Π (yi φ̂i P/P°)^νi = K  ->  Σ νi ln(yi φ̂i P/P°) − ln K = 0 (φ̂ from the virial equation)",
                  "ideal-liquid": "Π xi^νi = K  ->  Σ νi ln xi − ln K = 0",
                  "unifac": "Π (xi γi)^νi = K  ->  Σ νi ln(xi γi) − ln K = 0 (γ from UNIFAC)"}[phase_model])
        log.line("ln K = {}; P/P° = {}", f"{math.log(K):.6g}", f"{P_ / P0:.6g}")
    f_lo, f_hi = f(eps_lo), f(eps_hi)
    if f_lo * f_hi > 0:
        eps = eps_hi if f_hi < 0 else eps_lo
        complete = True
        log.line("f(ε) has the same sign at both limits: the reaction goes to the limit ε = {}", f"{eps:.6g}")
    else:
        eps = brentq(f, eps_lo, eps_hi, xtol=1e-14, maxiter=500)
        complete = False
        log.line("Solve f(ε) = 0 with Brent's method:")
        log.table(["it", "ε", "f(ε)"], tries)
        log.line("ε = {} mol", f"{eps:.10g}")
    ys = comp(eps)
    res = Result("Single-reaction equilibrium")
    res.section(rx.pretty(), [
        row("T", T_, T), row("P", P_, P), row("K", f"{K:.6g}"), row("ΔG°(T)", rx.dG(T_), E),
        row("Equilibrium extent ε", f"{eps:.6g} mol"),
    ])
    key = next(i for i, v in enumerate(nu) if v < 0 and n0[i] > 0)
    conv = -nu[key] * eps / n0[key]
    if log.on():
        log.step("Equilibrium composition")
        tot = sum(n0) + sum(nu) * eps + sum(v for _, v in inert)
        log.line("Total moles = Σn0 + Σν·ε = {}", f"{tot:.6g}")
        for i, s in enumerate(subs):
            log.line("{}: n = {} {} {}×{} = {}, {} = {}", s.name, f"{n0[i]:g}", "+" if nu[i] > 0 else "−",
                     f"{abs(nu[i]):g}", f"{eps:.6g}", f"{n0[i] + nu[i] * eps:.6g}", "x" if liquid else "y", f"{ys[i]:.6g}")
        la = ln_activity(eps)
        log.line("Check: Σ νi ln ai = {} = ln K = {}", f"{sum(v * a for v, a in zip(nu, la)):.6g}", f"{math.log(K):.6g}")
        log.line("Conversion of {} = −ν·ε/n0 = {} × {} / {} = {} %", subs[key].name, f"{-nu[key]:g}", f"{eps:.6g}",
                 f"{n0[key]:g}", f"{conv * 100:.6g}")
    res.section("Equilibrium", [row(f"Conversion of {subs[key].name}", f"{conv * 100:.5g} %")] + [
        row(s.name, f"n = {n0[i] + nu[i] * eps:.6g} mol   {'x' if liquid else 'y'} = {ys[i]:.6f}")
        for i, s in enumerate(subs)
    ] + [row(f"{s.name} (inert)", f"n = {v:.6g} mol   y = {ys[len(subs) + j]:.6f}") for j, (s, v) in enumerate(inert)])
    if complete:
        res.notes.append("K is so far from 1 that the reaction goes to the limit of the feed.")
    res.notes.append({"ideal": "Ideal-gas mixture: Π(yi P/P°)^νi = K.",
                      "virial": "Gas with virial fugacity coefficients: Π(yi φ̂i P/P°)^νi = K.",
                      "ideal-liquid": "Ideal liquid solution: Π(xi)^νi = K.",
                      "unifac": "Liquid with UNIFAC activity coefficients: Π(xi γi)^νi = K."}[phase_model])

    def plot():
        import matplotlib.pyplot as plt
        Ts = np.linspace(max(250, T_ - 300), T_ + 300, 60)
        xs = []
        for t in Ts:
            try:
                xs.append((from_si(t, display.temperature, T), single_reaction_conv(reaction, t, P_, feed, phase_model)))
            except Exception:
                pass
        fig, ax = plt.subplots(figsize=(8, 5))
        if xs:
            ax.plot(*zip(*xs))
        ax.plot(from_si(T_, display.temperature, T), conv * 100, "ko")
        ax.set_xlabel(f"T ({display.temperature})"); ax.set_ylabel(f"Conversion of {subs[key].name} (%)")
        ax.set_title(f"Equilibrium conversion at {fmt(P_, P)}"); ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    res.plot = plot
    res.plot_name = "conversion"
    if curve:
        with log.muted():
            res.diagram = _conversion_diagram(reaction, T_, P_, feed, phase_model, subs, key, conv)
    res.data = {"extent": eps, "conversion": conv}
    return res


def _conversion_diagram(reaction, T_, P_, feed, phase_model, subs, key, conv):
    if True:
        pts = []
        for t in np.linspace(max(250, T_ - 300), T_ + 300, 25):
            try:
                pts.append((t, single_reaction_conv(reaction, t, P_, feed, phase_model)))
            except Exception:
                pass
        return dict(title=f"Equilibrium conversion of {subs[key].name}", curves=[pts],
                    points=[(T_, conv * 100)], labels=[""], ylim=(0, 100), xlim=(pts[0][0], pts[-1][0]),
                    axes=(("T", T), ("X (%)", D)),
                    legend=[("accent", f"equilibrium conversion at {fmt(P_, P)}"), ("point", "this T")])


def single_reaction_conv(text, T_, P_, feed_dict, phase_model):
    feed = ", ".join(f"{s.cas}={v}" for s, v in feed_dict.items())
    return single_reaction(text, T_, P_, feed, phase_model, curve=False).data["conversion"] * 100


# Multi-reaction (Gibbs energy minimization)

def _g0(sub, T_):
    """Molar Gibbs energy of the ideal gas at T and 1 bar, elements in standard state at 298.15 K."""
    substances.require(sub, "Hf", "S0")
    H = sub.Hf + sub.H_ig(298.15, T_)
    S_ = sub.S0 + sub.S_ig(298.15, T_)
    return H - T_ * S_


def gibbs_minimization(species, T_, P_, feed):
    """Equilibrium of a gas mixture by minimizing G with element balances."""
    feed_d = _parse_feed(feed)
    for s in feed_d:
        if s not in species:
            species = species + [s]
    elements = sorted({el for s in species for el in simple_formula_parser(s.formula)})
    A = np.array([[simple_formula_parser(s.formula).get(el, 0) for s in species] for el in elements], float)
    n0 = np.array([feed_d.get(s, 0.0) for s in species])
    b = A @ n0
    g = np.array([_g0(s, T_) / (R * T_) for s in species])
    lnP = math.log(P_ / P0)
    scale = max(n0.sum(), 1e-12)

    def G(n):
        n = np.maximum(n, 1e-30)
        return float(np.sum(n * (g + np.log(n / n.sum()) + lnP))) / scale

    def grad(n):
        n = np.maximum(n, 1e-30)
        return (g + np.log(n / n.sum()) + lnP) / scale

    if log.on():
        log.step("Element balances: Σ aik nk = bi")
        log.table(["element"] + [s.formula for s in species] + ["b (fed)"],
                  [[el] + [f"{v:g}" for v in A[i]] + [f"{b[i]:.6g}"] for i, el in enumerate(elements)])
        log.step("Standard Gibbs energies at T = {} (ideal gas, 1 bar)", q(T_, T))
        log.line("G°i = ΔHf°i + ∫Cp dT − T(S°i + ∫Cp/T dT), elements at 298.15 K as reference")
        rows = []
        for s in species:
            H = s.Hf + s.H_ig(298.15, T_)
            S_ = s.S0 + s.S_ig(298.15, T_)
            rows.append([s.formula, q(s.Hf, E), q(H, E), q(S_, S), q(H - T_ * S_, E), f"{(H - T_ * S_) / (R * T_):.6g}"])
        log.table(["species", "ΔHf°", "H(T)", "S(T)", "G°(T)", "G°/RT"], rows)
        log.step("Minimize G/RT = Σ nk [G°k/RT + ln(yk P/P°)] subject to the element balances")
        log.line("Method: SLSQP (sequential quadratic programming), nk > 0; ln(P/P°) = {}", f"{lnP:.6g}")
    start = np.linalg.lstsq(A, b, rcond=None)[0]
    start = np.maximum(start, 0) + 0.1 * scale / len(species)
    start = start * 0.5 + n0 * 0.5 + 1e-6
    sol = minimize(G, start, jac=grad, method="SLSQP", bounds=[(1e-20, None)] * len(species),
                   constraints=[{"type": "eq", "fun": lambda n: (A @ n - b) / scale, "jac": lambda n: A / scale}],
                   options={"ftol": 1e-15, "maxiter": 1000})
    if not sol.success:
        raise ValueError(f"Gibbs minimization did not converge: {sol.message}")
    n = sol.x
    y = n / n.sum()
    res = Result("Multi-reaction equilibrium (Gibbs energy minimization)")
    res.section("Conditions", [row("T", T_, T), row("P", P_, P),
                               row("Elements", ", ".join(elements)),
                               row("Total moles at equilibrium", f"{n.sum():.6g} mol")])
    res.section("Equilibrium composition", [
        row(s.name, f"n = {ni:.6g} mol   y = {yi:.6f}   (fed {n0i:g})") for s, ni, yi, n0i in zip(species, n, y, n0)
    ])
    lam = g + np.log(y) + lnP
    if log.on():
        log.line("Converged in {} iterations: {}", sol.nit, sol.message)
        log.table(["species", "fed", "n", "y", "G°/RT + ln(yP/P°)"],
                  [[s.formula, f"{a:g}", f"{ni:.6g}", f"{yi:.6g}", f"{li:.6g}"] for s, a, ni, yi, li in zip(species, n0, n, y, lam)])
        pi = np.linalg.lstsq(A.T, lam, rcond=None)[0]
        log.step("Check: Lagrange multipliers (element potentials λ/RT)")
        log.line("At the minimum G°k/RT + ln(yk P/P°) = Σi λi aik for every species with nk > 0")
        log.line("λ/RT: {}", ", ".join(f"{el} = {v:.6g}" for el, v in zip(elements, pi)))
        log.table(["species", "Σ λi aik", "G°/RT + ln(yP/P°)", "difference"],
                  [[s.formula, f"{a_:.6g}", f"{li:.6g}", f"{a_ - li:.2e}"] for s, a_, li in zip(species, A.T @ pi, lam)])
        log.line("Element balance error: max |A n − b| = {} mol", f"{np.max(np.abs(A @ n - b)):.2e}")
    res.notes.append("Ideal-gas mixture; Lagrange multipliers satisfied when Σ λk aik = Gi/RT for every species.")
    res.notes.append(f"Max element-balance error: {np.max(np.abs(A @ n - b)):.2e} mol.")
    res.data = {"n": dict(zip([s.name for s in species], n)), "mu": lam}
    return res
