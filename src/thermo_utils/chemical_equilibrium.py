"""Chemical reaction equilibrium."""

import math

import numpy as np
from chemicals.elements import simple_formula_parser
from scipy.optimize import brentq, minimize

from thermo_utils import activity, reactions, substances, vle
from thermo_utils.result import Result, row
from thermo_utils.units import (
    MOLAR_ENERGY as E, MOLAR_ENTROPY as S, PRESSURE as P, TEMPERATURE as T, R, display, fmt, from_si,
)

P0 = 1e5


def equilibrium_constant(reaction, T_):
    rx = reactions.parse(reaction)
    dG, dH = rx.dG(T_), rx.dH(T_)
    K = math.exp(-dG / (R * T_))
    K_vh = math.exp(-rx.dG0() / (R * 298.15) - rx.dH0() / R * (1 / T_ - 1 / 298.15))
    res = Result("Equilibrium constant")
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


def single_reaction(reaction, T_, P_, feed, phase_model="ideal"):
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

    f = lambda e: sum(v * la for v, la in zip(nu, ln_activity(e))) - math.log(K)
    f_lo, f_hi = f(eps_lo), f(eps_hi)
    if f_lo * f_hi > 0:
        eps = eps_hi if f_hi < 0 else eps_lo
        complete = True
    else:
        eps = brentq(f, eps_lo, eps_hi, xtol=1e-14, maxiter=500)
        complete = False
    ys = comp(eps)
    res = Result("Single-reaction equilibrium")
    res.section(rx.pretty(), [
        row("T", T_, T), row("P", P_, P), row("K", f"{K:.6g}"), row("ΔG°(T)", rx.dG(T_), E),
        row("Equilibrium extent ε", f"{eps:.6g} mol"),
    ])
    key = next(i for i, v in enumerate(nu) if v < 0 and n0[i] > 0)
    conv = -nu[key] * eps / n0[key]
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
    res.data = {"extent": eps, "conversion": conv}
    return res


def single_reaction_conv(text, T_, P_, feed_dict, phase_model):
    feed = ", ".join(f"{s.cas}={v}" for s, v in feed_dict.items())
    return single_reaction(text, T_, P_, feed, phase_model).data["conversion"] * 100


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
    res.notes.append("Ideal-gas mixture; Lagrange multipliers satisfied when Σ λk aik = Gi/RT for every species.")
    res.notes.append(f"Max element-balance error: {np.max(np.abs(A @ n - b)):.2e} mol.")
    res.data = {"n": dict(zip([s.name for s in species], n)), "mu": lam}
    return res
