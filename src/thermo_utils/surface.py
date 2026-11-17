"""Surface phenomena: surface tension, Kelvin, adsorption, micelles."""

import csv
import math
import re

import numpy as np
from scipy.optimize import curve_fit

from thermo_utils import steplog as log
from thermo_utils.result import Result, row
from thermo_utils.steplog import K, q
from thermo_utils.units import (
    CONCENTRATION as C, LENGTH as L, MOLAR_ENERGY as E, MOLAR_ENTROPY as S, MOLAR_VOLUME as V,
    PRESSURE as P, SURFACE_TENSION as ST, TEMPERATURE as T, DEFAULT_INPUT_UNITS, R, display, fmt, from_si,
    to_si,
)

NA = 6.02214076e23


# Helpers

def sigma(sub, T_):
    try:
        model = sub._correlations.SurfaceTensions[0]
        value = model(T_)
    except Exception:
        value = None
    if value is None:
        raise ValueError(f"No surface tension data for {sub.name}.")
    return value


def _rho_vap(sub, T_):
    try:
        return sub.Psat(T_) * sub.MW / 1000 / (R * T_)
    except Exception:
        return 0.0


def _surface_entropy(value):
    return f"{value * 1e3:.6g} mN/(m·K)"


def read_columns(path, kinds):
    """Two numeric columns; units read from headers like 'P (kPa)'."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    header, body = rows[0], rows[1:]
    try:
        float(header[0].replace(",", "."))
        body, header = rows, [""] * len(kinds)
    except ValueError:
        pass
    units = []
    for h, _ in zip(header, kinds):
        m = re.search(r"\(([^)]+)\)", h)
        units.append(m.group(1) if m else None)
    cols = [[] for _ in kinds]
    for r in body:
        for i, (kind, unit) in enumerate(zip(kinds, units)):
            value = float(r[i].replace(",", "."))
            cols[i].append(kind(value, unit) if callable(kind) else to_si(value, unit or DEFAULT_INPUT_UNITS[kind], kind))
    return [np.array(c, dtype=float) for c in cols]


# Surface tension

def surface_tension(sub, T_):
    from chemicals.interface import Brock_Bird
    s0 = sigma(sub, T_)
    dT = 0.5
    dsdT = (sigma(sub, T_ + dT) - sigma(sub, T_ - dT)) / (2 * dT)
    res = Result(f"Surface tension of {sub.name}")
    if log.on():
        m = sub._correlations.SurfaceTensions[0]
        log.step("Surface tension from the databank correlation ({})", getattr(m, "method", "-"))
        log.line("σ(T) = {}", q(s0, ST))
        log.line("dσ/dT by central difference (±{} K): [σ({}) − σ({})]/{} K = [{} − {}]/{} = {}", dT, q(T_ + dT, T),
                 q(T_ - dT, T), 2 * dT, q(sigma(sub, T_ + dT), ST), q(sigma(sub, T_ - dT), ST), 2 * dT, _surface_entropy(dsdT))
        log.step("Surface thermodynamics (per unit area)")
        log.line("s = −dσ/dT = {}", _surface_entropy(-dsdT))
        log.line("h = σ − T·dσ/dT = {} − {} × {} = {}", q(s0, ST), K(T_), _surface_entropy(dsdT), q(s0 - T_ * dsdT, ST))
        log.step("Estimates")
        try:
            log.line("Brock-Bird: σ = Pc^(2/3)Tc^(1/3)Q(1 − Tr)^(11/9), Q = 0.1196[1 + Tbr ln(Pc/1.01325)/(1 − Tbr)] − 0.279")
            log.line("  Tr = {}, Tbr = {}  ->  σ = {}", f"{T_ / sub.Tc:.6g}", f"{sub.Tb / sub.Tc:.6g}",
                     q(Brock_Bird(T_, sub.Tb, sub.Tc, sub.Pc), ST))
        except Exception:
            log.line("Brock-Bird: not available")
        par = (sub._constants.Parachors or [None])[0]
        if par:
            try:
                dens = 1 / sub.Vliq(T_) - _rho_vap(sub, T_) / (sub.MW / 1000)
                log.line("Macleod-Sugden: σ^¼ = [P](ρL − ρV); [P] = {}, ρL − ρV = {} mol/m³  ->  σ = {}", f"{par:.6g}",
                         f"{dens:.6g}", q((par * dens) ** 4, ST))
            except Exception:
                pass
    res.section(f"At {fmt(T_, T)}", [
        row("σ", s0, ST),
        row("dσ/dT", _surface_entropy(dsdT)),
        row("Surface entropy s = −dσ/dT", _surface_entropy(-dsdT)),
        row("Surface enthalpy h = σ − T·dσ/dT", s0 - T_ * dsdT, ST),
    ])
    est = res.section("Estimates")
    try:
        est.append(row("Brock-Bird", Brock_Bird(T_, sub.Tb, sub.Tc, sub.Pc), ST))
    except Exception:
        pass
    parachor = (sub._constants.Parachors or [None])[0]
    if parachor:
        try:
            dens = 1 / sub.Vliq(T_) - _rho_vap(sub, T_) / (sub.MW / 1000)
            est.append(row("Macleod-Sugden (parachor)", (parachor * dens) ** 4, ST))
        except Exception:
            pass
    if not est:
        res.sections.pop()
    res.notes.append("h and s are per unit area (J/m² and J/(m²·K)).")
    res.notes.append("The estimates are for nonpolar liquids; they fail for water and alcohols.")

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        Ts = np.linspace(max(sub.Tm or 0, 0.4 * sub.Tc), 0.99 * sub.Tc, 150)
        pts = []
        for t in Ts:
            try:
                pts.append((from_si(t, display.temperature, T), from_si(sigma(sub, t), display.surface_tension, ST)))
            except Exception:
                pass
        ax.plot(*zip(*pts))
        ax.plot(from_si(T_, display.temperature, T), from_si(s0, display.surface_tension, ST), "ko")
        ax.set_xlabel(f"T ({display.temperature})"); ax.set_ylabel(f"σ ({display.surface_tension})")
        ax.set_title(f"Surface tension of {sub.name}"); ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    res.plot, res.plot_name = plot, f"sigma_{sub.name}"
    return res


# Kelvin equation

def kelvin(sub, T_, r, kind="drop"):
    s0 = sigma(sub, T_)
    Vl = sub.Vliq(T_)
    Ps = sub.Psat(T_)
    sign = 1 if kind == "drop" else -1
    ratio = math.exp(sign * 2 * s0 * Vl / (r * R * T_))
    res = Result(f"Kelvin equation: {sub.name}")
    if log.on():
        log.step("Data at T = {}", q(T_, T))
        log.line("σ = {}, V_liq = {}, Psat (flat) = {}", q(s0, ST), q(Vl, V), q(Ps, P))
        log.step("Kelvin equation for a {}", "drop (convex, + sign)" if sign > 0 else "bubble/capillary (concave, − sign)")
        log.line("ln(P/Psat) = {}2σV/(rRT) = {}2 × {} × {} / ({} × {} × {})", "+" if sign > 0 else "−", "+" if sign > 0 else "−",
                 f"{s0:.6g} N/m", f"{Vl:.6g} m³/mol", f"{r:.6g} m", R, f"{T_:.6g} K")
        log.line("ln(P/Psat) = {}  ->  P/Psat = {}", f"{math.log(ratio):.6g}", f"{ratio:.6g}")
        log.line("P = {} × {} = {}", q(Ps, P), f"{ratio:.6g}", q(Ps * ratio, P))
    res.section(f"{'Drop (convex)' if sign > 0 else 'Bubble or capillary (concave)'}, r = {fmt(r, L, 'nm')}", [
        row("σ", s0, ST), row("V liquid", Vl, V), row("Psat (flat surface)", Ps, P),
        row("P/Psat", f"{ratio:.6g}"), row("P (curved surface)", Ps * ratio, P),
    ])

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        rs = np.logspace(-9, -6, 200)
        k = 2 * s0 * Vl / (R * T_)
        ax.semilogx(rs * 1e9, np.exp(k / rs), label="drop")
        ax.semilogx(rs * 1e9, np.exp(-k / rs), label="bubble / capillary")
        ax.plot(r * 1e9, ratio, "ko")
        ax.axhline(1, color="gray", lw=0.8)
        ax.set_xlabel("r (nm)"); ax.set_ylabel("P/Psat"); ax.grid(alpha=0.3, which="both"); ax.legend()
        ax.set_title(f"Kelvin equation: {sub.name} at {T_:.2f} K")
        fig.tight_layout()
        return fig

    res.plot, res.plot_name = plot, f"kelvin_{sub.name}"
    return res


# Adsorption

def _q_to_mol_per_kg(value, unit):
    u = (unit or "mol/kg").lower().replace(" ", "")
    factors = {"mol/kg": 1.0, "mmol/g": 1.0, "mmol/kg": 1e-3, "mol/g": 1e3,
               "cm3/g": 1 / 22.414, "cm3stp/g": 1 / 22.414, "ml/g": 1 / 22.414, "mg/g": None}
    if u not in factors or factors[u] is None:
        raise ValueError(f"Unknown adsorbed-amount unit '{unit}'. Use mol/kg, mmol/g or cm3/g (STP).")
    return value * factors[u]


def _r2(y, yfit):
    ss = np.sum((y - yfit) ** 2)
    return 1 - ss / np.sum((y - np.mean(y)) ** 2)


def langmuir(p, qm, K):
    return qm * K * p / (1 + K * p)


def freundlich(p, K, n):
    return K * p ** (1 / n)


def bet(x, qm, c):
    return qm * c * x / ((1 - x) * (1 - x + c * x))


def adsorption(path, model="all", adsorbate=None, T_=None, P0=None, cross_section=None):
    p, q = read_columns(path, [P, _q_to_mol_per_kg])
    res = Result("Adsorption isotherm fit")
    if log.on():
        log.step("Data read from {} ({} points), q converted to mol/kg", path, len(p))
        log.table(["P (Pa)", "q (mol/kg)"], [[f"{a:.6g}", f"{b:.6g}"] for a, b in zip(p, q)])
        log.line("Each model fitted by nonlinear least squares (scipy curve_fit); R² = 1 − SS_res/SS_tot")
    fits = {}
    if model in ("langmuir", "all"):
        (qm, K), _ = curve_fit(langmuir, p, q, p0=[q.max() * 1.2, 1 / np.median(p)], maxfev=20000)
        fits["Langmuir"] = (lambda x, a=qm, b=K: langmuir(x, a, b), p)
        _log_fit("Langmuir: q = qm·K·P/(1 + K·P)", f"qm = {qm:.6g} mol/kg, K = {K:.6g} 1/Pa", p, q, langmuir(p, qm, K))
        res.section("Langmuir  q = qm·K·P/(1 + K·P)", [
            row("qm", f"{qm:.6g} mol/kg"), row("K", f"{K:.6g} 1/Pa"),
            row("R²", f"{_r2(q, langmuir(p, qm, K)):.5f}"),
        ])
    if model in ("freundlich", "all"):
        (Kf, n), _ = curve_fit(freundlich, p, q, p0=[q.mean() / np.mean(p) ** 0.5, 2], maxfev=20000)
        fits["Freundlich"] = (lambda x, a=Kf, b=n: freundlich(x, a, b), p)
        _log_fit("Freundlich: q = K·P^(1/n)", f"K = {Kf:.6g}, n = {n:.6g}", p, q, freundlich(p, Kf, n))
        res.section("Freundlich  q = K·P^(1/n)", [
            row("K", f"{Kf:.6g} mol/kg/Pa^(1/n)"), row("n", f"{n:.6g}"),
            row("R²", f"{_r2(q, freundlich(p, Kf, n)):.5f}"),
        ])
    if model in ("bet", "all"):
        if P0 is None and adsorbate is not None and T_ is not None:
            P0 = adsorbate.Psat(T_)
        if P0 is None:
            if model == "bet":
                raise ValueError("BET needs P0, or the adsorbate and T.")
        else:
            x = p / P0
            mask = (x > 0.05) & (x < 0.35)
            if mask.sum() < 3:
                mask = x < 0.9
            (qm, c), _ = curve_fit(bet, x[mask], q[mask], p0=[q[mask].mean(), 50], maxfev=20000)
            fits["BET"] = (lambda xx, a=qm, b=c, p0=P0: bet(xx / p0, a, b), p[mask])
            log.step("BET with P0 = {}{}", fmt(P0, P), " (Psat of the adsorbate at T)" if adsorbate is not None else "")
            log.line("x = P/P0; points with 0.05 < x < 0.35 used: {}", int(mask.sum()))
            _log_fit("BET: q = qm·c·x/[(1 − x)(1 − x + c·x)]", f"qm = {qm:.6g} mol/kg, c = {c:.6g}", p[mask], q[mask],
                     bet(x[mask], qm, c), step=False)
            rows = res.section("BET  q = qm·c·x/[(1 − x)(1 − x + c·x)],  x = P/P0", [
                row("P0", P0, P), row("qm (monolayer)", f"{qm:.6g} mol/kg"), row("c", f"{c:.6g}"),
                row("Points used (0.05 < x < 0.35)", str(int(mask.sum()))),
                row("R²", f"{_r2(q[mask], bet(x[mask], qm, c)):.5f}"),
            ])
            area = _cross_section(adsorbate, T_, cross_section)
            if area:
                log.line("Area per molecule = {} nm²{}", f"{area * 1e18:.4g}",
                         "" if cross_section else " (1.091·(V_liq/NA)^(2/3))")
                log.line("Specific area = qm·NA·area = {} mol/kg × {} × {} m² = {} m²/g", f"{qm:.6g}", f"{NA:.6g}",
                         f"{area:.4g}", f"{qm * NA * area / 1000:.6g}")
                rows.append(row("Area per adsorbed molecule", f"{area * 1e18:.4g} nm²"))
                rows.append(row("Specific surface area", f"{qm * NA * area / 1000:.6g} m²/g"))
    res.notes.append("Adsorbed amount in mol/kg (header 'q (cm3/g)' is read as STP volume).")

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        pu = display.pressure
        ax.plot([from_si(v, pu, P) for v in p], q, "ko", label="data")
        grid = np.linspace(p.min() * 0.5, p.max() * 1.05, 300)
        for name, (f, rng) in fits.items():
            g = grid[(grid >= rng.min() * 0.5) & (grid <= rng.max() * 1.05)] if name == "BET" else grid
            ax.plot([from_si(v, pu, P) for v in g], f(g), label=name)
        ax.set_xlabel(f"P ({pu})"); ax.set_ylabel("q (mol/kg)"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("Adsorption isotherm")
        fig.tight_layout()
        return fig

    res.plot, res.plot_name = plot, "adsorption"
    return res


def _log_fit(title, params, x, y, yfit, step=True):
    if not log.on():
        return
    if step:
        log.step(title)
    else:
        log.line(title)
    log.line("Fitted: {}   R² = {}", params, f"{_r2(y, yfit):.6f}")
    log.table(["P (Pa)", "q data", "q model", "residual"],
              [[f"{a:.6g}", f"{b:.6g}", f"{c:.6g}", f"{b - c:+.3g}"] for a, b, c in zip(x, y, yfit)])


def _cross_section(adsorbate, T_, given):
    if given:
        return given * 1e-18
    if adsorbate is None or T_ is None:
        return None
    try:
        return 1.091 * (adsorbate.Vliq(T_) / NA) ** (2 / 3)
    except Exception:
        return None


def szyszkowski(c, s0, gm, K, n_rt):
    return s0 - n_rt * gm * np.log(1 + K * c)


def gibbs_adsorption(path, T_, n=1):
    c, s = read_columns(path, [C, ST])
    nrt = n * R * T_
    s_max = s.max()
    f = lambda cc, s0, gm, K: szyszkowski(cc, s0, gm, K, nrt)
    (s0, gm, K), _ = curve_fit(f, c, s, p0=[s_max, 3e-6, 1 / np.median(c[c > 0])], maxfev=20000)
    res = Result("Gibbs adsorption isotherm")
    if log.on():
        log.step("Data read from {} ({} points)", path, len(c))
        log.line("Gibbs: Γ = −(1/nRT) dσ/d ln c, n = {}; nRT = {} J/mol", n, f"{nrt:.6g}")
        log.line("Szyszkowski fit σ = σ0 − nRT·Γmax·ln(1 + K·c) (nonlinear least squares), then Γ = Γmax·K·c/(1 + K·c)")
        log.line("σ0 = {}, Γmax = {} mol/m², K = {} m³/mol, R² = {}", q(s0, ST), f"{gm:.6g}", f"{K:.6g}",
                 f"{_r2(s, f(c, s0, gm, K)):.6f}")
        log.line("Area per molecule = 1/(Γmax·NA) = {} nm²", f"{1 / (gm * NA) * 1e18:.4g}")
        log.table(["c", "σ data", "σ fit", "Γ (mol/m²)"],
                  [[q(ci, C), q(si, ST), q(f(ci, s0, gm, K), ST), f"{gm * K * ci / (1 + K * ci):.4g}"] for ci, si in zip(c, s)])
    res.section("Szyszkowski fit  σ = σ0 − nRT·Γmax·ln(1 + K·c)", [
        row("σ0", s0, ST), row("Γmax", f"{gm:.6g} mol/m²"),
        row("Area per molecule at Γmax", f"{1 / (gm * NA) * 1e18:.4g} nm²"),
        row("K", f"{K * 1e3:.6g} L/mol"), row("R²", f"{_r2(s, f(c, s0, gm, K)):.5f}"),
    ])
    res.section("Surface excess at each point  Γ = Γmax·K·c/(1 + K·c)", [
        row(f"c = {fmt(ci, C)}", f"σ = {fmt(si, ST)}   Γ = {gm * K * ci / (1 + K * ci):.4g} mol/m²")
        for ci, si in zip(c, s)
    ])

    def plot():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        cu, su = display.concentration, display.surface_tension
        grid = np.linspace(0, c.max() * 1.05, 300)
        ax.plot([from_si(v, cu, C) for v in c], [from_si(v, su, ST) for v in s], "ko", label="data")
        ax.plot([from_si(v, cu, C) for v in grid], [from_si(v, su, ST) for v in f(grid, s0, gm, K)], label="fit")
        ax2 = ax.twinx()
        ax2.plot([from_si(v, cu, C) for v in grid], gm * K * grid / (1 + K * grid), "--", color="gray")
        ax2.set_ylabel("Γ (mol/m²)")
        ax.set_xlabel(f"c ({cu})"); ax.set_ylabel(f"σ ({su})"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"Gibbs adsorption at {T_:.2f} K")
        fig.tight_layout()
        return fig

    res.plot, res.plot_name = plot, "gibbs_adsorption"
    return res


# Micelles

def _cmc_breakpoint(c, s):
    lc = np.log(c)
    order = np.argsort(lc)
    lc, s = lc[order], s[order]
    best = None
    for k in range(2, len(lc) - 1):
        a = np.polyfit(lc[:k], s[:k], 1)
        b = np.polyfit(lc[k:], s[k:], 1) if len(lc) - k > 1 else (0.0, s[k:].mean())
        sse = np.sum((np.polyval(a, lc[:k]) - s[:k]) ** 2) + np.sum((np.polyval(b, lc[k:]) - s[k:]) ** 2)
        if best is None or sse < best[0]:
            best = (sse, a, b)
    _, a, b = best
    return math.exp((b[1] - a[1]) / (a[0] - b[0])), a, b


def micelles(T_, cmc=None, beta=0.0, T2=None, cmc2=None, path=None):
    from thermo_utils import substances
    water = substances.get("water")
    data = None
    if path:
        c, s = read_columns(path, [C, ST])
        cmc, a, b = _cmc_breakpoint(c[c > 0], s[c > 0])
        data = (c[c > 0], s[c > 0], a, b)
    if cmc is None:
        raise ValueError("Give the CMC or a CSV file with c and σ.")
    cw = 1 / water.Vliq(T_)
    x = cmc / (cmc + cw)
    dG = (1 + beta) * R * T_ * math.log(x)
    res = Result("Micellization")
    if log.on():
        if data:
            log.step("CMC from σ vs ln c: best split into two straight lines (least squares)")
            a, b = data[2], data[3]
            log.line("Below CMC: σ = {} ln c + {};  above: σ = {} ln c + {}", f"{a[0]:.6g}", f"{a[1]:.6g}", f"{b[0]:.6g}", f"{b[1]:.6g}")
            log.line("Intersection: ln CMC = (b1 − a1)/(a0 − b0) -> CMC = {}", q(cmc, C))
        log.step("Standard Gibbs energy of micellization (phase-separation model)")
        log.line("Water molar concentration = 1/V_liq = {} mol/m³", f"{cw:.6g}")
        log.line("x_CMC = CMC/(CMC + c_water) = {}", f"{x:.6g}")
        log.line("ΔG°mic = (1 + β)RT ln x_CMC = {} × {} × {} × ln({}) = {}", f"{1 + beta:g}", R, f"{T_:.6g} K", f"{x:.6g}", q(dG, E))
    rows = res.section(f"At {fmt(T_, T)}", [
        row("CMC", cmc, C), row("x_CMC (mole fraction)", f"{x:.6g}"),
        row("Counterion binding β", f"{beta:g}"), row("ΔG°mic", dG, E),
    ])
    if T2 is not None and cmc2 is not None:
        x2 = cmc2 / (cmc2 + 1 / water.Vliq(T2))
        dH = (1 + beta) * R * math.log(x2 / x) / (1 / T2 - 1 / T_)
        log.step("Temperature dependence (two CMC values)")
        log.line("ΔH°mic = (1 + β)R ln(x2/x1)/(1/T2 − 1/T1) = {}", q(dH, E))
        log.line("ΔS°mic = (ΔH° − ΔG°)/T = {}", q((dH - dG) / T_, S))
        rows += [row("ΔH°mic", dH, E), row("ΔS°mic", (dH - dG) / T_, S)]
    if data:
        res.notes.append("CMC from the break point of σ vs ln c (two straight lines).")

        def plot():
            import matplotlib.pyplot as plt
            c_, s_, a, b = data
            fig, ax = plt.subplots(figsize=(8, 5))
            su = display.surface_tension
            ax.semilogx(c_, [from_si(v, su, ST) for v in s_], "ko", label="data")
            for line in (a, b):
                g = np.logspace(math.log10(c_.min()), math.log10(c_.max()), 100)
                ax.semilogx(g, [from_si(v, su, ST) for v in np.polyval(line, np.log(g))], "--")
            ax.axvline(cmc, color="gray", lw=0.8, label=f"CMC = {fmt(cmc, C)}")
            ax.set_xlabel("c (mol/m³)"); ax.set_ylabel(f"σ ({su})"); ax.grid(alpha=0.3, which="both"); ax.legend()
            ax.set_title("Critical micelle concentration")
            fig.tight_layout()
            return fig

        res.plot, res.plot_name = plot, "cmc"
    return res
