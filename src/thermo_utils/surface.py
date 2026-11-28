"""Surface phenomena: surface tension and adsorption isotherms."""

import csv
import re

import numpy as np
from scipy.optimize import curve_fit

from thermo_utils import steplog as log
from thermo_utils.result import Result, row
from thermo_utils.steplog import K, q
from thermo_utils.units import mark_given, num
from thermo_utils.units import (
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
    return f"{num(value * 1e3)} mN/(m·K)"


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
            mark_given(cols[i][-1])
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
            log.line("  Tr = {}, Tbr = {}  ->  σ = {}", num(T_ / sub.Tc), num(sub.Tb / sub.Tc),
                     q(Brock_Bird(T_, sub.Tb, sub.Tc, sub.Pc), ST))
        except Exception:
            log.line("Brock-Bird: not available")
        par = (sub._constants.Parachors or [None])[0]
        if par:
            try:
                dens = 1 / sub.Vliq(T_) - _rho_vap(sub, T_) / (sub.MW / 1000)
                log.line("Macleod-Sugden: σ^¼ = [P](ρL − ρV); [P] = {}, ρL − ρV = {} mol/m³  ->  σ = {}", num(par),
                         num(dens), q((par * dens) ** 4, ST))
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
        log.table(["P (Pa)", "q (mol/kg)"], [[num(a), num(b)] for a, b in zip(p, q)])
        log.line("Each model fitted by nonlinear least squares (scipy curve_fit); R² = 1 − SS_res/SS_tot")
    fits = {}
    if model in ("langmuir", "all"):
        (qm, K), _ = curve_fit(langmuir, p, q, p0=[q.max() * 1.2, 1 / np.median(p)], maxfev=20000)
        fits["Langmuir"] = (lambda x, a=qm, b=K: langmuir(x, a, b), p)
        _log_fit("Langmuir: q = qm·K·P/(1 + K·P)", f"qm = {num(qm)} mol/kg, K = {num(K)} 1/Pa", p, q, langmuir(p, qm, K))
        res.section("Langmuir  q = qm·K·P/(1 + K·P)", [
            row("qm", f"{num(qm)} mol/kg"), row("K", f"{num(K)} 1/Pa"),
            row("R²", num(_r2(q, langmuir(p, qm, K)))),
        ])
    if model in ("freundlich", "all"):
        (Kf, n), _ = curve_fit(freundlich, p, q, p0=[q.mean() / np.mean(p) ** 0.5, 2], maxfev=20000)
        fits["Freundlich"] = (lambda x, a=Kf, b=n: freundlich(x, a, b), p)
        _log_fit("Freundlich: q = K·P^(1/n)", f"K = {num(Kf)}, n = {num(n)}", p, q, freundlich(p, Kf, n))
        res.section("Freundlich  q = K·P^(1/n)", [
            row("K", f"{num(Kf)} mol/kg/Pa^(1/n)"), row("n", num(n)),
            row("R²", num(_r2(q, freundlich(p, Kf, n)))),
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
            _log_fit("BET: q = qm·c·x/[(1 − x)(1 − x + c·x)]", f"qm = {num(qm)} mol/kg, c = {num(c)}", p[mask], q[mask],
                     bet(x[mask], qm, c), step=False)
            rows = res.section("BET  q = qm·c·x/[(1 − x)(1 − x + c·x)],  x = P/P0", [
                row("P0", P0, P), row("qm (monolayer)", f"{num(qm)} mol/kg"), row("c", num(c)),
                row("Points used (0.05 < x < 0.35)", str(int(mask.sum()))),
                row("R²", num(_r2(q[mask], bet(x[mask], qm, c)))),
            ])
            area = _cross_section(adsorbate, T_, cross_section)
            if area:
                log.line("Area per molecule = {} nm²{}", num(area * 1e18),
                         "" if cross_section else " (1.091·(V_liq/NA)^(2/3))")
                log.line("Specific area = qm·NA·area = {} mol/kg × {} × {} m² = {} m²/g", num(qm), num(NA),
                         num(area), num(qm * NA * area / 1000))
                rows.append(row("Area per adsorbed molecule", f"{num(area * 1e18)} nm²"))
                rows.append(row("Specific surface area", f"{num(qm * NA * area / 1000)} m²/g"))
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
    log.line("Fitted: {}   R² = {}", params, num(_r2(y, yfit)))
    log.table(["P (Pa)", "q data", "q model", "residual"],
              [[num(a), num(b), num(c), f"{b - c:+.3g}"] for a, b, c in zip(x, y, yfit)])


def _cross_section(adsorbate, T_, given):
    if given:
        return given * 1e-18
    if adsorbate is None or T_ is None:
        return None
    try:
        return 1.091 * (adsorbate.Vliq(T_) / NA) ** (2 / 3)
    except Exception:
        return None
