"""Governing equations shown with each calculation."""

EOS = {
    "ideal": [
        "PV = RT",
    ],
    "virial": [
        "Z = PV/RT = 1 + BP/RT",
        "B·Pc/(R·Tc) = B⁰ + ω·B¹",
        "B⁰ = 0.083 − 0.422/Tr^1.6     B¹ = 0.139 − 0.172/Tr^4.2",
    ],
    "vdw": [
        "P = RT/(V − b) − a/V²",
        "a = 27R²Tc²/(64Pc)     b = RTc/(8Pc)",
    ],
    "rk": [
        "P = RT/(V − b) − a/[T^½·V(V + b)]",
        "a = 0.42748·R²Tc^2.5/Pc     b = 0.08664·RTc/Pc",
    ],
    "srk": [
        "P = RT/(V − b) − a·α(T)/[V(V + b)]",
        "α = [1 + m(1 − √Tr)]²     m = 0.480 + 1.574ω − 0.176ω²",
        "a = 0.42748·R²Tc²/Pc     b = 0.08664·RTc/Pc",
    ],
    "pr": [
        "P = RT/(V − b) − a·α(T)/[V(V + b) + b(V − b)]",
        "α = [1 + κ(1 − √Tr)]²     κ = 0.37464 + 1.54226ω − 0.26992ω²",
        "a = 0.45724·R²Tc²/Pc     b = 0.07780·RTc/Pc",
    ],
    "pr-fit": [
        "P = RT/(V' − b) − a·α(T)/[V'(V' + b) + b(V' − b)]     V = V' − c",
        "α = Tr^(N(M−1))·exp[L(1 − Tr^(MN))]   (Twu)",
        "L, M, N fitted to the databank Psat; c fitted to the liquid density",
    ],
    "coolprop": [
        "a(T, ρ)/RT = α⁰(τ, δ) + αʳ(τ, δ)     τ = Tc/T, δ = ρ/ρc",
        "P = ρRT[1 + δ(∂αʳ/∂δ)τ]",
        "H, S, Cp... from derivatives of α",
    ],
}

PROPERTIES = [
    "H = H_ig + H_R     S = S_ig + S_R",
    "H_ig = ∫Cp_ig dT     S_ig = ∫Cp_ig/T dT − R ln(P/P°)",
    "U = H − PV     G = H − TS     A = U − TS",
]

BY_CALC = {
    "pvt": ["Z = PV/RT"],
    "state": PROPERTIES + ["ln φ = G_R/RT     f = φP"],
    "delta": [
        "ΔH = H_R2 − H_R1 + ∫Cp_ig dT",
        "ΔS = S_R2 − S_R1 + ∫Cp_ig/T dT − R ln(P2/P1)",
        "ΔU = ΔH − Δ(PV)     ΔG = ΔH − Δ(TS)     ΔA = ΔU − Δ(TS)",
    ],
    "residual": [
        "M_R = M − M_ig   (same T and P)",
        "G_R/RT = ln φ = ∫(Z − 1) dP/P",
        "H_R/RT = −T∫(∂Z/∂T)_P dP/P",
        "S_R/R = H_R/RT − G_R/RT",
    ],
    "saturation": [
        "G_liq = G_vap   (φ_liq = φ_vap)",
        "dPsat/dT = ΔHvap/(T·ΔVvap)   (Clapeyron)",
        "ΔHn/(RTn) = 1.092(ln Pc − 1.013)/(0.930 − Trn)   (Riedel, Pc in bar)",
        "ΔH2/ΔH1 = [(1 − Tr2)/(1 − Tr1)]^0.38   (Watson)",
    ],
    "sensible": [
        "Q = ΔH = ∫Cp dT     ΔS = ∫Cp/T dT",
        "<Cp>H = ΔH/(T2 − T1)     <Cp>S = ΔS/ln(T2/T1)",
    ],
    "latent": [
        "ΔHn/(RTn) = 1.092(ln Pc − 1.013)/(0.930 − Trn)   (Riedel, Pc in bar)",
        "ΔH2/ΔH1 = [(1 − Tr2)/(1 − Tr1)]^0.38   (Watson)",
        "ΔSvap = ΔHvap/Tsat",
    ],
    "reaction-heat": [
        "ΔH°298 = Σ νi·ΔHf°i     ΔG°298 = Σ νi·ΔGf°i",
        "ΔH°(T) = ΔH°298 + ∫ΔCp° dT     ΔCp° = Σ νi·Cp°i   (Kirchhoff)",
    ],
    "combustion": [
        "HHV = −ΔH°c   (H2O liquid in the products)",
        "LHV = HHV − n_H2O·ΔHvap(H2O, 298 K)",
    ],
    "flame": [
        "ΔH = Σ n_reac∫Cp dT (T_in → 298) + ΔH°298 + Σ n_prod∫Cp dT (298 → Tf) = 0",
        "Air: 21 % O2, 79 % N2",
    ],
    "phase-rule": ["F = 2 − π + N − r − s"],
    "device": [
        "Steady flow, adiabatic, ΔEk = ΔEp = 0:   Ws = ΔH",
        "Isentropic outlet: S2' = S1     Ws,isen = H2' − H1",
        "Compressor, pump: η = Ws,isen/Ws     Turbine: η = Ws/Ws,isen",
        "Valve: H2 = H1     Power = ṅ·Ws",
    ],
    "fugacity-mix": [
        "f̂i = φ̂i·yi·P     Lewis-Randall: f̂i = yi·fi",
        "Virial: ln φ̂i = (P/RT)[Bii + ½ΣjΣk yj·yk(2δji − δjk)]     δij = 2Bij − Bii − Bjj",
        "Cubic: a = ΣiΣj yi·yj·√(ai·aj)(1 − kij)     b = Σ yi·bi",
    ],
    "activity": [
        "f̂i = xi·γi·fi     G^E/RT = Σ xi ln γi",
        "ln γi = [∂(nG^E/RT)/∂ni]T,P,nj",
        "H^E = −RT²[∂(G^E/RT)/∂T]P,x     S^E = (H^E − G^E)/T",
        "ΔG_mix = G^E + RT Σ xi ln xi",
    ],
    "vle": [
        "γ-φ: yi·Φi·P = xi·γi·Pi_sat     Φi = φ̂i/φi_sat",
        "Modified Raoult: yi·P = xi·γi·Pi_sat",
        "φ-φ: yi·φ̂i_vap = xi·φ̂i_liq",
        "Flash: Σ zi(Ki − 1)/[1 + V(Ki − 1)] = 0     Ki = yi/xi",
    ],
    "diagram": [
        "yi·Φi·P = xi·γi·Pi_sat",
        "Azeotrope: x1 = y1   (α12 = 1)",
    ],
    "fit": [
        "γi = yi·P/(xi·Pi_sat)     G^E/RT = x1 ln γ1 + x2 ln γ2",
        "Margules: G^E/(x1x2RT) = A21·x1 + A12·x2",
        "van Laar: x1x2RT/G^E = x1/A'21 + x2/A'12",
    ],
    "henry": ["yi·P = xi·Hi"],
    "lle": ["xi_α·γi_α = xi_β·γi_β   (each species)"],
    "stability": ["Stable if d²(ΔG_mix/RT)/dx1² > 0 for every x1"],
    "sle": [
        "ln(xi·γi) = −(ΔHfus,i/R)(1/T − 1/Tm,i)   (immiscible solids)",
        "Eutectic: both solubility curves meet",
    ],
    "solubility": ["ln(x·γ) = −(ΔHfus/R)(1/T − 1/Tm)"],
    "osmotic": [
        "Π = −(RT/V_solvent)·ln(x_solvent·γ_solvent)",
        "Dilute: Π ≈ x_solute·RT/V_solvent",
    ],
    "k": [
        "ΔG° = −RT ln K",
        "d ln K/dT = ΔH°/RT²   (van 't Hoff)",
        "ΔH°(T) = ΔH°298 + ∫ΔCp° dT",
    ],
    "equilibrium": [
        "Gas: Π (yi·φ̂i·P/P°)^νi = K     Liquid: Π (xi·γi)^νi = K",
        "ni = ni0 + νi·ε     yi = ni/Σ ni",
    ],
    "gibbs": [
        "min G = Σ ni[G°i + RT ln(yi·P/P°)]",
        "Element balances: Σ ni·aik = Ak",
    ],
}


def for_calc(calc_id, model=None):
    lines = list(BY_CALC.get(calc_id, []))
    if model in EOS:
        lines += EOS[model]
    return lines
