import numpy as np
from config import *


def compute_state_space(C1_phe=0.0, C1_nic=0.0, R_op=R0):

    eps = 1e-8  # small number to avoid divide-by-zero

    # ---- PK dynamics ----
    # Discrete-time central compartment decay:
    # C1_phe[k+1] ≈ C1_phe[k] - dt*(k10_phe + k12_phe)*C1_phe[k]
    # So the coefficient multiplying C1_phe in the linearized A matrix is:
    a11 = 1 - dt*(k10_phe + k12_phe)

    # C1_nic[k+1] ≈ C1_nic[k] - dt*(k10_nic + k12_nic)*C1_nic[k]
    a22 = 1 - dt*(k10_nic + k12_nic)

    # ---- Linearized PD sensitivity ----
    # Resistance R changes as a function of drug concentrations
    # Linearize R around current concentrations to get sensitivity:
    # dR/dC1 ≈ ∂R/∂C1 = (Emax_Rphe * EC50_Rphe) / (EC50_Rphe + C1_phe)^2
    dR_dC1 = (Emax_Rphe * EC50_Rphe) / (EC50_Rphe + C1_phe + eps)**2

    # dR/dC2 ≈ ∂R/∂C2 = (Emax_Rnic * EC50_Rnic) / (EC50_Rnic + C1_nic)^2
    dR_dC2 = (Emax_Rnic * EC50_Rnic) / (EC50_Rnic + C1_nic + eps)**2

    # ---- MAP dynamics (Windkessel linearization) ----
    # a31 = ∂MAP/∂C1_phe ≈ dt * Qmean * dR_dC1
    # This captures how a change in phe concentration changes R, which in turn changes MAP
    a31 = dt * Qmean * dR_dC1

    # a32 = ∂MAP/∂C1_nic ≈ dt * Qmean * dR_dC2
    # Similarly, change in nic concentration affects MAP through resistance
    a32 = dt * Qmean * dR_dC2

    # a33 = discrete-time decay of MAP due to Windkessel dynamics
    # From dP/dt = -(P - Pv)/(R_op * C), discretized: P[k+1] = a33*P[k] + ...
    a33 = 1 - dt/(R_op * C)

    # ---- Assemble A matrix ----
    # Rows: [C1_phe, C1_nic, MAP]
    # Columns: [C1_phe, C1_nic, MAP]
    # Diagonal: self-decay / retention of each state
    # Off-diagonal: cross-coupling effects (concentration → MAP)
    A = np.array([
        [a11, 0,   0],   # phe compartment only affected by its own decay
        [0,   a22, 0],   # nic compartment only affected by its own decay
        [a31, a32, a33]  # MAP affected by both drug concentrations and its own decay
    ])
        # ---- Input matrix ----
    # Drug infusion directly affects central compartment concentrations:
    # ΔC1_phe ≈ dt * u_phe / V1_phe
    # ΔC1_nic ≈ dt * u_nic / V1_nic
    b11 = dt / V1_phe
    b22 = dt / V1_nic

    # MAP is affected indirectly via resistance changes:
    # ΔMAP ≈ dt * Qmean * (dR/dC1 * ΔC1_phe + dR/dC2 * ΔC1_nic)
    # Here, dR/dC1 and dR/dC2 are the linearized PD sensitivities
    b31 = dt * Qmean * dR_dC1 / V1_phe  # infusion → central concentration → resistance → MAP
    b32 = dt * Qmean * dR_dC2 / V1_nic  # infusion → central concentration → resistance → MAP

    B = np.array([
        [b11, 0],   # phe infusion → phe central compartment
        [0,   b22], # nic infusion → nic central compartment
        [b31, b32]  # infusion → concentration → resistance → MAP
    ])
    return A, B