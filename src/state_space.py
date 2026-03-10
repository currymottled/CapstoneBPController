import numpy as np
from config import *


def compute_state_space(C1_phe=0.0, C1_nic=0.0, R_op=R0):

    eps = 1e-8  # small number to avoid divide-by-zero

    # ---- PK dynamics ----
    a11 = 1 - beat_period*(k10_phe + k12_phe)
    a22 = 1 - beat_period*(k10_nic + k12_nic)

    # PK dynamics
    # Central compartment discrete decay
    a11 = 1 - dt * (k10_phe + k12_phe)
    a22 = 1 - dt * (k10_nic + k12_nic)

    a31 = (beat_period * Qmean * dR_dC1) / (R_op * C)
    a32 = (beat_period * Qmean * dR_dC2) / (R_op * C)

    # ---- Windkessel decay ----
    a33 = 1 - beat_period/(R_op * C)

    A = np.array([
        [a11, 0.0, 0.0, 0.0],   # C1_phe
        [0.0, a22, 0.0, 0.0],   # C1_nic
        [a31, a32, a33, 0.0],   # MAP
        [0.0, 0.0, a43, a44]    # Integral state
    ])

    # ---- Input matrix ----
    b11 = beat_period / V1_phe
    b22 = beat_period / V1_nic

    # Indirect effect on MAP via resistance
    b31 = dt * Qmean * dR_dC1 / V1_phe
    b32 = dt * Qmean * dR_dC2 / V1_nic

    # Integrator has no direct input
    B = np.array([
        [b11, 0.0],   # phe infusion → C1_phe
        [0.0, b22],   # nic infusion → C1_nic
        [b31, b32],   # infusion → MAP
        [0.0, 0.0]    # no direct effect on integral
    ])

    return A, B