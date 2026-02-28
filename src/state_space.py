import numpy as np
from config import *


def compute_state_space(C1_phe=0.0, C1_nic=0.0, R_op=R0):

    eps = 1e-8

    # ---- PK dynamics ----
    a11 = 1 - beat_period*(k10_phe + k12_phe)
    a22 = 1 - beat_period*(k10_nic + k12_nic)

    # ---- Linearized PD sensitivity ----
    dR_dC1 = (Emax_Rphe * EC50_Rphe) / (EC50_Rphe + C1_phe + eps)**2
    dR_dC2 = (Emax_Rnic * EC50_Rnic) / (EC50_Rnic + C1_nic + eps)**2

    a31 = (beat_period * Qmean * dR_dC1) / (R_op * C)
    a32 = (beat_period * Qmean * dR_dC2) / (R_op * C)

    # ---- Windkessel decay ----
    a33 = 1 - beat_period/(R_op * C)

    A = np.array([
        [a11, 0,   0],
        [0,   a22, 0],
        [a31, a32, a33]
    ])

    # ---- Input matrix ----
    b11 = beat_period / V1_phe
    b22 = beat_period / V1_nic

    B = np.array([
        [b11, 0],
        [0, b22],
        [0, 0]
    ])

    return A, B