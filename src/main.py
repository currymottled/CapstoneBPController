import numpy as np
import matplotlib.pyplot as plt

from config import *
from pump import Qin, HR
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from control import run_controller
from signal_process import BPProcessor


# =====================================
# Allocate State Arrays
# =====================================

C1_phe = np.zeros(N)
C2_phe = np.zeros(N)

C1_nic = np.zeros(N)
C2_nic = np.zeros(N)

R = np.zeros(N)

P = np.zeros(N)
Pin = np.zeros(N)
Qout = np.zeros(N)

u_phe = np.zeros(N)
u_nic = np.zeros(N)

# Initial pressure
P[0] = initialize_windkessel()

current_u = np.array([0.0, 0.0])

# =====================================
# Signal Processor
# =====================================

fs = int(1/dt)
bp = BPProcessor(fs, HR)

last_beat_count = 0

# =====================================
# Closed-Loop Simulation
# =====================================

for k in range(N - 1):

    # Apply current infusion
    u_phe[k] = current_u[0]
    u_nic[k] = current_u[1]

    # ---- PK ----
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(
        C1_phe[k], C2_phe[k], u_phe[k]
    )

    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(
        C1_nic[k], C2_nic[k], u_nic[k]
    )

    # ---- PD ----
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # ---- Windkessel ----
    P[k+1], Pin[k], Qout[k] = update_windkessel(
        P[k], R[k], Qin[k]
    )

    # ---- Beat-based control update ----
    if k > fs:

        filtered = bp.bandpass_filter(P[:k+1])
        peaks, troughs = bp.detect_beats(P[:k+1])

        if len(troughs) > last_beat_count:

            last_beat_count = len(troughs)

            MAP_beats = bp.estimate_map(P[:k+1])

            i = len(MAP_beats) - 1
            beat_start = troughs[i]

            # Compute full control history
            u_phe_full, u_nic_full = run_controller(
                C1_phe,
                C1_nic,
                MAP_beats,
                troughs
            )

            current_u[0] = u_phe_full[k]
            current_u[1] = u_nic_full[k]


print("Simulation complete.")