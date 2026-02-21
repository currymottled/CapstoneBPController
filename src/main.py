import numpy as np
import matplotlib.pyplot as plt

from config import *
from pump import Qin, HR
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import compute_lqr_gain, run_controller
from signal_process import BPProcessor


# =====================================
# Compute Linear Model + LQR Gain
# =====================================

A, B = compute_state_space()   # linearize around baseline
K = compute_lqr_gain(A, B, Q, R_lqr) # Q & R state & control penalty matrices in config


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

P[0] = initialize_windkessel()


# =====================================
# 3️⃣ Signal Processor
# =====================================

fs = int(1/dt)
bp = BPProcessor(fs, HR)

last_beat_count = 0


# =====================================
# 4️⃣ Closed-Loop Simulation
# =====================================

for k in range(N - 1):

    # ---- PK ----
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(
        C1_phe[k],
        C2_phe[k],
        u_phe[k]
    )

    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(
        C1_nic[k],
        C2_nic[k],
        u_nic[k]
    )

    # ---- PD ----
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # ---- Windkessel ----
    P[k+1], Pin[k], Qout[k] = update_windkessel(
        P[k], R[k], Qin[k]
    )

    # ---- Beat-based controller update ----
    if k > fs:

        filtered = bp.bandpass_filter(P[:k+1])
        peaks, troughs = bp.detect_beats(P[:k+1])

        if len(troughs) > last_beat_count:

            last_beat_count = len(troughs)

            MAP_beats = bp.estimate_map(P[:k+1])

            # Compute infusion for full history
            u_phe, u_nic = run_controller(
                C1_phe,
                C1_nic,
                MAP_beats,
                troughs,
                K
            )


print("Simulation complete.")


# =====================================
# 5️⃣ Optional Plot
# =====================================

plt.figure()
plt.plot(t, P, label="MAP")
plt.axhline(target_map, linestyle='--', label="Target")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.legend()
plt.show()