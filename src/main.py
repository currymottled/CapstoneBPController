import numpy as np
import matplotlib.pyplot as plt

from config import *
from pump import Qin
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import beat_synchronous_controller
from signal_process import BPProcessor

# -----------------------
# Initialize simulation arrays
# -----------------------

# Drugs
C1_phe = np.zeros(N)
C2_phe = np.zeros(N)
C1_nic = np.zeros(N)
C2_nic = np.zeros(N)
u_phe = np.zeros(N)
u_nic = np.zeros(N)
current_u_phe = 0.0
current_u_nic = 0.0

# Cardio state
R = np.zeros(N)
P = np.zeros(N)
P[0] = initialize_windkessel(SV=SV, HR=HR, Pv=Pv, R0=R0)
Pin = np.zeros(N)
Qout = np.zeros(N)

# Beat tracking
beat_indices = []
MAP_beats = []
last_trough = 0

# Signal processing
bp = BPProcessor(fs, HR)

# Warm-up (avoid transients)

for k in range(warmup_steps - 1):
    # PK update (baseline, no infusion)
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(C1_phe[k], C2_phe[k], current_u_phe)
    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(C1_nic[k], C2_nic[k], current_u_nic)

    # PD update
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # Windkessel update
    P[k+1], Pin[k], Qout[k] = update_windkessel(P[k], R[k], Qin[k])

# -----------------------
# Main simulation loop
# -----------------------
for k in range(warmup_steps - 1, N - 1):
    # 1. Apply current infusion command
    u_phe[k] = current_u_phe
    u_nic[k] = current_u_nic

    # 2. PK update
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(C1_phe[k], C2_phe[k], u_phe[k])
    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(C1_nic[k], C2_nic[k], u_nic[k])

    # 3. PD update
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # 4. Windkessel update
    P[k+1], Pin[k], Qout[k] = update_windkessel(P[k], R[k], Qin[k])

    # 5. Beat detection
    peaks, troughs = bp.detect_beats(P[:k+1])

    if len(troughs) > 0 and troughs[-1] != last_trough:
        start = last_trough
        end = troughs[-1]
        last_trough = end

        # 6. Compute MAP for this beat
        MAP_k = np.mean(P[start:end])
        MAP_beats.append(MAP_k)
        beat_indices.append(end)

        # 7. Compute linearized state-space for this beat
        A, B = compute_state_space(C1_phe[start], C1_nic[start], R[start])

        # 8. Run constrained LQR controller (QP)
        current_u_phe, current_u_nic = beat_synchronous_controller(
            C1_phe[start],
            C1_nic[start],
            MAP_k,
            A,
            B,
            Q=Q,
            R_lqr=R_lqr
        )

# -----------------------
# End simulation
# -----------------------
print("Closed-loop simulation complete.")


# Pressure Waveform
plt.figure(figsize=(12, 4))
plt.plot(t, P, label="Arterial Pressure")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.title("Pressure Waveform")
plt.grid(True)
plt.legend()
plt.show()

# MAP response
plt.figure(figsize=(10, 5))
beat_times = [beat_index * dt for beat_index in beat_indices]
plt.plot(beat_times, MAP_beats, 'o-', label="Estimated MAP")
plt.axhline(target_map, linestyle='--', color='black', label="Target MAP")
plt.xlabel("Time (s)")
plt.ylabel("Mean Arterial Pressure (mmHg)")
plt.title("Beat-Averaged MAP")
plt.legend()
plt.grid(True)
plt.show()

# 1st compartment concentrations
plt.figure(figsize=(10,5))
plt.plot(t, C1_phe, label="C1_phe (Phenylephrine)")
plt.plot(t, C1_nic, label="C1_nic (Nicardipine)")
plt.xlabel("Time (s)")
plt.ylabel("Concentration (ug/L)")
plt.title("Effect-Site Concentrations")
plt.legend()
plt.grid(True)
plt.show()

# infusion
plt.figure(figsize=(10,5))
plt.plot(t, u_phe, label="u_phe (Phenylephrine infusion)")
plt.plot(t, u_nic, label="u_nic (Nicardipine infusion)")
plt.xlabel("Time (s)")
plt.ylabel("Infusion Rate (ug/s)")
plt.title("Drug Infusion Commands")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(t[:-1], R[:-1], linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("mmHg*s/L")
plt.title("Peripheral Resistance R Over Time")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t, P, label="Arterial Pressure")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.title("Pressure Waveform")
plt.grid(True)
plt.legend()
plt.show()

# Send to database: 
# P(pressure)
# MAP_beats, beat_times (beat averaged MAP and beat time indices) 
# C1_phe, C1_nic (drug concentrations)