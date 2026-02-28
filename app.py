import numpy as np
from flask import Flask, render_template, jsonify
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# import simulation components
from config import *
from pump import Qin
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import compute_lqr_gain, beat_synchronous_controller
from signal_process import BPProcessor

app = Flask(__name__)

def run_simulation():
    # Initialize - Drugs
    C1_phe = np.zeros(N)
    C2_phe = np.zeros(N)
    C1_nic = np.zeros(N)
    C2_nic = np.zeros(N)
    u_phe = np.zeros(N)
    u_nic = np.zeros(N)
    current_u_phe = 0.0 
    current_u_nic = 0.0
    
    # Initialize - Cardio State
    R = np.zeros(N)
    P = np.zeros(N)
    P[0] = initialize_windkessel()
    Pin = np.zeros(N)
    Qout = np.zeros(N)
    
    # Initialize - Beat Times
    beat_indices = []
    MAP_beats = []
    
    # Initialize - Signal Processing
    bp = BPProcessor(fs, HR)
    last_trough = 0
    
    # Initialize - Controller
    A, B = compute_state_space()
    K = compute_lqr_gain(A, B, Q, R_lqr)

    for k in range(N - 1):
        u_phe[k] = current_u_phe
        u_nic[k] = current_u_nic

        C1_phe[k+1], C2_phe[k+1] = update_pk_phe(C1_phe[k], C2_phe[k], u_phe[k])
        C1_nic[k+1], C2_nic[k+1] = update_pk_nic(C1_nic[k], C2_nic[k], u_nic[k])

        R[k] = compute_R(C1_phe[k], C1_nic[k])
        P[k+1], Pin[k], Qout[k] = update_windkessel(P[k], R[k], Qin[k])

        peaks, troughs = bp.detect_beats(P[:k+1])

        if len(troughs) > 0 and troughs[-1] != last_trough:
            start = last_trough
            end = troughs[-1]
            last_trough = end
            MAP_k = np.mean(P[start:end])
            MAP_beats.append(MAP_k)
            beat_indices.append(end)
            
            current_u_phe, current_u_nic = beat_synchronous_controller(
                C1_phe[start], C1_nic[start], MAP_k, K
            )

    beat_times = [float(idx * dt) for idx in beat_indices]
    
    # Downsample time-series graphs for faster frontend rendering (N is 10000)
    # 10000 points is fine for Chart.js if optimized, but let's downsample by 10 for drawing continuous arrays to avoid UI lag.
    ds = 10
    
    return {
        "time": t[::ds].tolist(),
        "map_beats": [float(m) for m in MAP_beats],
        "beat_times": beat_times,
        "target_map": float(target_map),
        "c1_phe": C1_phe[::ds].tolist(),
        "c1_nic": C1_nic[::ds].tolist(),
        "u_phe": u_phe[::ds].tolist(),
        "u_nic": u_nic[::ds].tolist()
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['GET'])
def simulate():
    data = run_simulation()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
