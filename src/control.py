import numpy as np
from scipy.linalg import solve_discrete_are
from config import *


def compute_lqr_gain(A, B, Q, R): # A & B computed via compute_state_space, Q & R from config

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    return K

# Beat synchronous control function

def beat_synchronous_controller(C1_phe_k, C1_nic_k, MAP_k, K):
    """
    Beat-synchronous LQR controller.
    Runs ONCE per beat and returns infusion commands for that beat.

    Inputs:
        C1_phe_k : phe concentration at the start of the beat
        C1_nic_k : nic concentration at the start of the beat
        MAP_k    : MAP for the completed beat
        K        : LQR gain matrix

    Returns:
        u_phe_k, u_nic_k : infusion commands for the NEXT beat
    """

    # Build state vector
    x_k = np.array([
        C1_phe_k,
        C1_nic_k,
        MAP_k
    ])

    # Reference state
    x_ref = np.array([
        0.0,
        0.0,
        target_map
    ])

    # LQR control law
    u_k = -K @ (x_k - x_ref)

    # Extract drug commands (must be positive)
    u_phe_k = max(0.0, u_k[0])
    u_nic_k = max(0.0, u_k[1])

    return u_phe_k, u_nic_k
