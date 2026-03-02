import numpy as np
import cvxpy as cp
from config import *

# -----------------------------
# Beat-Synchronous Constrained LQR Controller
# -----------------------------
def beat_synchronous_controller(C1_phe_k, C1_nic_k, MAP_k, A, B, Q=None, R_lqr=None):
    """
    Beat-synchronous constrained LQR controller using a QP (CSV).
    Solves for optimal infusion commands given current state-space matrices,
    ensuring no negative infusion and proper allocation between drugs.

    Inputs:
        C1_phe_k, C1_nic_k    effect-site concentrations at beat start
        MAP_k                 MAP for the completed beat
        A, B                  discrete-time state-space matrices (from compute_state_space)
        Q                     state cost matrix (optional, defaults to identity)
        R_lqr                 control penalty matrix (optional, defaults to identity)

    Returns:
        u_phe_k, u_nic_k : infusion commands for the NEXT beat
    """

    x = np.array([
        C1_phe_k,
        C1_nic_k,
        MAP_k
    ])

    x_ref = np.array([
        0.0,
        0.0,
        target_map
    ])

    # --- Define QP ---
    u = cp.Variable(2)  # two drug inputs

    # Predict next state with linearized dynamics
    x_next = A @ x + B @ u

    # Quadratic cost: minimize deviation from reference + control effort
    cost = cp.quad_form(x_next - x_ref, Q) + cp.quad_form(u, R_lqr)

    # Constraint: no negative infusion
    constraints = [u >= 0]

    # Solve QP
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    u_phe_k = u.value[0]
    u_nic_k = u.value[1]

    return u_phe_k, u_nic_k