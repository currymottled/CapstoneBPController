import numpy as np
from config import *


# Initial Concentrations

def initialize_pk():
    """
    Returns initial concentrations.
    """
    C1_phe_0 = 0.0
    C2_phe_0 = 0.0
    C1_nic_0 = 0.0
    C2_nic_0 = 0.0

    return C1_phe_0, C2_phe_0, C1_nic_0, C2_nic_0


# Update Functions

def update_pk_phe(C1_k, C2_k, u_k):
    """
    One-step 2-compartment update for phenylephrine.
    """

    dC1 = (u_k / V1_phe) \
        - k10_phe * C1_k \
        - k12_phe * C1_k \
        + k21_phe * C2_k

    dC2 = k12_phe * C1_k - k21_phe * C2_k

    C1_k1 = C1_k + dt * dC1
    C2_k1 = C2_k + dt * dC2

    return C1_k1, C2_k1


def update_pk_nic(C1_k, C2_k, u_k):
    """
    One-step 2-compartment update for nicardipine.
    """

    dC1 = (u_k / V1_nic) \
        - k10_nic * C1_k \
        - k12_nic * C1_k \
        + k21_nic * C2_k

    dC2 = k12_nic * C1_k - k21_nic * C2_k

    C1_k1 = C1_k + dt * dC1
    C2_k1 = C2_k + dt * dC2

    return C1_k1, C2_k1