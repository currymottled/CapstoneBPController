import numpy as np
from config import *
from pump import HR, Qin
from signal_process import Pin, MAP_est, Za_est, R_est, C_est

# Build state vector at each time step: [R, C, Za, Qin[k], Pin[k]]
state_vector = np.column_stack((np.full(N, R),
                                np.full(N, C),
                                np.full(N, Za),
                                Qin,
                                Pin))

# Quadratic cost: squared deviation from target MAP
cost_function = (MAP_est[-1] - target_map)**2