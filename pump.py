from config import *
import numpy as np

# Blood flow shape functions
def Qsys(shape, scale, t):
    return (t**(shape-1)) * np.exp(-t/scale) # gamma
def Qdia(Qend, tau_d, sys_time, t):
    return Qend * np.exp(-(t - sys_time)/tau_d) # exp decay

# --- Compute combined area of unscaled shape profiles ---
tb_sys = np.arange(0.0, sys_time, dt)
tb_dia = np.arange(sys_time, beat_period, dt)

kernel_sys = Qsys(shape, scale, tb_sys)
Qend_unscaled = Qsys(shape, scale, sys_time)
kernel_dia = Qdia(Qend_unscaled, tau_d, sys_time, tb_dia)

area_sys = np.trapezoid(kernel_sys, tb_sys)
area_dia = np.trapezoid(kernel_dia, tb_dia)
area_total = area_sys + area_dia

# Scale factor so that inflow per beat = SV
Qscale = SV / area_total

# --- Generate full Qin over N samples ---
Qin = np.zeros(N)
for k in range(N):
    tb = t[k] % beat_period
    if tb < sys_time:
        Qin[k] = Qscale * Qsys(shape, scale, tb)
    else:
        Qin[k] = Qscale * Qdia(Qend_unscaled, tau_d, sys_time, tb)
