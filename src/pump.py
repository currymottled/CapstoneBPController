from config import *
import numpy as np

# Blood flow shape function (systolic ejection)
def Qsys(shape, scale, t):
    return (t**(shape-1)) * np.exp(-t/scale)  # gamma-like profile

# --- Build systolic flow kernel with dicrotic notch ---
tb_sys = np.arange(0.0, sys_time, dt)
kernel_sys = Qsys(shape, scale, tb_sys)

# Add a brief backflow dip near end of systole (aortic valve closure → dicrotic notch)
notch_center = sys_time * 0.92         # notch occurs near end of systole
notch_width  = sys_time * 0.04         # narrow dip
notch_depth  = -0.35 * np.max(kernel_sys)
notch = notch_depth * np.exp(-0.5 * ((tb_sys - notch_center) / notch_width)**2)

kernel_sys = kernel_sys + notch

# --- Compute area of unscaled systolic profile ---
area_sys = np.trapz(kernel_sys, tb_sys)

# Scale factor so that inflow per beat = SV
Qscale = SV / area_sys

# --- Generate full Qin over N samples ---
Qin = np.zeros(N)
for k in range(N):
    tb = t[k] % beat_period
    if tb < sys_time:
        idx = int(tb / dt)
        if idx < len(kernel_sys):
            Qin[k] = Qscale * kernel_sys[idx]
    else:
        Qin[k] = 0.0   # diastole: no forward flow
