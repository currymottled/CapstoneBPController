from numpy import kaiser
from numpy.linalg import lstsq
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
import numpy as np

class BPProcessor:

    def __init__(self, pressure, fs):
        self.raw = np.asarray(pressure)
        self.fs = fs

        self.filtered = None
        self.peaks = None
        self.troughs = None

        self.SBP = []
        self.DBP = []
        self.MAP = []
        self.PP  = []
        self.tau = []
        
def __init__(self, pressure_waveform, sampling_rate, heart_rate):
        self.raw = np.asarray(pressure_waveform)
        self.fs = sampling_rate
        self.hr = heart_rate

        # Beat interval in seconds and samples
        self.beat_interval_sec = 60.0 / self.hr
        self.samples_per_beat = int(self.fs * self.beat_interval_sec)

        # Storage
        self.filtered = None
        self.peaks = None
        self.troughs = None
        self.SBP = None
        self.DBP = None
        self.MAP = None

        def preprocess(self):
            nyq = 0.5 * self.fs
            low = 0.5 / nyq
            high = 15.0 / nyq

            b, a = butter(4, [low, high], btype="band")
            self.filtered = filtfilt(b, a, self.raw)

            return self.filtered

def detect_beats(self, min_hr=40, max_hr=180):
      
    signal = self.filtered if self.filtered is not None else self.raw

    min_dist = int(self.fs * 60 / max_hr)

    self.peaks, _ = find_peaks(signal, distance=min_dist)
    self.troughs, _ = find_peaks(-signal, distance=min_dist)

    return self.peaks, self.troughs

def is_valid_beat(self, beat):
    sbp = np.max(beat)
    dbp = np.min(beat)
    pp  = sbp - dbp

    if not (60 <= sbp <= 250):
        return False
    if not (30 <= dbp <= 150):
        return False
    if not (10 <= pp <= 150):
        return False

    return True

def estimate_map(self):
    if self.troughs is None:
        raise RuntimeError("Run detect_beats() first.")
    signal = self.filtered if self.filtered is not None else self.raw
    MAP = []
    for i in range(len(self.troughs) - 1):
        start, end = self.troughs[i], self.troughs[i+1]
        beat_segment = signal[start:end]
        MAP.append(np.mean(beat_segment))
    self.MAP = np.array(MAP)
    return self.MAP



def estimate_co_over_c(self, method='area'):

# CO/C = cardiac output / arterial compliance 
        if self.tau is None:
            self.estimate_tau()
        
        if self.SBP is None:
            self.extract_features()
        
        signal = self.filtered if self.filtered is not None else self.raw
        co_c_values = []
        
        for idx, i in enumerate(self.valid_indices):
            if i >= len(self.peaks) - 1:
                continue
            
            if np.isnan(self.tau[idx]):
                co_c_values.append(np.nan)
                continue
            
            peak_idx = self.peaks[i]
            trough_idx = self.troughs[i]
            next_trough_idx = self.troughs[i + 1]
            
            if method == 'area':
                # Estimate stroke volume from beat area
                beat_segment = signal[trough_idx:next_trough_idx]
                beat_area = np.trapz(beat_segment, dx=1/self.fs)
                
                # CO/C ≈ Beat Area / tau
                co_c = beat_area / self.tau[idx]
                
            elif method == 'derivative':
                # CO/C = dP/dt|peak + P_peak/tau
                # Estimate derivative at peak
                window = 5
                start = max(0, peak_idx - window)
                end = min(len(signal), peak_idx + window)
                
                t_local = np.arange(end - start) / self.fs
                p_local = signal[start:end]
                
                # Numerical derivative
                dp_dt = np.gradient(p_local, t_local)
                peak_derivative = dp_dt[window] if len(dp_dt) > window else dp_dt[len(dp_dt)//2]
                
                co_c = peak_derivative + signal[peak_idx] / self.tau[idx]
            
            else:
                raise ValueError("Method must be 'area' or 'derivative'")
            
            # Sanity check
            if 0 < co_c < 1000:  # Reasonable physiological range
                co_c_values.append(co_c)
            else:
                co_c_values.append(np.nan)
        
        self.CO_over_C = np.array(co_c_values)
        return self.CO_over_C
    
def tau_est(self, t, P):
    
    Pinf = np.min(P)
    P_shifted = P - Pinf + 1  # Avoid log(0)
    
    # Remove any non-positive values
    valid = P_shifted > 0
    if np.sum(valid) < 5:
        raise ValueError("Insufficient valid points")
    
    t_valid = t[valid]
    log_P = np.log(P_shifted[valid])
    
    # Linear fit: log_P = a - t/tau
    coeffs = np.polyfit(t_valid, log_P, 1)
    slope = coeffs[0]
    
    tau = -1.0 / slope
    return tau

def estimate_za(self, flow_signal=None):
    if self.peaks is None or self.troughs is None:
        raise RuntimeError("Run detect_beats() first.")
    
    if self.SBP is None:
        self.extract_features()
    
    signal = self.filtered if self.filtered is not None else self.raw
    za_values = []
    
    for idx, i in enumerate(self.valid_indices):
        if i >= len(self.peaks) - 1:
            continue
        
        peak_idx = self.peaks[i]
        trough_idx = self.troughs[i]
        next_trough_idx = self.troughs[i + 1]
        
        if flow_signal is not None:
            # Direct calculation: Za = dP/dQ in early systole
            systolic_start = trough_idx
            systolic_end = peak_idx
            
            P_sys = signal[systolic_start:systolic_end]
            Q_sys = flow_signal[systolic_start:systolic_end]
            
            # Linear regression dP vs dQ
            if len(P_sys) > 5 and len(Q_sys) > 5:
                dP = np.gradient(P_sys, 1/self.fs)
                dQ = np.gradient(Q_sys, 1/self.fs)
                
                valid = (dQ != 0) & (np.abs(dP) < 1000) & (np.abs(dQ) < 1000)
                if np.sum(valid) > 3:
                    za = np.median(dP[valid] / dQ[valid])
                else:
                    za = np.nan
            else:
                za = np.nan
        else:
            # Approximation without flow signal
            # Za ≈ (PP × ejection_time) / stroke_volume
            # SV estimated from beat area
            beat_segment = signal[trough_idx:next_trough_idx]
            beat_area = np.trapz(beat_segment, dx=1/self.fs)
            
            ejection_time = (peak_idx - trough_idx) / self.fs
            pp = self.PP[idx]
            
            if beat_area > 0 and ejection_time > 0:
                za = (pp * ejection_time) / beat_area
            else:
                za = np.nan
        
        # Sanity check: Za typically 0.03-0.15 mmHg·s/mL
        if 0.01 < za < 0.5:
            za_values.append(za)
        else:
            za_values.append(np.nan)
    self.Za = np.array(za_values)
    return self.Za

def get_summary_statistics(self): 
    stats = {}
    parameters = {
        'SBP': self.SBP,
        'DBP': self.DBP,
        'MAP': self.MAP,
        'PP': self.PP,
        'tau': self.tau,
        'CO_over_C': self.CO_over_C,
        'Za': self.Za
    }
    
    for name, values in parameters.items():
        if values is not None:
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                stats[name] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'median': np.median(valid_values),
                    'min': np.min(valid_values),
                    'max': np.max(valid_values),
                    'n_valid': len(valid_values)
                }
    return stats
