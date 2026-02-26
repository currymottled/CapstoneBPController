import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


class BPProcessor:

    """
    Blood pressure processing.
    Bandpass -> Beat Detect -> Artifact Rejection -> MAP Estimate
    """
    def __init__(self, sampling_rate, heart_rate):

        self.fs = sampling_rate
        self.hr = heart_rate

        self.beat_interval_sec = 1.0 / self.hr
        self.samples_per_beat = int(self.fs * self.beat_interval_sec)

        self._filter_coeffs = None
        self._last_signal_id = None

        self.filtered = None
        self.peaks = None
        self.troughs = None
        self.SBP = None
        self.DBP = None
        self.MAP = None


    # -------------------------
    # Compute filter coefficients once
    # -------------------------
    def _get_filter(self, order=4):

        if self._filter_coeffs is None:

            nyq = 0.5 * self.fs
            lowcut = (self.hr / 3.0) / nyq
            highcut = (self.hr * 5.0) / nyq

            b, a = butter(order, [lowcut, highcut], btype='band')
            self._filter_coeffs = (b, a)

        return self._filter_coeffs


    # -------------------------
    # Bandpass Filtering
    # -------------------------
    def bandpass_filter(self, signal, order=4):

        # Only refilter if signal object changed
        if id(signal) == self._last_signal_id and self.filtered is not None:
            return self.filtered

        b, a = self._get_filter(order)

        self.filtered = filtfilt(b, a, signal)
        self._last_signal_id = id(signal)

        return self.filtered


    # -------------------------
    # Beat Detection
    # -------------------------
    def detect_beats(self, signal):

        if self.filtered is None:
            signal_to_use = signal
        else:
            signal_to_use = self.filtered

        min_distance = int(0.5 * self.samples_per_beat)

        self.peaks, _ = find_peaks(signal_to_use, distance=min_distance)
        self.troughs, _ = find_peaks(-signal_to_use, distance=min_distance)

        self.SBP = signal_to_use[self.peaks]
        self.DBP = signal_to_use[self.troughs]

        return self.peaks, self.troughs


    # -------------------------
    # MAP Estimation
    # -------------------------
    def estimate_map(self, signal):

        if self.troughs is None:
            raise RuntimeError("Run detect_beats() first.")

        if self.filtered is None:
            signal_to_use = signal
        else:
            signal_to_use = self.filtered

        MAP = []

        for i in range(len(self.troughs) - 1):
            start = self.troughs[i]
            end = self.troughs[i+1]
            MAP.append(np.mean(signal_to_use[start:end]))

        self.MAP = np.array(MAP)
        return self.MAP