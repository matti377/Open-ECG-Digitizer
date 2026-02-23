import sys
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

def load_lead(record_name, lead="II"):
    rec = wfdb.rdrecord(record_name)
    if lead not in rec.sig_name:
        raise ValueError(f"Lead {lead} not found (available: {rec.sig_name})")
    idx = rec.sig_name.index(lead)
    return rec.p_signal[:, idx].astype(np.float64)

def bandpass_filter(signal, fs, lowcut=5.0, highcut=15.0):
    nyq = 0.5 * fs
    b, a = butter(2, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks_pan_tompkins(signal, fs):
    # 1) bandpass
    filtered = bandpass_filter(signal, fs)

    # 2) derivative
    diff = np.diff(filtered)
    squared = diff**2

    # 3) moving window integration
    window = int(0.150 * fs)  # 150 ms window
    integrated = np.convolve(squared, np.ones(window)/window, mode='same')

    # 4) peak detection with adaptive threshold
    threshold = np.mean(integrated) * 1.2
    peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.3*fs))

    return peaks

def analyze(record_name, lead="II", fs=500):
    print(f"Loading lead {lead} from record {record_name}...")
    signal = load_lead(record_name, lead)

    print("Running Pan–Tompkins peak detection...")
    r_peaks = detect_r_peaks_pan_tompkins(signal, fs)

    if len(r_peaks) < 2:
        print("Not enough peaks detected.")
        return

    rr_intervals = np.diff(r_peaks) / fs * 1000
    heart_rates = 60000 / rr_intervals

    print("\n=== Rhythm Analysis (Pan–Tompkins) ===")
    print(f"Total beats detected: {len(r_peaks)}")
    print(f"Mean Heart Rate: {np.mean(heart_rates):.1f} BPM")
    print(f"Min Heart Rate: {np.min(heart_rates):.1f} BPM")
    print(f"Max Heart Rate: {np.max(heart_rates):.1f} BPM")
    print(f"Mean RR: {np.mean(rr_intervals):.1f} ms")
    print(f"RR STD: {np.std(rr_intervals):.1f} ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ecg_pt_rhythm.py record_name [lead] [fs]")
        sys.exit(1)

    rec = sys.argv[1]
    lead = sys.argv[2] if len(sys.argv) > 2 else "II"
    fs = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    analyze(rec, lead, fs)
