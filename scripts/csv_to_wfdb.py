import sys
import os
import numpy as np
import pandas as pd
import wfdb


def extract_blocks(signal):
    valid = np.isfinite(signal)
    if valid.sum() == 0:
        return []

    diff = np.diff(valid.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if valid[0]:
        starts = np.insert(starts, 0, 0)
    if valid[-1]:
        ends = np.append(ends, len(signal))

    return list(zip(starts, ends))


def reconstruct_sequential(signal, min_len=80):
    blocks = extract_blocks(signal)
    if not blocks:
        return None

    segments = []
    for start, end in sorted(blocks, key=lambda x: x[0]):
        seg = signal[start:end]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= min_len:
            segments.append(seg)

    if not segments:
        return None

    return np.concatenate(segments)


def csv_to_wfdb(csv_file, wfdb_name="record", fs=500, output_dir="output_data/wfdb"):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # Reconstruct each lead from valid segments (DO NOT fill NaNs with zero)
    reconstructed = {}
    for col in df.columns:
        sig = df[col].to_numpy(dtype=np.float64)
        rec = reconstruct_sequential(sig, min_len=max(40, int(0.08 * fs)))
        if rec is not None and len(rec) > 0:
            reconstructed[col] = rec

    if not reconstructed:
        raise ValueError("No valid ECG segments found in CSV.")

    # Align all leads to the same length for WFDB (truncate to shortest)
    min_len = min(len(x) for x in reconstructed.values())
    if min_len < fs:  # less than 1 second
        raise ValueError(f"Reconstructed leads too short (min_len={min_len} samples).")

    lead_names = list(reconstructed.keys())
    p_signal = np.column_stack([reconstructed[name][:min_len] for name in lead_names]).astype(np.float64)

    # Per-lead normalization to avoid one lead dominating all others
    for i in range(p_signal.shape[1]):
        max_val = np.nanmax(np.abs(p_signal[:, i]))
        if np.isfinite(max_val) and max_val > 0:
            p_signal[:, i] = p_signal[:, i] / max_val

    # Optional global scale to ~mV-ish range (screening only)
    p_signal *= 1.0  # +/- ~1 mV normalized range

    wfdb.wrsamp(
        record_name=wfdb_name,
        fs=fs,
        units=["mV"] * p_signal.shape[1],
        sig_name=lead_names,
        p_signal=p_signal,
        fmt=["16"] * p_signal.shape[1],
        write_dir=output_dir
    )

    print(f"Saved WFDB record: {output_dir}/{wfdb_name}.dat / {output_dir}/{wfdb_name}.hea")
    print(f"Leads: {lead_names}")
    print(f"Samples: {p_signal.shape[0]}  Channels: {p_signal.shape[1]}  fs={fs} Hz")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_wfdb.py path/to/file.csv [fs]")
        sys.exit(1)

    csv_file = sys.argv[1]
    fs = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    csv_to_wfdb(csv_file, fs=fs)