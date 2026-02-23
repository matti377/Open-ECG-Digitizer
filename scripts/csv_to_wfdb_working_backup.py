import sys
import os
import numpy as np
import pandas as pd
import wfdb


def csv_to_wfdb(csv_file, wfdb_name="record", fs=500, output_dir="output_data/wfdb"):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Replace NaN and Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    p_signal = df.values.astype(np.float64)

    # Normalize to realistic ECG amplitude (±2 mV)
    max_val = np.max(np.abs(p_signal))
    if max_val > 0:
        p_signal = p_signal / max_val * 2.0

    n_signals = p_signal.shape[1]

    wfdb.wrsamp(
        record_name=wfdb_name,
        fs=fs,
        units=["mV"] * n_signals,
        sig_name=df.columns.tolist(),
        p_signal=p_signal,
        fmt=["16"] * n_signals,
        write_dir=output_dir
    )

    print(f"Saved WFDB record: {output_dir}/{wfdb_name}.dat / {output_dir}/{wfdb_name}.hea · fs={fs} Hz")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_wfdb.py path/to/file.csv")
        sys.exit(1)

    csv_to_wfdb(sys.argv[1])