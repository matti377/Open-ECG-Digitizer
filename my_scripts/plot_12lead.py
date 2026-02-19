import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def extract_blocks(signal):
    valid = ~np.isnan(signal)
    diff = np.diff(valid.astype(int))

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if valid[0]:
        starts = np.insert(starts, 0, 0)
    if valid[-1]:
        ends = np.append(ends, len(signal))

    return list(zip(starts, ends))


def reconstruct_sequential(signal):
    """
    Reconstruct full lead by concatenating all valid segments
    in time order.
    """
    blocks = extract_blocks(signal)
    if not blocks:
        return None

    blocks = sorted(blocks, key=lambda x: x[0])

    segments = []
    for start, end in blocks:
        segment = signal[start:end]
        if len(segment) > 100:  # ignore tiny noise fragments
            segments.append(segment)

    if not segments:
        return None

    return np.concatenate(segments)


def plot_ecg(csv_path, fs=500):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path, dtype=np.float32)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Downsample for speed
    target_fs = 200
    if fs > target_fs:
        step = fs // target_fs
        df = df.iloc[::step]
        fs = target_fs

    layout = [
        ["I", "aVR", "V1"],
        ["II", "aVL", "V2"],
        ["III", "aVF", "V3"],
        ["V4", "V5", "V6"],
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 8))

    max_val = np.nanmax(np.abs(df.values))
    ylim = max(1.5, max_val * 1.2)

    for r in range(4):
        for c in range(3):
            lead = layout[r][c]
            ax = axes[r, c]
            ax.set_title(lead, fontsize=9)

            if lead not in df.columns:
                ax.text(0.5, 0.5, "NO DATA",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="red")
                continue

            signal = df[lead].values

            reconstructed = reconstruct_sequential(signal)

            if reconstructed is None:
                ax.text(0.5, 0.5, "NO VALID SEGMENT",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="orange")
                continue

            t = np.arange(len(reconstructed)) / fs
            ax.plot(t, reconstructed, linewidth=1)

            ax.set_ylim(-ylim, ylim)
            ax.grid(True, linewidth=0.3)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_12lead.py path/to/file.csv [sampling_rate]")
        sys.exit(1)

    csv_file = sys.argv[1]
    sampling_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    plot_ecg(csv_file, sampling_rate)
