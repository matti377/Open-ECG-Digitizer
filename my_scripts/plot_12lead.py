import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_ecg(csv_path, fs=500):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    lead_names = list(df.columns)
    signals = df.apply(pd.to_numeric, errors="coerce").values

    n_samples = signals.shape[0]
    time = np.arange(n_samples) / fs

    layout = [
        ["I", "aVR", "V1"],
        ["II", "aVL", "V2"],
        ["III", "aVF", "V3"],
        ["V4", "V5", "V6"],
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)

    valid_values = signals[~np.isnan(signals)]
    ylim = max(1.5, np.percentile(np.abs(valid_values), 99) * 1.2) if len(valid_values) else 2

    for r in range(4):
        for c in range(3):
            lead = layout[r][c]
            ax = axes[r, c]
            ax.set_title(lead, fontsize=9)

            if lead in lead_names:
                idx = lead_names.index(lead)
                signal = signals[:, idx]

                if np.isnan(signal).all():
                    ax.text(0.5, 0.5, "NO DATA",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=10, color="red")
                else:
                    # Plot real data
                    valid_mask = ~np.isnan(signal)
                    ax.plot(time[valid_mask],
                            signal[valid_mask],
                            color="blue",
                            linewidth=1)

                    # Interpolated full signal (for gap reconstruction)
                    interp_signal = pd.Series(signal).interpolate().values

                    # Detect contiguous NaN segments
                    nan_mask = np.isnan(signal)

                    # Find start and end indices of NaN blocks
                    diff = np.diff(nan_mask.astype(int))
                    gap_starts = np.where(diff == 1)[0] + 1
                    gap_ends = np.where(diff == -1)[0] + 1

                    # Edge cases: start/end with NaN
                    if nan_mask[0]:
                        gap_starts = np.insert(gap_starts, 0, 0)
                    if nan_mask[-1]:
                        gap_ends = np.append(gap_ends, len(signal))

                    # Plot only missing segments
                    for start, end in zip(gap_starts, gap_ends):
                        ax.plot(time[start:end],
                                interp_signal[start:end],
                                color="red",
                                linewidth=1)

            ax.set_ylim(-ylim, ylim)
            ax.set_xticks(np.arange(0, time.max(), 0.2))
            ax.set_yticks(np.arange(-ylim, ylim, 0.5))
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

