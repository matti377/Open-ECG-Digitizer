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

    # Determine amplitude from real values only
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
                    print(f"WARNING: Lead {lead} contains no data.")
                    ax.text(0.5, 0.5, "NO DATA",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=10, color="red")
                else:
                    # Plot real data only
                    valid_mask = ~np.isnan(signal)
                    ax.plot(time[valid_mask], signal[valid_mask],
                            linewidth=1, color="blue")

                    # Interpolate for gap visualization
                    interp_signal = pd.Series(signal).interpolate().values

                    # Identify gap segments
                    nan_mask = np.isnan(signal)
                    gap_indices = np.where(nan_mask)[0]

                    if len(gap_indices) > 0:
                        print(f"Lead {lead}: {len(gap_indices)} missing samples filled in red.")

                        ax.plot(time[nan_mask],
                                interp_signal[nan_mask],
                                linewidth=1,
                                color="red")

            else:
                print(f"WARNING: Lead {lead} not found in CSV.")
                ax.text(0.5, 0.5, "MISSING",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=10, color="red")

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

