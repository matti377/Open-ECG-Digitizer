import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb


def plot_wfdb_record(record_base, seconds=None):
    """
    record_base: path without extension, e.g. 'output_data/wfdb/record'
    seconds: optional number of seconds to plot from the start
    """
    hea = record_base + ".hea"
    dat = record_base + ".dat"

    if not os.path.exists(hea) or not os.path.exists(dat):
        raise FileNotFoundError(f"Missing WFDB files: {hea} and/or {dat}")

    # Read WFDB record
    if seconds is None:
        record = wfdb.rdrecord(record_base)
    else:
        # Read only first N samples for faster plotting
        tmp = wfdb.rdrecord(record_base, sampto=1)  # get fs safely
        fs = int(tmp.fs)
        sampto = max(1, int(seconds * fs))
        record = wfdb.rdrecord(record_base, sampto=sampto)

    if record.p_signal is None:
        raise ValueError("Record has no p_signal (physical signal).")

    sig = record.p_signal  # shape (n_samples, n_channels)
    fs = int(record.fs)
    names = [str(s) for s in record.sig_name]

    print("WFDB loaded:")
    print(f"  record_name = {record.record_name}")
    print(f"  fs          = {fs} Hz")
    print(f"  shape       = {sig.shape}")
    print(f"  leads       = {names}")

    # Per-lead stats (great for checking conversion quality)
    print("\nLead stats:")
    for i, name in enumerate(names):
        x = sig[:, i]
        finite = np.isfinite(x)
        valid_pct = 100.0 * finite.mean()
        if finite.any():
            xmin = np.nanmin(x)
            xmax = np.nanmax(x)
            xstd = np.nanstd(x)
        else:
            xmin = xmax = xstd = np.nan
        print(f"  {name:>4}: valid={valid_pct:6.2f}%  min={xmin:8.4f}  max={xmax:8.4f}  std={xstd:8.4f}")

    # Build time axis
    t = np.arange(sig.shape[0]) / fs

    # ECG 12-lead display layout if all standard leads are present
    standard_layout = [
        ["I", "aVR", "V1"],
        ["II", "aVL", "V2"],
        ["III", "aVF", "V3"],
        ["V4", "V5", "V6"],
    ]

    name_to_idx = {n: i for i, n in enumerate(names)}

    if all(lead in name_to_idx for row in standard_layout for lead in row):
        fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True)
        max_abs = np.nanmax(np.abs(sig)) if np.isfinite(sig).any() else 1.0
        ylim = max(1.5, float(max_abs) * 1.2)

        for r in range(4):
            for c in range(3):
                lead = standard_layout[r][c]
                ax = axes[r, c]
                idx = name_to_idx[lead]
                ax.plot(t, sig[:, idx], linewidth=1)
                ax.set_title(lead, fontsize=10)
                ax.set_ylim(-ylim, ylim)
                ax.grid(True, linewidth=0.3)
                ax.tick_params(labelsize=8)

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        axes[-1, 2].set_xlabel("Time (s)")
        plt.suptitle(f"WFDB ECG Record: {record.record_name}", y=0.995)
        plt.tight_layout()
        plt.show()
    else:
        # Generic stacked plot for any channel names/count
        n_ch = sig.shape[1]
        fig, axes = plt.subplots(n_ch, 1, figsize=(14, max(4, 1.8 * n_ch)), sharex=True)
        if n_ch == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(t, sig[:, i], linewidth=1)
            ax.set_title(names[i], fontsize=9)
            ax.grid(True, linewidth=0.3)
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f"WFDB ECG Record: {record.record_name}", y=0.995)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_wfdb_record.py path/to/record [seconds]")
        print("Example: python plot_wfdb_record.py output_data/wfdb/record 10")
        sys.exit(1)

    record_base = sys.argv[1]  # no extension
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_wfdb_record(record_base, seconds)