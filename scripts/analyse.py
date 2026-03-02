# option_a_ecg_report.py
# One-click ECG "report" (screening, not diagnosis)
# Input: WFDB record (.hea + .dat) like output_data/wfdb/record
# Output: analysis_output/summary.json + plots (12-lead grid, analyzed lead with R-peaks, ST bar chart)

import os
import sys
import json
from pathlib import Path

import numpy as np
import wfdb
import neurokit2 as nk
import matplotlib.pyplot as plt


# ---- Settings (simple + robust for short clips) ----
PREFERRED_LEADS = ["II", "I", "V2", "V5", "V6", "III", "V1", "V3", "V4", "aVL", "aVF", "aVR"]
MIN_SECONDS_FOR_INTERVALS = 4.0  # short clips may not support stable interval analysis

# ST "screen" windows relative to R peak (rough, not clinical J-point)
BASELINE_WIN_MS = (-200, -120)   # baseline before R
ST_WIN_MS = (60, 100)            # ST window after R

# mm conversion: only meaningful if your signal is truly in mV.
# Standard ECG paper: 10 mm per 1 mV.
MM_PER_MV = 10.0


def ms_to_samples(ms: float, fs: int) -> int:
    return int(round(ms * fs / 1000.0))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-8:
        return x - med
    return (x - med) / (sd + 1e-8)


def hr_from_rpeaks(rpeaks: np.ndarray, fs: int) -> float:
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return np.nan
    rr = np.diff(rpeaks) / fs
    rr = rr[(rr > 0.2) & (rr < 2.5)]  # 24–300 bpm sanity
    if len(rr) == 0:
        return np.nan
    return float(np.median(60.0 / rr))


def rhythm_flags_from_rpeaks(rpeaks: np.ndarray, fs: int):
    """
    Very simple screening flags.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 3:
        return ["Too few beats to assess rhythm regularity"]

    rr = np.diff(rpeaks) / fs
    rr_ms = rr * 1000.0
    rr_std = float(np.std(rr_ms)) if len(rr_ms) >= 2 else np.nan

    flags = []
    if np.isfinite(rr_std):
        if rr_std < 80:
            flags.append("Rhythm appears relatively regular")
        elif rr_std > 120:
            flags.append("Irregular rhythm pattern (screening)")
    return flags


def tachy_brady_flag(hr_bpm: float):
    if not np.isfinite(hr_bpm):
        return "HR unavailable"
    if hr_bpm < 60:
        return "Bradycardia range (<60 bpm)"
    if hr_bpm > 100:
        return "Tachycardia range (>100 bpm)"
    return "Normal range (60 - 100 bpm)"


def estimate_st_deviation(clean: np.ndarray, rpeaks: np.ndarray, fs: int) -> float:
    """
    ST deviation estimate per lead:
    median(ST window) - median(baseline window), aggregated over beats.
    This is a rough screening measurement.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return np.nan

    b0 = ms_to_samples(BASELINE_WIN_MS[0], fs)
    b1 = ms_to_samples(BASELINE_WIN_MS[1], fs)
    s0 = ms_to_samples(ST_WIN_MS[0], fs)
    s1 = ms_to_samples(ST_WIN_MS[1], fs)

    st_vals = []
    n = len(clean)

    for r in rpeaks:
        b_start, b_end = r + b0, r + b1
        s_start, s_end = r + s0, r + s1

        if b_start < 0 or s_end >= n:
            continue
        if b_end <= b_start or s_end <= s_start:
            continue

        baseline = float(np.median(clean[b_start:b_end]))
        st = float(np.median(clean[s_start:s_end]))
        st_vals.append(st - baseline)

    if len(st_vals) == 0:
        return np.nan
    return float(np.median(st_vals))


def lead_order(sig_names):
    upper = [s.upper() for s in sig_names]
    order = []
    for p in PREFERRED_LEADS:
        pu = p.upper()
        for i, s in enumerate(upper):
            if s == pu and i not in order:
                order.append(i)
    for i in range(len(sig_names)):
        if i not in order:
            order.append(i)
    return order


def process_lead(sig_1d: np.ndarray, fs: int):
    """
    Robust for short ECGs:
    - filter + z-score
    - detect peaks
    Returns: dict with cleaned, rpeaks, hr, quality score
    """
    x = np.asarray(sig_1d, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(int(1.5 * fs), 200):
        return None

    # light filter (helps digitized traces)
    try:
        xf = nk.signal_filter(x, sampling_rate=fs, lowcut=0.5, highcut=40, method="butterworth", order=3)
    except Exception:
        xf = x

    xz = zscore(xf)

    # try both polarities
    best = None
    for polarity, candidate in [("as_is", xz), ("inverted", -xz)]:
        try:
            clean = nk.ecg_clean(candidate, sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
            rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)

            hr = hr_from_rpeaks(rpeaks, fs)
            score = (len(rpeaks) * 10.0) + (np.nanstd(clean) * 2.0)

            if len(rpeaks) < 2:
                continue

            res = {
                "clean": clean,
                "rpeaks": rpeaks,
                "hr_bpm": hr,
                "polarity": polarity,
                "score": float(score),
            }
            if best is None or res["score"] > best["score"]:
                best = res
        except Exception:
            continue

    return best


def save_12lead_grid(t, sig, sig_names, out_path: Path):
    """
    12-lead grid plot if standard leads exist, otherwise stacked plot.
    """
    name_to_idx = {n: i for i, n in enumerate(sig_names)}

    layout = [
        ["I", "aVR", "V1"],
        ["II", "aVL", "V2"],
        ["III", "aVF", "V3"],
        ["V4", "V5", "V6"],
    ]
    has_layout = all(lead in name_to_idx for row in layout for lead in row)

    if has_layout:
        fig, axes = plt.subplots(4, 3, figsize=(14, 9), sharex=True)
        max_abs = float(np.nanmax(np.abs(sig))) if np.isfinite(sig).any() else 1.0
        ylim = max(1.5, max_abs * 1.2)

        for r in range(4):
            for c in range(3):
                lead = layout[r][c]
                ax = axes[r, c]
                idx = name_to_idx[lead]
                ax.plot(t, sig[:, idx], linewidth=1)
                ax.set_title(lead, fontsize=10)
                ax.set_ylim(-ylim, ylim)
                ax.grid(True, linewidth=0.3)
                ax.tick_params(labelsize=8)

        plt.suptitle("12-lead overview (raw)", y=0.995)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        n_ch = sig.shape[1]
        fig, axes = plt.subplots(n_ch, 1, figsize=(14, max(4, 1.6 * n_ch)), sharex=True)
        if n_ch == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(t, sig[:, i], linewidth=1)
            ax.set_title(sig_names[i], fontsize=9)
            ax.grid(True, linewidth=0.3)
        plt.suptitle("ECG overview (raw)", y=0.995)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def main():
    record_base = sys.argv[1] if len(sys.argv) > 1 else "output_data/wfdb/record"

    if not os.path.exists(record_base + ".hea") or not os.path.exists(record_base + ".dat"):
        raise FileNotFoundError(f"Missing WFDB pair: {record_base}.hea / {record_base}.dat")

    rec = wfdb.rdrecord(record_base)
    fs = int(rec.fs)
    sig = rec.p_signal
    sig_names = list(rec.sig_name)

    if sig is None:
        raise ValueError("No p_signal in record (check conversion).")

    n_samples, n_leads = sig.shape
    duration_s = n_samples / fs

    out_dir = Path("output_data/analysis_output")
    out_dir.mkdir(exist_ok=True)

    # Save 12-lead raw grid plot
    t = np.arange(n_samples) / fs
    save_12lead_grid(t, sig, sig_names, out_dir / "12lead_grid.png")

    # Process multiple leads and compute per-lead HR + ST screen
    results = []
    for idx in lead_order(sig_names):
        lead_name = sig_names[idx]
        res = process_lead(sig[:, idx], fs)
        if res is None:
            continue

        st_dev_units = estimate_st_deviation(res["clean"], res["rpeaks"], fs)  # units of your signal (often normalized)
        st_mm_equiv = st_dev_units * MM_PER_MV  # only "real" if units are mV

        results.append({
            "lead_index": idx,
            "lead_name": lead_name,
            "polarity": res["polarity"],
            "score": res["score"],
            "rpeak_count": int(len(res["rpeaks"])),
            "hr_bpm": res["hr_bpm"],
            "st_dev_units": st_dev_units,
            "st_mm_equiv": st_mm_equiv,
            "clean": res["clean"],
            "rpeaks": res["rpeaks"],
        })

    if not results:
        summary = {
            "record_name": rec.record_name,
            "sampling_rate_hz": fs,
            "duration_seconds": float(duration_s),
            "warnings": ["No usable leads found."],
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    # Choose best lead for the "main" plot
    best = sorted(results, key=lambda x: x["score"], reverse=True)[0]

    # HR consensus across leads
    hr_vals = [r["hr_bpm"] for r in results if np.isfinite(r["hr_bpm"])]
    hr_consensus = float(np.median(hr_vals)) if hr_vals else np.nan

    # Rhythm flags from best lead (simple)
    rhythm_flags = rhythm_flags_from_rpeaks(best["rpeaks"], fs)

    # ST per lead bar plot (mm-equivalent)
    lead_names = [r["lead_name"] for r in results]
    st_vals = [r["st_mm_equiv"] for r in results]

    plt.figure(figsize=(10, 4))
    plt.bar(lead_names, st_vals)
    plt.axhline(0, linewidth=1)
    plt.title("ST deviation per lead (mm-equivalent)")
    plt.xlabel("Lead")
    plt.ylabel("ST deviation (mm-equivalent)")
    plt.tight_layout()
    plt.savefig(out_dir / "st_per_lead.png", dpi=150)
    plt.close()

    # Save analyzed lead plot with R-peaks
    plt.figure(figsize=(14, 4))
    x = best["clean"]
    plt.plot(x, linewidth=1)
    rp = best["rpeaks"]
    rp = rp[(rp >= 0) & (rp < len(x))]
    if len(rp) > 0:
        plt.scatter(rp, x[rp], s=15)
    plt.title(f"Analyzed lead: {best['lead_name']} (cleaned) with R-peaks")
    plt.xlabel("Samples")
    plt.ylabel("Normalized amplitude")
    plt.tight_layout()
    plt.savefig(out_dir / "analyzed_lead.png", dpi=150)
    plt.close()

    # Build summary
    warnings_list = []
    if duration_s < MIN_SECONDS_FOR_INTERVALS:
        warnings_list.append("Short ECG clip: HR is ok, ST is a rough screen, no diagnosis.")

    # ST “max abs” screen
    st_abs = [abs(v) for v in st_vals if np.isfinite(v)]
    max_abs_st = float(np.max(st_abs)) if st_abs else np.nan

    summary = {
        "record_name": rec.record_name,
        "sampling_rate_hz": fs,
        "n_samples": int(n_samples),
        "n_leads": int(n_leads),
        "lead_names": sig_names,
        "duration_seconds": float(duration_s),
        "heart_rate_bpm_median_across_leads": hr_consensus,
        "tachy_brady_flag": tachy_brady_flag(hr_consensus),
        "rhythm_flags_best_lead": rhythm_flags,
        "best_lead": {
            "lead_index": int(best["lead_index"]),
            "lead_name": best["lead_name"],
            "polarity": best["polarity"],
            "rpeak_count": int(best["rpeak_count"]),
            "hr_bpm": best["hr_bpm"],
            "st_mm_equiv": best["st_mm_equiv"],
        },
        "st_mm_equiv_max_abs_across_leads": max_abs_st,
        "notes": [
            "This is a screening demo for education.",
            "ST in mm-equivalent is only meaningful if your signal units are true mV (10 mm = 1 mV).",
            "Image digitization often changes amplitude, so ST results may be off without calibration."
        ],
        "warnings": warnings_list,
        "per_lead": [
            {
                "lead_name": r["lead_name"],
                "hr_bpm": r["hr_bpm"],
                "rpeak_count": r["rpeak_count"],
                "st_mm_equiv": r["st_mm_equiv"],
                "polarity": r["polarity"],
                "score": r["score"],
            }
            for r in results
        ],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== OPTION A ECG REPORT (SCREENING, NOT DIAGNOSIS) ===")
    print(json.dumps(summary, indent=2))

    print("\nSaved files:")
    print(f"- {out_dir / 'summary.json'}")
    print(f"- {out_dir / '12lead_grid.png'}")
    print(f"- {out_dir / 'analyzed_lead.png'}")
    print(f"- {out_dir / 'st_per_lead.png'}")


if __name__ == "__main__":
    main()