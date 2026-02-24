import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


# ----------------------------
# IO / utilities
# ----------------------------

def load_wfdb_record(record_base_path: str, verbose: bool = False):
    """
    record_base_path = path WITHOUT extension
    Example: 'output_data/wfdb/record'
    """
    record = wfdb.rdrecord(record_base_path)
    fs = int(record.fs)
    sig = record.p_signal  # shape: (n_samples, n_channels)
    sig_names = list(record.sig_name)

    if verbose:
        print("WFDB loaded:")
        print("  fs =", fs)
        print("  shape =", sig.shape)
        for i, name in enumerate(sig_names):
            x = sig[:, i]
            finite = np.isfinite(x)
            if finite.any():
                print(
                    f"  {name:>4}: "
                    f"min={np.nanmin(x):.4f}, max={np.nanmax(x):.4f}, std={np.nanstd(x):.4f}, "
                    f"valid={100.0 * finite.mean():.1f}%"
                )
            else:
                print(f"  {name:>4}: all NaN")

    return record, fs, sig, sig_names


def extract_blocks(signal: np.ndarray) -> List[Tuple[int, int]]:
    """Return contiguous finite-value blocks [(start, end), ...]."""
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


def reconstruct_sequential(signal: np.ndarray, min_len: int = 80) -> Optional[np.ndarray]:
    """
    Concatenate valid segments in time order.
    Useful for digitized ECGs that contain NaN gaps.
    """
    blocks = extract_blocks(signal)
    if not blocks:
        return None

    segments = []
    for start, end in sorted(blocks, key=lambda x: x[0]):
        seg = np.asarray(signal[start:end], dtype=float)
        seg = seg[np.isfinite(seg)]
        if len(seg) >= min_len:
            segments.append(seg)

    if not segments:
        return None

    return np.concatenate(segments)


def median_or_nan(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


# ----------------------------
# ECG processing helpers
# ----------------------------

def preprocess_lead_for_ecg(lead: np.ndarray, fs: int) -> Optional[np.ndarray]:
    """
    Basic preprocessing for digitized ECG:
    - finite only
    - filter
    - z-normalize
    """
    lead = np.asarray(lead, dtype=float)
    lead = lead[np.isfinite(lead)]

    if len(lead) < max(int(1.5 * fs), 200):
        return None

    if np.nanstd(lead) < 1e-6:
        return None

    try:
        lead_f = nk.signal_filter(
            lead,
            sampling_rate=fs,
            lowcut=0.5,
            highcut=40,
            method="butterworth",
            order=3,
        )
    except Exception:
        lead_f = lead

    std = np.nanstd(lead_f)
    if not np.isfinite(std) or std < 1e-8:
        return None

    lead_z = (lead_f - np.nanmedian(lead_f)) / (std + 1e-8)
    return lead_z


def safe_ecg_process_or_peaks(lead_signal: np.ndarray, fs: int):
    """
    Robust ECG processing:
    - Short signals: ecg_clean + ecg_peaks only
    - Longer signals: ecg_process full pipeline
    Returns: signals_df, info, mode
    """
    lead_signal = np.asarray(lead_signal, dtype=float)
    lead_signal = lead_signal[np.isfinite(lead_signal)]

    # Short-signal fallback: avoid segmentation-heavy pipeline
    if len(lead_signal) < max(4 * fs, 1500):
        try:
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs, method="neurokit")
            _, peaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")

            rpeaks = peaks_info.get("ECG_R_Peaks", [])
            peak_col = np.zeros(len(cleaned), dtype=int)
            if rpeaks is not None:
                rpeaks_arr = np.asarray(rpeaks, dtype=int)
                rpeaks_arr = rpeaks_arr[(rpeaks_arr >= 0) & (rpeaks_arr < len(peak_col))]
                peak_col[rpeaks_arr] = 1

            signals_df = pd.DataFrame(
                {
                    "ECG_Raw": lead_signal,
                    "ECG_Clean": cleaned,
                    "ECG_R_Peaks": peak_col,
                }
            )
            return signals_df, peaks_info, "short_fallback"
        except Exception:
            return None, None, "failed"

    # Normal path
    try:
        signals_df, info = nk.ecg_process(lead_signal, sampling_rate=fs, method="neurokit")
        return signals_df, info, "full"
    except Exception:
        # fallback even for long signals
        try:
            cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs, method="neurokit")
            _, peaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
            rpeaks = peaks_info.get("ECG_R_Peaks", [])
            peak_col = np.zeros(len(cleaned), dtype=int)
            if rpeaks is not None:
                rpeaks_arr = np.asarray(rpeaks, dtype=int)
                rpeaks_arr = rpeaks_arr[(rpeaks_arr >= 0) & (rpeaks_arr < len(peak_col))]
                peak_col[rpeaks_arr] = 1
            signals_df = pd.DataFrame(
                {"ECG_Raw": lead_signal, "ECG_Clean": cleaned, "ECG_R_Peaks": peak_col}
            )
            return signals_df, peaks_info, "fallback_after_full_fail"
        except Exception:
            return None, None, "failed"


def compute_rr_metrics(rpeaks, fs: int) -> Dict[str, float]:
    """
    Basic RR/HR metrics.
    Allows as few as 2 peaks (1 RR) for short clips.
    """
    rpeaks = np.asarray(rpeaks, dtype=int)
    if len(rpeaks) < 2:
        return {}

    rr = np.diff(rpeaks) / fs  # seconds
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if len(rr) == 0:
        return {}

    hr = 60.0 / rr
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if len(rr) >= 2 else np.nan
    pnn50 = (
        (np.sum(np.abs(np.diff(rr)) > 0.05) / max(1, len(rr) - 1)) * 100
        if len(rr) >= 2 else np.nan
    )

    return {
        "rr_count": int(len(rr)),
        "hr_mean_bpm": float(np.mean(hr)),
        "hr_median_bpm": float(np.median(hr)),
        "hr_min_bpm": float(np.min(hr)),
        "hr_max_bpm": float(np.max(hr)),
        "rr_std_ms": float(np.std(rr) * 1000.0) if len(rr) >= 2 else np.nan,
        "rmssd_ms": float(rmssd * 1000.0) if np.isfinite(rmssd) else np.nan,
        "pnn50_pct": float(pnn50) if np.isfinite(pnn50) else np.nan,
    }


def classify_rhythm_basic(rr_metrics: Dict[str, float]) -> List[str]:
    """
    Very rough rhythm heuristics (screening only, not diagnosis).
    """
    if not rr_metrics:
        return ["Not enough R-peaks to estimate rhythm"]

    flags = []
    hr = rr_metrics.get("hr_median_bpm", np.nan)
    rr_std = rr_metrics.get("rr_std_ms", np.nan)
    pnn50 = rr_metrics.get("pnn50_pct", np.nan)

    if np.isfinite(hr):
        if hr < 50:
            flags.append("Bradycardia range (HR < 50 bpm)")
        elif hr > 100:
            flags.append("Tachycardia range (HR > 100 bpm)")

    if np.isfinite(rr_std) and np.isfinite(pnn50):
        if rr_std > 120 and pnn50 > 20:
            flags.append("Irregular rhythm pattern (screening flag)")
        elif rr_std < 80:
            flags.append("Rhythm appears relatively regular")

    return flags


def estimate_intervals_from_delineation(cleaned, rpeaks, fs: int) -> List[Dict]:
    """
    NeuroKit delineation-based beat features.
    Often unavailable on short/noisy signals -> returns [] safely.
    """
    beat_features = []
    try:
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    except Exception:
        return beat_features

    q_peaks = np.array(waves.get("ECG_Q_Peaks", []), dtype=float)
    s_peaks = np.array(waves.get("ECG_S_Peaks", []), dtype=float)
    t_offsets = np.array(waves.get("ECG_T_Offsets", []), dtype=float)
    t_peaks = np.array(waves.get("ECG_T_Peaks", []), dtype=float)
    rpeaks_arr = np.array(rpeaks, dtype=float)

    n = min(
        len(rpeaks_arr),
        len(q_peaks) if len(q_peaks) else len(rpeaks_arr),
        len(s_peaks) if len(s_peaks) else len(rpeaks_arr),
    )
    if n == 0:
        return beat_features

    def get_idx(a, i):
        if i < len(a) and np.isfinite(a[i]):
            return int(a[i])
        return None

    for i in range(n):
        r = get_idx(rpeaks_arr, i)
        q = get_idx(q_peaks, i)
        s = get_idx(s_peaks, i)
        t_off = get_idx(t_offsets, i)
        t_pk = get_idx(t_peaks, i)

        feat = {"r_index": r, "qrs_ms": np.nan, "qt_ms": np.nan, "st_dev": np.nan, "t_r_gap_ms": np.nan}

        if q is not None and s is not None and s > q:
            feat["qrs_ms"] = (s - q) / fs * 1000.0

        if q is not None and t_off is not None and t_off > q:
            feat["qt_ms"] = (t_off - q) / fs * 1000.0

        # rough ST deviation estimate (relative units if normalized)
        if s is not None:
            st_idx = s + int(0.08 * fs)
            if 0 <= st_idx < len(cleaned):
                st_val = float(cleaned[st_idx])
                if q is not None:
                    base_idx = q - int(0.12 * fs)
                    if 0 <= base_idx < len(cleaned):
                        baseline = float(cleaned[base_idx])
                        feat["st_dev"] = st_val - baseline

        if t_pk is not None and r is not None:
            feat["t_r_gap_ms"] = (t_pk - r) / fs * 1000.0

        beat_features.append(feat)

    return beat_features


def qt_correction_bazett(qt_ms, rr_s):
    if not np.isfinite(qt_ms) or not np.isfinite(rr_s) or rr_s <= 0:
        return np.nan
    qt_s = qt_ms / 1000.0
    return (qt_s / np.sqrt(rr_s)) * 1000.0


def beat_ai_anomaly_detection(beat_df: pd.DataFrame):
    """
    Unsupervised ML on beat-level features.
    """
    beat_df = beat_df.copy()
    use_cols = [c for c in ["qrs_ms", "qt_ms", "st_dev", "rr_prev_ms", "r_amp"] if c in beat_df.columns]

    if len(use_cols) < 2 or len(beat_df) < 10:
        beat_df["ai_anomaly"] = 0
        beat_df["ai_score"] = np.nan
        return beat_df, {"ai_model": "skipped (not enough beat features/data)"}

    X = beat_df[use_cols].copy()
    for c in use_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if np.isfinite(med) else 0.0)

    model = IsolationForest(
        n_estimators=200,
        contamination=min(0.10, max(0.02, 2 / max(len(X), 1))),
        random_state=42,
    )
    preds = model.fit_predict(X)  # -1 anomaly, 1 normal
    scores = model.decision_function(X)

    beat_df["ai_anomaly"] = (preds == -1).astype(int)
    beat_df["ai_score"] = scores

    summary = {
        "ai_model": "IsolationForest",
        "features": use_cols,
        "anomalous_beats": int(beat_df["ai_anomaly"].sum()),
        "total_beats": int(len(beat_df)),
        "anomaly_ratio_pct": float(100 * beat_df["ai_anomaly"].mean()),
    }
    return beat_df, summary


# ----------------------------
# Multi-lead processing
# ----------------------------

def build_lead_trial_order(sig_names: List[str]) -> List[int]:
    preferred = ["II", "I", "V5", "V6", "V2", "III", "V1", "V3", "V4", "AVL", "AVF", "AVR", "MLII"]
    upper = [s.upper() for s in sig_names]

    order = []
    for p in preferred:
        for i, s in enumerate(upper):
            if s == p and i not in order:
                order.append(i)

    for i in range(len(sig_names)):
        if i not in order:
            order.append(i)

    return order


def process_single_lead(raw_lead: np.ndarray, fs: int):
    """
    Returns dict with processed lead results or None.
    """
    # Reconstruct valid segments for digitized ECG
    lead = reconstruct_sequential(raw_lead, min_len=max(40, int(0.08 * fs)))
    if lead is None:
        return None

    lead = preprocess_lead_for_ecg(lead, fs)
    if lead is None:
        return None

    best = None

    # Try both polarities because digitized leads can be inverted
    for polarity, candidate in [("as_is", lead), ("inverted", -lead)]:
        signals_df, info, mode = safe_ecg_process_or_peaks(candidate, fs)
        if signals_df is None or info is None:
            continue

        rpeaks = info.get("ECG_R_Peaks", [])
        rpeaks = np.asarray(rpeaks if rpeaks is not None else [], dtype=int)

        if len(rpeaks) < 2:
            continue

        rr_metrics = compute_rr_metrics(rpeaks, fs)

        score = 0.0
        score += min(len(rpeaks), 10) * 10.0
        score += (1.0 if mode == "full" else 0.0) * 15.0
        score += (1.0 if "ECG_Quality" in signals_df.columns else 0.0) * 5.0
        score += np.nanstd(signals_df["ECG_Clean"].values) * 2.0 if "ECG_Clean" in signals_df.columns else 0.0

        candidate_result = {
            "lead_processed": candidate,
            "signals_df": signals_df,
            "info": info,
            "mode": mode,
            "polarity": polarity,
            "rpeaks": rpeaks,
            "rr_metrics": rr_metrics,
            "score": float(score),
        }

        if best is None or candidate_result["score"] > best["score"]:
            best = candidate_result

    return best


def process_multiple_leads(sig: np.ndarray, sig_names: List[str], fs: int, max_leads: int = 6):
    """
    Try several leads and return:
    - best lead for detailed analysis
    - per-lead summaries
    """
    order = build_lead_trial_order(sig_names)
    lead_summaries = []
    candidates = []

    for idx in order[:max_leads]:
        res = process_single_lead(sig[:, idx].astype(float), fs)
        if res is None:
            lead_summaries.append({
                "lead_index": idx,
                "lead_name": sig_names[idx],
                "status": "failed",
            })
            continue

        rpeaks = res["rpeaks"]
        lead_summaries.append({
            "lead_index": idx,
            "lead_name": sig_names[idx],
            "status": "ok",
            "mode": res["mode"],
            "polarity": res["polarity"],
            "rpeak_count": int(len(rpeaks)),
            "score": float(res["score"]),
            "hr_median_bpm": res["rr_metrics"].get("hr_median_bpm", np.nan),
        })
        candidates.append((idx, sig_names[idx], res))

    # If nothing worked, try all remaining leads before giving up
    if not candidates:
        for idx in order[max_leads:]:
            res = process_single_lead(sig[:, idx].astype(float), fs)
            if res is None:
                continue
            candidates.append((idx, sig_names[idx], res))
            lead_summaries.append({
                "lead_index": idx,
                "lead_name": sig_names[idx],
                "status": "ok",
                "mode": res["mode"],
                "polarity": res["polarity"],
                "rpeak_count": int(len(res["rpeaks"])),
                "score": float(res["score"]),
                "hr_median_bpm": res["rr_metrics"].get("hr_median_bpm", np.nan),
            })

    if not candidates:
        return None, lead_summaries

    # Pick highest-score lead
    best_idx, best_name, best_res = sorted(candidates, key=lambda x: x[2]["score"], reverse=True)[0]

    # Multi-lead HR consensus (median across successful leads)
    hr_values = [
        c[2]["rr_metrics"].get("hr_median_bpm", np.nan)
        for c in candidates
        if np.isfinite(c[2]["rr_metrics"].get("hr_median_bpm", np.nan))
    ]
    hr_consensus = float(np.median(hr_values)) if len(hr_values) > 0 else np.nan

    selected = {
        "lead_index": best_idx,
        "lead_name": best_name,
        **best_res,
        "multi_lead_hr_consensus_bpm": hr_consensus,
        "n_successful_leads": len(candidates),
        "lead_summaries": lead_summaries,
    }
    return selected, lead_summaries


# ----------------------------
# Main analysis
# ----------------------------

def analyze_record(record_base_path: str, verbose: bool = True):
    record, fs, sig, sig_names = load_wfdb_record(record_base_path, verbose=verbose)

    result = {
        "record_name": getattr(record, "record_name", Path(record_base_path).name),
        "sampling_rate_hz": fs,
        "n_samples": int(sig.shape[0]),
        "n_leads": int(sig.shape[1]),
        "lead_names": sig_names,
        "warnings": [],
        "screening_flags": [],
    }

    duration_s = sig.shape[0] / fs
    result["duration_seconds"] = float(duration_s)
    if duration_s < 4:
        result["warnings"].append(
            "Very short ECG clip (<4s): limited analysis. HR may work, interval/delineation metrics may fail."
        )

    selected, lead_summaries = process_multiple_leads(sig, sig_names, fs, max_leads=6)
    result["lead_trial_summary"] = lead_summaries

    if selected is None:
        result["warnings"].append("ECG processing failed on all tested leads.")
        result["ai_summary"] = {"ai_model": "skipped (no usable lead)"}
        return result, pd.DataFrame(), None

    lead_idx = selected["lead_index"]
    lead_name = selected["lead_name"]
    signals_df = selected["signals_df"]
    info = selected["info"]
    rpeaks = selected["rpeaks"]
    rr_metrics = selected["rr_metrics"]
    cleaned = signals_df["ECG_Clean"].values if "ECG_Clean" in signals_df.columns else selected["lead_processed"]

    result["analysis_lead_index"] = int(lead_idx)
    result["analysis_lead_name"] = str(lead_name)
    result["analysis_mode"] = selected["mode"]
    result["analysis_polarity"] = selected["polarity"]
    result["n_successful_leads"] = int(selected["n_successful_leads"])
    result["multi_lead_hr_consensus_bpm"] = selected["multi_lead_hr_consensus_bpm"]

    # Add basic RR/HR metrics even on short ECG if >= 2 peaks
    if rr_metrics:
        result.update(rr_metrics)
        result["screening_flags"].extend(classify_rhythm_basic(rr_metrics))
    else:
        result["warnings"].append("Too few R-peaks for HR estimation.")
        result["ai_summary"] = {"ai_model": "skipped (too few beats)"}
        return result, pd.DataFrame(), signals_df

    # signal quality only if available in full NeuroKit pipeline
    if "ECG_Quality" in signals_df.columns:
        q = signals_df["ECG_Quality"].values
        result["signal_quality_mean"] = float(np.nanmean(q))
        result["signal_quality_median"] = float(np.nanmedian(q))

    # If fewer than 3 peaks, skip delineation but keep plot + HR
    if len(rpeaks) < 3:
        result["warnings"].append("Not enough R-peaks detected for interval/delineation analysis.")
        result["ai_summary"] = {"ai_model": "skipped (too few beats for beat-feature AI)"}
        return result, pd.DataFrame(), signals_df

    # DWT delineation often fails on noisy/short digitized ECGs -> safe fallback
    beat_features = estimate_intervals_from_delineation(cleaned, rpeaks, fs)
    beat_df = pd.DataFrame(beat_features)

    # Add beat-level rr_prev and r_amp
    rpeaks_arr = np.asarray(rpeaks, dtype=int)
    if len(rpeaks_arr) > 0:
        rr_prev_ms = [np.nan] + list(np.diff(rpeaks_arr) / fs * 1000.0)
        r_amp = [float(cleaned[r]) if 0 <= r < len(cleaned) else np.nan for r in rpeaks_arr]

        n = min(len(beat_df), len(rpeaks_arr))
        if n > 0:
            beat_df = beat_df.iloc[:n].copy()
            beat_df["rr_prev_ms"] = rr_prev_ms[:n]
            beat_df["r_amp"] = r_amp[:n]

    if beat_df.empty:
        result["warnings"].append("Beat delineation failed; interval metrics unavailable.")
        result["ai_summary"] = {"ai_model": "skipped (no beat features)"}
        return result, beat_df, signals_df

    # Summary interval metrics (screening only)
    qrs_med = median_or_nan(beat_df["qrs_ms"]) if "qrs_ms" in beat_df else np.nan
    qt_med = median_or_nan(beat_df["qt_ms"]) if "qt_ms" in beat_df else np.nan
    st_med = median_or_nan(beat_df["st_dev"]) if "st_dev" in beat_df else np.nan

    hr_med = result.get("hr_median_bpm", np.nan)
    rr_med_s = (60.0 / hr_med) if np.isfinite(hr_med) and hr_med > 0 else np.nan
    qtc_bazett = qt_correction_bazett(qt_med, rr_med_s)

    result["qrs_median_ms"] = qrs_med
    result["qt_median_ms"] = qt_med
    result["qtc_bazett_median_ms"] = qtc_bazett
    result["st_deviation_median_selected_lead"] = st_med

    # Rule-based screening flags
    if np.isfinite(qrs_med) and qrs_med >= 120:
        result["screening_flags"].append("Wide QRS pattern (>= 120 ms) on analyzed lead")
    if np.isfinite(qtc_bazett):
        if qtc_bazett > 470:
            result["screening_flags"].append("QTc appears prolonged (screening flag)")
        elif qtc_bazett < 330:
            result["screening_flags"].append("QTc appears short (screening flag)")
    if np.isfinite(st_med) and abs(st_med) > 0.1:
        result["screening_flags"].append("Relative ST deviation detected on analyzed lead (screening flag)")

    # AI beat anomaly detection
    beat_df, ai_summary = beat_ai_anomaly_detection(beat_df)
    result["ai_summary"] = ai_summary

    if ai_summary.get("anomaly_ratio_pct", 0) > 20:
        result["screening_flags"].append("High proportion of anomalous beats by AI (screening flag)")
    elif ai_summary.get("anomaly_ratio_pct", 0) > 5:
        result["screening_flags"].append("Some anomalous beats detected by AI")

    return result, beat_df, signals_df


# ----------------------------
# Plotting / CLI
# ----------------------------

def save_quick_plot(signals_df: Optional[pd.DataFrame], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if signals_df is None or "ECG_Clean" not in signals_df.columns:
        return False

    x = signals_df["ECG_Clean"].values
    plt.figure(figsize=(14, 4))
    plt.plot(x, linewidth=1)

    if "ECG_R_Peaks" in signals_df.columns:
        r_idx = np.where(np.asarray(signals_df["ECG_R_Peaks"].values) == 1)[0]
        r_idx = r_idx[(r_idx >= 0) & (r_idx < len(x))]
        if len(r_idx) > 0:
            plt.scatter(r_idx, x[r_idx], s=12)

    plt.title("Analyzed Lead (Cleaned) with R-peaks")
    plt.xlabel("Samples")
    plt.ylabel("Normalized amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main():
    record_base = "output_data/wfdb/record"
    if len(sys.argv) > 1:
        record_base = sys.argv[1]

    if not os.path.exists(record_base + ".hea") or not os.path.exists(record_base + ".dat"):
        raise FileNotFoundError(f"Could not find WFDB pair: {record_base}.hea and {record_base}.dat")

    result, beat_df, signals_df = analyze_record(record_base, verbose=True)

    print("\n=== BASIC ECG ANALYSIS (SCREENING, NOT DIAGNOSIS) ===")
    print(json.dumps(result, indent=2, default=lambda x: None))

    out_dir = Path("output_data/analysis_output")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=lambda x: None)

    if beat_df is not None and not beat_df.empty:
        beat_df.to_csv(out_dir / "beat_features.csv", index=False)
        print(f"\nSaved beat features: {out_dir / 'beat_features.csv'}")

    plot_saved = save_quick_plot(signals_df, out_dir / "analyzed_lead.png")
    if plot_saved:
        print(f"Saved plot: {out_dir / 'analyzed_lead.png'}")
    else:
        print("Plot not saved (no usable processed signal available).")

    print(f"\nSaved summary: {out_dir / 'summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()