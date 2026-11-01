

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import TRIAL_KEYS
from src.validation_common import (transverse_residual, resample_uniform, welch_psd,
                                    dominant_frequency, native_sampling_rate)

# A trial's RAW (native) mouse-event rate must be at least this multiple of
# the injected frequency for a frequency-domain check to be meaningful.
# Bare Nyquist (2x) is a mathematical minimum for reconstruction in
# principle, but is fragile for a noisy, non-bandlimited real signal;
# 3x gives some margin. Trials that fail this are SKIPPED (not silently
# reported as "detected ~1 Hz"), because linear interpolation cannot
# manufacture frequency content the raw samples never captured.
NYQUIST_MARGIN = 3.0


# Frequency-validation search band.
# We already know the frequency that was intentionally injected.
# A narrower band prevents unrelated low-frequency voluntary movement
# from being selected as the "tremor" peak.
FREQUENCY_SEARCH_HALF_WIDTH_HZ = 1.5

# Minimum number of tremor cycles required for a meaningful
# frequency-domain validation.
MIN_TREMOR_CYCLES = 3.0


# --------------------------------------------------------------------------
# Per-trial checks
# --------------------------------------------------------------------------

def _frequency_check(trial_df: pd.DataFrame, injected_freq: float) -> Dict:
    """Validate recovery of the intentionally injected tremor frequency.

    Frequency validation uses ONLY move events and the signed transverse
    residual. Press/release/double-click events are not used because they
    are not guaranteed to provide sufficiently dense temporal sampling.

    A trial is skipped when:
      1. too few move samples are available,
      2. native sampling rate is insufficient,
      3. the movement duration is too short to contain enough tremor cycles,
      4. no PSD peak can be found near the injected frequency.

    This function does NOT modify the generated tremor or amplitude
    validation.
    """

    # ---------------------------------------------------------------
    # 1. Use only continuous mouse-move samples
    # ---------------------------------------------------------------
    
    if str(trial_df["task"].iloc[0]).lower() == "idle":
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": np.nan,
            "native_fs_hz": np.nan,
            "skipped_reason": "idle task — no intended movement direction",
        }
    
    move = (
        trial_df[trial_df.event == "move"]
        .sort_values("elapsed_sec")
        .copy()
    )

    if len(move) < 16:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": np.nan,
            "native_fs_hz": np.nan,
            "skipped_reason": f"only {len(move)} move samples (<16)",
        }

    # ---------------------------------------------------------------
    # 2. Check native sampling rate
    # ---------------------------------------------------------------
    t = move["elapsed_sec"].to_numpy(dtype=float)

    native_fs = native_sampling_rate(t)

    if not np.isfinite(native_fs):
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": np.nan,
            "native_fs_hz": np.nan,
            "skipped_reason": "native sampling rate could not be estimated",
        }

    if native_fs < NYQUIST_MARGIN * injected_freq:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": np.nan,
            "native_fs_hz": native_fs,
            "skipped_reason": (
                f"native sampling rate {native_fs:.1f} Hz < "
                f"{NYQUIST_MARGIN}x injected {injected_freq:.1f} Hz "
                f"— insufficient native sampling"
            ),
        }

    # ---------------------------------------------------------------
    # 3. Check whether the movement lasts long enough
    #    to contain several cycles of the injected tremor.
    # ---------------------------------------------------------------
    duration = float(t[-1] - t[0])

    required_duration = MIN_TREMOR_CYCLES / injected_freq

    if duration < required_duration:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": np.nan,
            "native_fs_hz": native_fs,
            "skipped_reason": (
                f"movement duration {duration:.3f}s < "
                f"{required_duration:.3f}s required for "
                f"{MIN_TREMOR_CYCLES:.0f} tremor cycles"
            ),
        }

    # ---------------------------------------------------------------
    # 4. Calculate SIGNED transverse residual
    #
    # This is the important signal for frequency validation.
    # Do NOT use residual magnitude here.
    # ---------------------------------------------------------------
    res_t = transverse_residual(move)

    valid = np.isfinite(t) & np.isfinite(res_t)

    t = t[valid]
    res_t = res_t[valid]

    if len(t) < 16:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": np.nan,
            "native_fs_hz": native_fs,
            "skipped_reason": "too few valid transverse-residual samples",
        }

    # ---------------------------------------------------------------
    # 5. Uniform resampling
    # ---------------------------------------------------------------
    t_u, res_u, fs = resample_uniform(
        t,
        res_t,
        max_freq_of_interest=10.0,
    )

    if len(t_u) < 16:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": fs,
            "native_fs_hz": native_fs,
            "skipped_reason": "too few points after resampling",
        }

    # ---------------------------------------------------------------
    # 6. Welch PSD
    # ---------------------------------------------------------------
    freqs, pxx = welch_psd(res_u, fs)

    if len(freqs) == 0 or len(pxx) == 0:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": fs,
            "native_fs_hz": native_fs,
            "skipped_reason": "empty PSD",
        }

    # ---------------------------------------------------------------
    # 7. IMPORTANT:
    #    Search only around the frequency that was actually injected.
    #
    #    Previously this was +/- 3 Hz.
    #    That allowed unrelated voluntary-movement components to win.
    # ---------------------------------------------------------------
    search_lo = max(
        0.5,
        injected_freq - FREQUENCY_SEARCH_HALF_WIDTH_HZ
    )

    search_hi = min(
        fs / 2.0 - 0.5,
        injected_freq + FREQUENCY_SEARCH_HALF_WIDTH_HZ
    )

    if search_hi <= search_lo:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": fs,
            "native_fs_hz": native_fs,
            "skipped_reason": "invalid frequency search band",
        }

    detected = dominant_frequency(
        freqs,
        pxx,
        search_lo=search_lo,
        search_hi=search_hi,
    )

    if detected is None:
        return {
            "detected_frequency_hz": np.nan,
            "frequency_error_hz": np.nan,
            "frequency_error_percent": np.nan,
            "fs_used": fs,
            "native_fs_hz": native_fs,
            "skipped_reason": (
                f"no PSD peak found within "
                f"{search_lo:.1f}-{search_hi:.1f} Hz"
            ),
        }

    # ---------------------------------------------------------------
    # 8. Frequency error
    # ---------------------------------------------------------------
    err_hz = abs(detected - injected_freq)

    err_pct = (
        100.0 * err_hz / injected_freq
        if injected_freq
        else np.nan
    )

    return {
        "detected_frequency_hz": float(detected),
        "frequency_error_hz": float(err_hz),
        "frequency_error_percent": float(err_pct),
        "fs_used": float(fs),
        "native_fs_hz": float(native_fs),
        "skipped_reason": None,
    }


def _amplitude_check(trial_df: pd.DataFrame) -> Dict:
    """RMS / peak / percentile amplitude of the 2D residual for one trial.

    NOTE: the configured amplitude A is the pre-multiplication scale in
    T(t) = A*r(t)*sin(...), with 0 < r(t) <= 1, so RMS residual is
    expected to be well below A, not equal to it. See docstring at the
    top of this file / README for the distinction.
    """
    move = trial_df[trial_df.event == "move"]
    res_x = ((move["observed_x"] - move["ground_truth_x"])*move["screen_width"]).to_numpy(dtype=float)
    res_y = ((move["observed_y"] - move["ground_truth_y"])*move["screen_height"]).to_numpy(dtype=float)
    mag = np.sqrt(res_x ** 2 + res_y ** 2)
    if len(mag) == 0:
        return {"rms_residual_px": np.nan, "peak_residual_px": np.nan,
                "std_residual_px": np.nan, "p95_residual_px": np.nan}
    return {
        "rms_residual_px": float(np.sqrt(np.mean(mag ** 2))),
        "peak_residual_px": float(np.max(mag)),
        "std_residual_px": float(np.std(mag)),
        "p95_residual_px": float(np.percentile(mag, 95)),
    }


def _envelope_stats(trial_df: pd.DataFrame) -> Dict:
    """Proxy check on the stochastic envelope r(t).

    IMPORTANT CAVEAT: r(t) itself is not persisted in the output CSVs
    (only the final observed trajectory is). We approximate it here via
    the Hilbert-transform envelope of the resampled transverse residual,
    which is only an accurate estimate of A*r(t) when r(t) varies slowly
    relative to the tremor period — an approximation, not an exact
    recovery. It is adequate for a smoothness/range sanity check, not for
    precisely re-deriving the OU process parameters.
    """
    from scipy.signal import hilbert

    move = trial_df[trial_df.event == "move"].sort_values("elapsed_sec")
    t = move["elapsed_sec"].to_numpy(dtype=float)
    res_t = transverse_residual(move)
    t_u, res_u, fs = resample_uniform(t, res_t, max_freq_of_interest=10.0)
    if len(t_u) < 16:
        return {"envelope_mean": np.nan, "envelope_std": np.nan,
                "envelope_min": np.nan, "envelope_max": np.nan,
                "envelope_lag1_autocorr": np.nan}

    envelope = np.abs(hilbert(res_u))
    lag1 = np.corrcoef(envelope[:-1], envelope[1:])[0, 1] if len(envelope) > 2 else np.nan #autocorrelation of the envelope, to check if the envelope is smooth or not. If the envelope is smooth, the autocorrelation should be high, if it is not smooth, the autocorrelation should be low.
    return {
        "envelope_mean": float(np.mean(envelope)),
        "envelope_std": float(np.std(envelope)),
        "envelope_min": float(np.min(envelope)),
        "envelope_max": float(np.max(envelope)),
        "envelope_lag1_autocorr": float(lag1) if lag1 == lag1 else np.nan,  # NaN-safe
    }


# --------------------------------------------------------------------------
# Clean-preservation / event-preservation (dataset-wide, not per-trial)
# --------------------------------------------------------------------------

def _clean_preservation_check(rows_df: pd.DataFrame, tol: float = 1e-9) -> Dict:
    """For clean trials (tremor_status == 0): observed must equal
    ground_truth exactly (within floating-point tolerance)."""
    clean = rows_df[rows_df.tremor_status == 0]
    if clean.empty:
        return {"n_clean_rows": 0, "n_violations": 0, "max_error_px": 0.0}
    err = np.sqrt(((clean.observed_x - clean.ground_truth_x)*clean.screen_width) ** 2 +
                  ((clean.observed_y - clean.ground_truth_y)*clean.screen_height) ** 2)
    violations = int((err > tol).sum())
    return {"n_clean_rows": len(clean), "n_violations": violations,
            "max_error_px": float(err.max())}


def _event_preservation_check(rows_df: pd.DataFrame, tol: float = 1e-9) -> Dict:
   
    tremor_rows = rows_df[rows_df.tremor_status == 1]
    TREMOR_EVENTS = {"move", "press", "release", "double_click"}
    non_move = tremor_rows[tremor_rows.event != "move"]
    if tremor_rows.empty:
        return {
            "n_expected_tremor_event_rows": 0,
            "n_other_event_rows": 0,
            "n_violations": 0,
            "max_error_px": 0.0,
        }

    tremor_events = tremor_rows[
        tremor_rows["event"].isin(TREMOR_EVENTS)
    ]
    other_events = tremor_rows[
        ~tremor_rows["event"].isin(TREMOR_EVENTS)
    ]
    if other_events.empty:
        max_error_px = 0.0
        violations = 0
    else:
        err_x_px = (
            other_events["observed_x"]
            - other_events["ground_truth_x"]
        ) * other_events["screen_width"]

        err_y_px = (
            other_events["observed_y"]
            - other_events["ground_truth_y"]
        ) * other_events["screen_height"]

        err = np.sqrt(err_x_px ** 2 + err_y_px ** 2)

        violations = int((err > tol).sum())
        max_error_px = float(err.max())

    return {
        "n_expected_tremor_event_rows": len(tremor_events),
        "n_other_event_rows": len(other_events),
        "n_violations": violations,
        "max_error_px": max_error_px,
    }


def _phase_check(meta_df: pd.DataFrame) -> Dict:
    phases = meta_df.loc[meta_df.tremor_status == 1, "tremor_phase_rad"].dropna()
    if phases.empty:
        return {"n_phases": 0, "phase_min": np.nan, "phase_max": np.nan, "phase_mean": np.nan}
    return {"n_phases": len(phases), "phase_min": float(phases.min()),
            "phase_max": float(phases.max()), "phase_mean": float(phases.mean())}


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_signal_validation(rows_df: pd.DataFrame, meta_df: pd.DataFrame,
                           frequencies_hz, amplitudes_px, outdir: str,
                           seed: int = 42) -> Dict:
    """Run all Part-1 (mathematical/signal) checks and write CSV/plot
    outputs into outdir. Returns a summary dict (also used by
    validation_report.py).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    tremor_meta = meta_df[meta_df.tremor_status == 1].dropna(subset=["tremor_frequency_hz"])

    trial_rows: List[Dict] = []
    skipped: List[str] = []

    for _, meta_row in tremor_meta.iterrows():
        key = tuple(meta_row[k] for k in TRIAL_KEYS)
        trial_df = rows_df[(rows_df.participant_id == key[0]) &
                            (rows_df.session_id == key[1]) &
                            (rows_df.trial_id == key[2])]
        if trial_df.empty:
            continue

        freq = float(meta_row["tremor_frequency_hz"])
        amp = float(meta_row["tremor_amplitude_px"])

        freq_result = _frequency_check(trial_df, freq)
        amp_result = _amplitude_check(trial_df)
        env_result = _envelope_stats(trial_df)

        if freq_result.get("skipped_reason"):
            skipped.append(
                f"{key}: {freq_result['skipped_reason']}"
            )

        if freq_result.get("skipped_reason"):
            frequency_status = "SKIPPED"
        elif freq_result["frequency_error_hz"] > 1.0:
            frequency_status = "CHECK"
        else:
            frequency_status = "VALIDATED"

        # IMPORTANT:
        # Always append the trial.
        # A skipped frequency check does NOT mean the trial
        # should disappear from trial_level_validation.csv.
        trial_rows.append({
            "participant_id": key[0],
            "session_id": key[1],
            "trial_id": key[2],
            "task": meta_row.get("task"),
            "injected_frequency_hz": freq,
            "configured_amplitude_px": amp,
            "frequency_validation_status": frequency_status,
            **freq_result,
            **amp_result,
            **env_result,
        })

    trial_df_out = pd.DataFrame(trial_rows)
    trial_df_out.to_csv(outdir / "trial_level_validation.csv", index=False)
    print(f"[validation_signal] wrote {outdir/'trial_level_validation.csv'} "
          f"({len(trial_df_out)} trials, {len(skipped)} skipped for frequency)")
    for s in skipped[:10]:
        print(f"  [skipped] {s}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped)-10} more (see log)")

    # --- aggregate by frequency ---------------------------------------------
    freq_agg = (trial_df_out.dropna(subset=["detected_frequency_hz"])
                .groupby("injected_frequency_hz")
                .agg(n_trials=("detected_frequency_hz", "count"),
                     mean_detected_hz=("detected_frequency_hz", "mean"),
                     mean_abs_error_hz=("frequency_error_hz", "mean"),
                     mean_error_percent=("frequency_error_percent", "mean"),
                     max_error_hz=("frequency_error_hz", "max"))
                .reset_index())
    freq_agg.to_csv(outdir / "frequency_validation.csv", index=False)
    print(f"[validation_signal] wrote {outdir/'frequency_validation.csv'}")

    # --- aggregate by amplitude ----------------------------------------------
    amp_agg = (trial_df_out.groupby("configured_amplitude_px")
               .agg(n_trials=("rms_residual_px", "count"),
                    mean_rms_px=("rms_residual_px", "mean"),
                    median_rms_px=("rms_residual_px", "median"),
                    mean_peak_px=("peak_residual_px", "mean"))
               .reset_index())
    amp_agg.to_csv(outdir / "amplitude_validation.csv", index=False)
    print(f"[validation_signal] wrote {outdir/'amplitude_validation.csv'}")

    amp_rms_corr = (trial_df_out[["configured_amplitude_px", "rms_residual_px"]]
                     .dropna().corr().iloc[0, 1]
                     if len(trial_df_out.dropna(subset=["rms_residual_px"])) > 2 else np.nan)

    clean_check = _clean_preservation_check(rows_df)
    event_check = _event_preservation_check(rows_df)
    phase_check = _phase_check(meta_df)

    summary = {
        "n_tremor_trials_evaluated": len(trial_df_out),
        "n_trials_skipped_for_frequency": len(skipped),
        "mean_frequency_error_hz": float(trial_df_out["frequency_error_hz"].mean(skipna=True)),
        "mean_frequency_error_percent": float(trial_df_out["frequency_error_percent"].mean(skipna=True)),
        "amplitude_rms_correlation": float(amp_rms_corr) if amp_rms_corr == amp_rms_corr else np.nan,
        "n_clean_rows": clean_check["n_clean_rows"],
        "clean_preservation_violations": clean_check["n_violations"],
        "clean_preservation_max_error_px": clean_check["max_error_px"],
        "event_preservation_violations": event_check["n_violations"],
        "event_preservation_max_error_px": event_check["max_error_px"],
        "phase_min_rad": phase_check["phase_min"],
        "phase_max_rad": phase_check["phase_max"],
        "envelope_mean_lag1_autocorr": float(trial_df_out["envelope_lag1_autocorr"].mean(skipna=True)),
    }
    pd.DataFrame([summary]).to_csv(outdir / "signal_validation_summary.csv", index=False)
    print(f"[validation_signal] wrote {outdir/'signal_validation_summary.csv'}")

    _make_signal_plots(rows_df, trial_df_out, frequencies_hz, plot_dir)

    return summary


def _make_signal_plots(rows_df: pd.DataFrame, trial_df_out: pd.DataFrame,
                        frequencies_hz, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PSD per frequency, using the densest available trial (same approach
    # as the generation-stage sanity plot, but on the signed transverse
    # residual computed independently here).
    fig, axes = plt.subplots(1, len(frequencies_hz), figsize=(5 * len(frequencies_hz), 4))
    if len(frequencies_hz) == 1:
        axes = [axes]
    tremor_rows = rows_df[rows_df.tremor_status == 1]
    for ax, freq in zip(axes, frequencies_hz):
        sub = tremor_rows[tremor_rows.tremor_frequency_hz == freq]
        if sub.empty:
            continue
        counts = sub[sub.event == "move"].groupby(TRIAL_KEYS).size().sort_values(ascending=False)
        if counts.empty:
            continue
        key = dict(zip(TRIAL_KEYS, counts.index[0]))
        trial = sub[(sub.participant_id == key["participant_id"]) & (sub.session_id == key["session_id"]) &
                    (sub.trial_id == key["trial_id"])]
        move = trial[trial.event == "move"].sort_values("elapsed_sec")
        t = move["elapsed_sec"].to_numpy(dtype=float)
        res_t = transverse_residual(move)
        t_u, res_u, fs = resample_uniform(t, res_t, max_freq_of_interest=10.0)
        if len(t_u) < 8:
            continue
        freqs, pxx = welch_psd(res_u, fs)
        ax.semilogy(freqs, pxx)
        ax.axvline(freq, color="red", linestyle="--", label=f"target {freq} Hz")
        ax.set_title(f"{freq} Hz PSD (fs≈{fs:.0f}Hz)")
        ax.set_xlabel("Hz")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "psd_by_frequency.png", dpi=120)
    plt.close(fig)

    # Detected vs injected frequency scatter.
    valid = trial_df_out.dropna(subset=["detected_frequency_hz"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(valid["injected_frequency_hz"], valid["detected_frequency_hz"], alpha=0.5)
        lims = [min(frequencies_hz) - 1, max(frequencies_hz) + 1]
        ax.plot(lims, lims, "r--", label="perfect agreement")
        ax.set_xlabel("Injected frequency (Hz)")
        ax.set_ylabel("Detected frequency (Hz)")
        ax.set_title("Detected vs. injected tremor frequency")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "detected_vs_injected_freq.png", dpi=120)
        plt.close(fig)

    # Amplitude vs RMS residual.
    valid = trial_df_out.dropna(subset=["rms_residual_px"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        for amp, grp in valid.groupby("configured_amplitude_px"):
            ax.scatter([amp] * len(grp), grp["rms_residual_px"], alpha=0.4, label=f"{amp}px")
        means = valid.groupby("configured_amplitude_px")["rms_residual_px"].mean()
        ax.plot(means.index, means.values, "k-o", label="mean")
        ax.set_xlabel("Configured amplitude A (px)")
        ax.set_ylabel("RMS residual (px)")
        ax.set_title("Configured amplitude vs. measured RMS residual")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plot_dir / "amplitude_vs_rms.png", dpi=120)
        plt.close(fig)
