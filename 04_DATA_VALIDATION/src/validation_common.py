

from typing import Optional, Tuple

import numpy as np

from src.tremor_model import _unit_normals as compute_unit_normals  # re-export, see below
from src.config import EPS_DT


def transverse_residual(trial_df) -> np.ndarray:
    """Recompute the SIGNED tremor residual projected onto the local
    direction-of-travel normal, i.e. recover T(t) = observed - ground_truth,
    expressed as a single signed scalar rather than an (x, y) vector.

    This works because n(t) is deterministic given only ground_truth_x/y
    and dt — the exact same function used at injection time
    (src/tremor_model._unit_normals) can be re-run here on the ground-truth
    trajectory to recover the same normals, without needing to have stored
    n(t) anywhere in the output CSVs.

        residual_transverse(t) = (obs_x - gt_x)*n_x(t) + (obs_y - gt_y)*n_y(t)

    Using the SIGNED transverse residual (not sqrt(res_x^2+res_y^2)) is
    required for correct frequency-domain analysis: the magnitude is a
    full-wave-rectified sinusoid, and rectification doubles the apparent
    frequency (|sin(2*pi*f*t)| has fundamental frequency 2f, not f).
    """
    gt_x = trial_df["ground_truth_x"].to_numpy(dtype=float)
    gt_y = trial_df["ground_truth_y"].to_numpy(dtype=float)
    obs_x = trial_df["observed_x"].to_numpy(dtype=float)
    obs_y = trial_df["observed_y"].to_numpy(dtype=float)
    dt = trial_df["dt"].fillna(0.0).to_numpy(dtype=float)

    nx, ny = compute_unit_normals(gt_x, gt_y, dt)
    res_x = obs_x - gt_x
    res_y = obs_y - gt_y
    return res_x * nx + res_y * ny


def native_sampling_rate(t: np.ndarray) -> float:
    """Median native sampling rate (Hz) of an irregular timestamp array.
    Used to check whether a trial's RAW event rate can even represent a
    given tremor frequency BEFORE resampling — interpolation cannot
    manufacture frequency content that was never sampled in the first
    place. A trial with, say, a 2 Hz native mouse-move rate cannot be used
    to validate an injected 8-10 Hz tremor no matter how finely it is
    resampled afterwards; see resample_uniform() / _frequency_check().
    """
    t = np.sort(t[np.isfinite(t)])
    diffs = np.diff(t)
    diffs = diffs[diffs > EPS_DT]
    if len(diffs) == 0:
        return 0.0
    return 1.0 / np.median(diffs)


def resample_uniform(t: np.ndarray, signal: np.ndarray,
                      max_freq_of_interest: float = 10.0,
                      nyquist_multiplier: float = 4.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resample an irregularly-sampled (t, signal) pair onto a uniform grid
    via linear interpolation, choosing a sampling rate fs such that the
    Nyquist frequency comfortably exceeds max_freq_of_interest (default
    10 Hz, i.e. the highest tremor frequency used in this project).

    Mouse-move events are event-driven, not fixed-rate: idle/slow segments
    can have gaps far larger than a 4-10 Hz tremor period, which aliases
    badly under Welch's method if you use the raw event timestamps
    directly. We therefore always resample first and report the fs used.

    Returns (t_uniform, signal_uniform, fs_used). Returns (empty, empty, fs)
    if there are fewer than 8 usable points (not enough for a meaningful
    PSD estimate) — callers should check len(t_uniform) before using it.
    """
    order = np.argsort(t)
    t = t[order]
    signal = signal[order]
    # Drop non-finite / duplicate-timestamp points before resampling.
    finite = np.isfinite(t) & np.isfinite(signal)
    t, signal = t[finite], signal[finite]
    t, unique_idx = np.unique(t, return_index=True)
    signal = signal[unique_idx]

    if len(t) < 8:
        return np.array([]), np.array([]), float("nan")

    diffs = np.diff(t)
    diffs = diffs[diffs > EPS_DT]
    median_dt = np.median(diffs) if len(diffs) else 1.0 / 60.0
    native_fs = 1.0 / median_dt

    fs = max(native_fs, nyquist_multiplier * max_freq_of_interest)

    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)
    if len(t_uniform) < 8:
        return np.array([]), np.array([]), fs
    signal_uniform = np.interp(t_uniform, t, signal)
    return t_uniform, signal_uniform, fs


def welch_psd(signal_uniform: np.ndarray, fs: float, nperseg_cap: int = 256):
    """Thin wrapper around scipy.signal.welch with a sane default nperseg."""
    from scipy.signal import welch
    if len(signal_uniform) < 8 or not np.isfinite(fs):
        return np.array([]), np.array([])
    nperseg = min(nperseg_cap, len(signal_uniform))
    freqs, pxx = welch(signal_uniform - np.mean(signal_uniform), fs=fs, nperseg=nperseg)
    return freqs, pxx


def dominant_frequency(freqs: np.ndarray, pxx: np.ndarray,
                        search_lo: float = 1.0, search_hi: float = 20.0) -> Optional[float]:
    """Return the frequency (Hz) of the PSD's largest peak within
    [search_lo, search_hi], excluding the DC/very-low-frequency band
    (drift, slow envelope modulation) which would otherwise dominate.
    Returns None if there's no data in that band.
    """
    if len(freqs) == 0:
        return None
    mask = (freqs >= search_lo) & (freqs <= search_hi)
    if not mask.any():
        return None
    band_freqs = freqs[mask]
    band_pxx = pxx[mask]
    return float(band_freqs[np.argmax(band_pxx)])
