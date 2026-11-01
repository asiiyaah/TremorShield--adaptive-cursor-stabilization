"""validate_dataset(): leakage assertions, quality-check counts, and a
frequency-validation figure (trajectory / residual / PSD per Hz).

This is a GENERATION-STAGE sanity check, not the full RobustPDX / Kalman
filter validation planned for the next stage.
"""

import os

import numpy as np
import pandas as pd

from .config import TRIAL_KEYS, TremorConfig


def validate_dataset(train_users: set, test_users: set,
                      train_combined: pd.DataFrame, test_combined: pd.DataFrame,
                      train_meta: pd.DataFrame, test_meta: pd.DataFrame,
                      cfg: TremorConfig, outdir: str) -> None:
    print("\n" + "=" * 70)
    print("VALIDATION / QUALITY CHECKS")
    print("=" * 70)

    assert train_users.isdisjoint(test_users), "Participant leakage detected!"
    print(f"[OK] train/test participants disjoint "
          f"({len(train_users)} train, {len(test_users)} test)")

    train_trials = set(map(tuple, train_meta[TRIAL_KEYS].to_numpy()))
    test_trials = set(map(tuple, test_meta[TRIAL_KEYS].drop_duplicates().to_numpy()))
    overlap = train_trials & test_trials
    assert not overlap, f"Trial leakage detected: {overlap}"
    print(f"[OK] no trial appears in both train and test "
          f"({len(train_trials)} train trials, {len(test_trials)} distinct test trials)")

    print(f"\nParticipants: {len(train_users)+len(test_users)} total "
          f"({len(train_users)} train / {len(test_users)} test)")
    print(f"Training trials: {len(train_meta)} "
          f"({(train_meta.tremor_status==1).sum()} tremor / "
          f"{(train_meta.tremor_status==0).sum()} clean, "
          f"{(train_meta.tremor_status==1).mean()*100:.1f}% tremor)")
    print(f"Test trials (distinct): {len(test_trials)}; "
          f"paired renderings written: {len(test_meta)} "
          f"({(test_meta.tremor_status==1).sum()} tremor / "
          f"{(test_meta.tremor_status==0).sum()} clean)")

    print(f"\nRows: train_combined={len(train_combined):,}, "
          f"test_combined={len(test_combined):,}")

    for name, d in [("train_combined", train_combined), ("test_combined", test_combined)]:
        num = d.select_dtypes(include=[np.number])
        n_missing = num.isna().sum().sum()
        n_inf = np.isinf(num.to_numpy(dtype=float, na_value=0.0)).sum()
        n_dup = d.duplicated(subset=["participant_id", "session_id", "trial_id", "elapsed_sec", "tremor_status"]).sum()
        print(f"[{name}] missing(numeric)={n_missing:,}  inf={n_inf:,}  "
              f"duplicate rows={n_dup:,}")

    tremor_meta = pd.concat([train_meta, test_meta])
    tremor_meta = tremor_meta[tremor_meta.tremor_status == 1]
    print("\nFrequency distribution among tremor trials:")
    print(tremor_meta["tremor_frequency_hz"].value_counts().sort_index().to_string())
    print("\nAmplitude distribution among tremor trials:")
    print(tremor_meta["tremor_amplitude_px"].value_counts().sort_index().to_string())

    print("\nTask distribution — train:")
    print(train_meta["task"].value_counts().to_string())
    print("\nTask distribution — test (distinct trials):")
    print(test_meta.drop_duplicates(TRIAL_KEYS)["task"].value_counts().to_string())

    #_make_validation_plots(train_combined, cfg, outdir)


def _make_validation_plots(train_combined: pd.DataFrame, cfg: TremorConfig, outdir: str) -> None:
    """One example tremor trial per frequency: trajectory overlay, residual
    over time, and PSD (Welch) of the residual, to sanity-check that a
    requested f Hz actually shows a spectral peak near f Hz.

    Two subtleties handled here (see chat / tremor_model.py docstring):
      1. Mouse-move events are irregular; sparse trials alias badly under
         Welch, so we resample onto a uniform grid first and pick the
         densest available trial per frequency.
      2. The PSD is computed on the SIGNED residual (res_x), not
         sqrt(res_x^2+res_y^2) — the magnitude is a rectified sinusoid and
         rectification doubles the apparent frequency.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    tremor_rows = train_combined[train_combined.tremor_status == 1]
    fig, axes = plt.subplots(len(cfg.frequencies_hz), 3, figsize=(15, 4 * len(cfg.frequencies_hz)))

    for row_i, freq in enumerate(cfg.frequencies_hz):
            sub = tremor_rows[tremor_rows.tremor_frequency_hz == freq]
            if sub.empty:
                  continue
            move_counts = (sub[sub.event == "move"]
                        .groupby(TRIAL_KEYS).size().sort_values(ascending=False))
            if move_counts.empty:
                  continue
            key = dict(zip(TRIAL_KEYS, move_counts.index[0]))
            trial = sub[(sub.participant_id == key["participant_id"]) &
                        (sub.session_id == key["session_id"]) &
                        (sub.trial_id == key["trial_id"])].sort_values("elapsed_sec")
            move = trial[trial.event == "move"]

            t = move["elapsed_sec"].to_numpy()
            W = move["screen_width"].to_numpy(dtype=float)
            H = move["screen_height"].to_numpy(dtype=float)

            res_x_norm = (
            move["observed_x"].to_numpy()
            - move["ground_truth_x"].to_numpy()
            )

            res_y_norm = (
            move["observed_y"].to_numpy()
            - move["ground_truth_y"].to_numpy()
            )

            res_x = res_x_norm * W
            res_y = res_y_norm * H

            res_mag = np.sqrt(res_x**2 + res_y**2)

            ax0, ax1, ax2 = axes[row_i]

            ax0.plot(move["ground_truth_x"], move["ground_truth_y"], label="ground truth", lw=1.5)
            ax0.plot(move["observed_x"], move["observed_y"], label="observed (tremor)", lw=0.8, alpha=0.8)
            ax0.set_title(f"{freq} Hz — trajectory ({key['participant_id']}, trial {key['trial_id']}, "
                        f"n={len(move)})")
            ax0.legend(fontsize=8)
            ax0.set_aspect("equal", adjustable="datalim")

            ax1.plot(t, res_mag, lw=0.8)
            ax1.set_title(f"{freq} Hz — residual magnitude (px)")
            ax1.set_xlabel("elapsed_sec")

            median_dt = np.median(np.diff(t)[np.diff(t) > 0]) if len(t) > 2 else 1.0 / 60.0
            fs = min(1.0 / median_dt, 4 * max(cfg.frequencies_hz))
            fs = max(fs, 4 * max(cfg.frequencies_hz))  # ensure Nyquist >= 2x max freq
            t_uniform = np.arange(t.min(), t.max(), 1.0 / fs)
            if len(t_uniform) > 8:
                  res_x_uniform = np.interp(t_uniform, t, res_x)  # signed! see docstring
                  f_psd, pxx = welch(res_x_uniform - res_x_uniform.mean(), fs=fs,
                                    nperseg=min(256, len(res_x_uniform)))
                  ax2.semilogy(f_psd, pxx)
            else:
                  ax2.text(0.5, 0.5, "too few samples\nfor a reliable PSD",
                        ha="center", va="center", transform=ax2.transAxes)
            ax2.axvline(freq, color="red", linestyle="--", label=f"target {freq} Hz")
            ax2.set_title(f"{freq} Hz — PSD of residual (resampled, fs≈{fs:.0f}Hz)")
            ax2.set_xlabel("Hz")
            ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(plot_dir, "tremor_frequency_validation.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[validate_dataset] Saved validation figure: {out_path}")
