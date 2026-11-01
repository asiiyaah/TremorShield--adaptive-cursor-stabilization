

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.config import TRIAL_KEYS


def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))


def _direction_changes(x: np.ndarray, y: np.ndarray, threshold_rad: float = np.deg2rad(15)) -> int:
    """Count heading changes between consecutive segments that exceed
    threshold_rad (default 15 degrees) — small floating jitter shouldn't
    count as a deliberate direction change."""
    if len(x) < 3:
        return 0
    dx, dy = np.diff(x), np.diff(y)
    heading = np.arctan2(dy, dx)
    dh = np.diff(heading)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    return int(np.sum(np.abs(dh) > threshold_rad))


def trial_spatial_features(trial_df: pd.DataFrame) -> Dict:
    """Compute paired clean-vs-tremor spatial features for one trial.
    Uses 'move' events only (press/release/double_click are discrete
    actions, not part of the continuous trajectory)."""
    move = trial_df[trial_df.event == "move"].sort_values("elapsed_sec") #move represents continuous cursor movement.others are discrete events, not part of the continuous trajectory.
    gt_x = move["ground_truth_x"].to_numpy(dtype=float)
    gt_y = move["ground_truth_y"].to_numpy(dtype=float)
    ob_x = move["observed_x"].to_numpy(dtype=float)
    ob_y = move["observed_y"].to_numpy(dtype=float)

    if len(gt_x) < 3:
        return {"skipped_reason": f"only {len(gt_x)} move samples (<3)"}
    
    dev_x_px = (
    ob_x - gt_x
) * move["screen_width"].to_numpy(dtype=float)

    dev_y_px = (
        ob_y - gt_y
    ) * move["screen_height"].to_numpy(dtype=float)
    dev = np.sqrt(
    dev_x_px ** 2 +
    dev_y_px ** 2
)

    
    W = move["screen_width"].to_numpy(dtype=float)
    H = move["screen_height"].to_numpy(dtype=float)

    gt_x_px = gt_x * W
    gt_y_px = gt_y * H

    ob_x_px = ob_x * W
    ob_y_px = ob_y * H
    path_clean = _path_length(gt_x_px, gt_y_px)
    path_tremor = _path_length(ob_x_px, ob_y_px)


    path_change_pct = (100.0 * (path_tremor - path_clean) / path_clean
                        if path_clean > 0 else np.nan)

    dc_clean = _direction_changes(gt_x_px, gt_y_px)
    dc_tremor = _direction_changes(ob_x_px, ob_y_px)

    bbox_clean = (float(gt_x_px.max() - gt_x_px.min()), float(gt_y_px.max() - gt_y_px.min()))
    bbox_tremor = (float(ob_x_px.max() - ob_x_px.min()), float(ob_y_px.max() - ob_y_px.min()))
    bbox_area_clean = bbox_clean[0] * bbox_clean[1]
    bbox_area_tremor = bbox_tremor[0] * bbox_tremor[1]
    bbox_change_pct = (100.0 * (bbox_area_tremor - bbox_area_clean) / bbox_area_clean
                        if bbox_area_clean > 0 else np.nan)

    return {
        "n_move_samples": len(gt_x),
        "path_length_clean_px": path_clean,
        "path_length_tremor_px": path_tremor,
        "path_length_change_percent": path_change_pct,
        "deviation_mean_px": float(dev.mean()),
        "deviation_median_px": float(np.median(dev)),
        "deviation_rms_px": float(np.sqrt(np.mean(dev ** 2))),
        "deviation_p95_px": float(np.percentile(dev, 95)),
        "deviation_max_px": float(dev.max()),
        "direction_changes_clean": dc_clean,
        "direction_changes_tremor": dc_tremor,
        "direction_changes_increase": dc_tremor - dc_clean,
        "x_std_clean_px": float(gt_x_px.std()),
        "x_std_tremor_px": float(ob_x_px.std()),
        "y_std_clean_px": float(gt_y_px.std()),
        "y_std_tremor_px": float(ob_y_px.std()),
        "bbox_area_clean_px2": bbox_area_clean,
        "bbox_area_tremor_px2": bbox_area_tremor,
        "bbox_area_change_percent": bbox_change_pct,
        "skipped_reason": None,
    }


def run_spatial_validation(rows_df: pd.DataFrame, meta_df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """Compute spatial_validation.csv over every tremor trial. Returns the
    resulting DataFrame (also consumed by validation_statistics.py)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    tremor_meta = meta_df[meta_df.tremor_status == 1].dropna(subset=["tremor_frequency_hz"])

    rows = []
    skipped = 0
    for _, meta_row in tremor_meta.iterrows():
        key = tuple(meta_row[k] for k in TRIAL_KEYS)
        trial_df = rows_df[(rows_df.participant_id == key[0]) &
                            (rows_df.session_id == key[1]) &
                            (rows_df.trial_id == key[2])]
        if trial_df.empty:
            continue
        feats = trial_spatial_features(trial_df)
        if feats.get("skipped_reason"):
            skipped += 1
            continue
        rows.append({
            "participant_id": key[0], "session_id": key[1], "trial_id": key[2],
            "task": meta_row.get("task"),
            "tremor_frequency_hz": float(meta_row["tremor_frequency_hz"]),
            "tremor_amplitude_px": float(meta_row["tremor_amplitude_px"]),
            **feats,
        })

    spatial_df = pd.DataFrame(rows)
    spatial_df.to_csv(outdir / "spatial_validation.csv", index=False)
    print(f"[validation_spatial] wrote {outdir/'spatial_validation.csv'} "
          f"({len(spatial_df)} trials, {skipped} skipped)")

    _make_spatial_plots(spatial_df, plot_dir)
    return spatial_df


def _make_spatial_plots(spatial_df: pd.DataFrame, plot_dir: Path) -> None:
    if spatial_df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # RMS deviation by amplitude
    groups = [g["deviation_rms_px"].dropna().to_numpy()
              for _, g in spatial_df.groupby("tremor_amplitude_px")]
    labels = sorted(spatial_df["tremor_amplitude_px"].unique())
    # boxplot(labels=...) was renamed to tick_labels=... in newer matplotlib
    # (and the old name has since been dropped in some installs), so set
    # tick labels via set_xticklabels() instead — works on every version.
    axes[0].boxplot(groups)
    axes[0].set_xticklabels([f"{l}px" for l in labels])
    axes[0].set_title("RMS deviation by amplitude")
    axes[0].set_ylabel("px")

    # path length change % by frequency
    groups = [g["path_length_change_percent"].dropna().to_numpy()
              for _, g in spatial_df.groupby("tremor_frequency_hz")]
    labels = sorted(spatial_df["tremor_frequency_hz"].unique())
    axes[1].boxplot(groups)
    axes[1].set_xticklabels([f"{l}Hz" for l in labels])
    axes[1].set_title("Path-length increase (%) by frequency")
    axes[1].set_ylabel("%")

    # direction change increase by amplitude
    groups = [g["direction_changes_increase"].dropna().to_numpy()
              for _, g in spatial_df.groupby("tremor_amplitude_px")]
    labels = sorted(spatial_df["tremor_amplitude_px"].unique())
    axes[2].boxplot(groups)
    axes[2].set_xticklabels([f"{l}px" for l in labels])
    axes[2].set_title("Direction-change increase by amplitude")
    axes[2].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(plot_dir / "spatial_summary.png", dpi=120)
    plt.close(fig)
