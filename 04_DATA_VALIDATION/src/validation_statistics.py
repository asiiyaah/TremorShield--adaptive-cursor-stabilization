

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def distribution_stats(x: np.ndarray) -> Dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {k: np.nan for k in
                ["n", "mean", "median", "std", "iqr", "ci95_low", "ci95_high"]}
    mean = float(np.mean(x))
    sem = stats.sem(x) if len(x) > 1 else 0.0
    ci = 1.96 * sem
    return {
        "n": len(x),
        "mean": mean,
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def compare_two_distributions(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> Dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    result = {f"{label_a}_{k}": v for k, v in distribution_stats(a).items()}
    result.update({f"{label_b}_{k}": v for k, v in distribution_stats(b).items()})
    if len(a) >= 2 and len(b) >= 2:
        ks_stat, ks_p = stats.ks_2samp(a, b)
        wasserstein = stats.wasserstein_distance(a, b)
    else:
        ks_stat = ks_p = wasserstein = float("nan")
    result.update({
        "ks_statistic": float(ks_stat), "ks_p_value": float(ks_p),
        "wasserstein_distance": float(wasserstein),
        "cohens_d": cohens_d(a, b),
    })
    return result


# Features present in BOTH our spatial_validation.csv and
# validation_robustpdx feature output, i.e. the only fair comparisons.
_SHARED_FEATURES = {
    # synthetic spatial_validation.csv column -> robustpdx feature column
    "path_length_tremor_px": "path_length_px",
    "x_std_tremor_px": "x_std_px",
    "y_std_tremor_px": "y_std_px",
    "bbox_area_tremor_px2": "bbox_area_px2",
}


def run_statistical_comparison(spatial_df: pd.DataFrame,
                                robustpdx_df: Optional[pd.DataFrame],
                                outdir: str) -> pd.DataFrame:
    """Compare synthetic-tremor spatial features against RobustPDX PD /
    non-PD groups (if available). Writes robustpdx_comparison.csv and
    boxplot/ECDF figures. Returns the comparison DataFrame (possibly
    with a single 'skipped' row if RobustPDX wasn't available).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if robustpdx_df is None or robustpdx_df.empty or spatial_df.empty:
        note = pd.DataFrame([{
            "note": "RobustPDX comparison skipped: RobustPDX data not available "
                    "or no synthetic tremor trials to compare. Synthetic-only "
                    "spatial results are still in spatial_validation.csv."
        }])
        note.to_csv(outdir / "robustpdx_comparison.csv", index=False)
        print("[validation_statistics] RobustPDX comparison skipped "
              "(no RobustPDX data / no spatial data) — see robustpdx_comparison.csv")
        return note

    groups = sorted(robustpdx_df["status"].astype(str).str.lower().unique())
    rows = []
    for synth_col, ref_col in _SHARED_FEATURES.items():
        if synth_col not in spatial_df.columns or ref_col not in robustpdx_df.columns:
            continue
        synth_vals = spatial_df[synth_col].dropna().to_numpy()
        for group in groups:
            ref_vals = robustpdx_df.loc[robustpdx_df["status"].astype(str).str.lower() == group, ref_col].dropna().to_numpy()
            comp = compare_two_distributions(synth_vals, ref_vals, "synthetic", f"robustpdx_{group}")
            comp["feature"] = synth_col
            comp["reference_group"] = group
            rows.append(comp)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(outdir / "robustpdx_comparison.csv", index=False)
    print(f"[validation_statistics] wrote {outdir/'robustpdx_comparison.csv'} "
          f"({len(comparison_df)} feature x group comparisons)")

    _make_comparison_plots(spatial_df, robustpdx_df, plot_dir)
    return comparison_df


def _make_comparison_plots(spatial_df: pd.DataFrame, robustpdx_df: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for synth_col, ref_col in _SHARED_FEATURES.items():
        if synth_col not in spatial_df.columns or ref_col not in robustpdx_df.columns:
            continue
        fig, (ax_box, ax_ecdf) = plt.subplots(1, 2, figsize=(11, 4.5))

        data, labels = [], []
        data.append(spatial_df[synth_col].dropna().to_numpy()); labels.append("Synthetic")
        for group in sorted(robustpdx_df["status"].astype(str).unique()):
            vals = robustpdx_df.loc[robustpdx_df["status"].astype(str) == group, ref_col].dropna().to_numpy()
            data.append(vals); labels.append(f"RobustPDX {group}")
        ax_box.boxplot(data)
        ax_box.set_xticklabels(labels)
        ax_box.set_title(synth_col)
        ax_box.tick_params(axis="x", rotation=30)

        for d, lab in zip(data, labels):
            if len(d) == 0:
                continue
            xs = np.sort(d)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax_ecdf.plot(xs, ys, label=lab)
        ax_ecdf.set_title(f"ECDF — {synth_col}")
        ax_ecdf.legend(fontsize=7)

        fig.tight_layout()
        fig.savefig(plot_dir / f"compare_{synth_col}.png", dpi=120)
        plt.close(fig)
