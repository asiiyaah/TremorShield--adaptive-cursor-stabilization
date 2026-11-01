"""
TREMORSHIELD — validation stage entry point.

Runs, in order:

    1. Part 1 — mathematical/signal validation
       (src/validation_signal.py)

    2. Part 2a — spatial validation using TREMORSHIELD data
       (src/validation_spatial.py)

    3. Part 2b — RobustPDX spatial reference
       (src/validation_robustpdx.py)

    4. Statistical comparison
       (src/validation_statistics.py)

    5. validation_report.md
       (src/validation_report.py)

This stage does NOT re-run the tremor injection pipeline.
It only reads the CSVs already produced by run_pipeline.py.

RobustPDX is optional.

Example:

    python run_validation.py ^
        --train data/final/train_combined.csv ^
        --test data/final/test_combined.csv ^
        --metadata data/final/tremor_metadata.csv ^
        --outdir data/final/validation ^
        --robustpdx data/robustpdx/01_raw_dataset.csv
"""


import argparse
from pathlib import Path

import pandas as pd

from src.config import TremorConfig
from src.validation_signal import run_signal_validation
from src.validation_spatial import run_spatial_validation
from src.validation_robustpdx import (
    RobustPDXColumnMap,
    run_robustpdx_validation,
)
from src.validation_statistics import run_statistical_comparison
from src.validation_report import generate_report


def main():

    # ============================================================
    # COMMAND-LINE ARGUMENTS
    # ============================================================

    parser = argparse.ArgumentParser(
        description=(
            "TREMORSHIELD synthetic-tremor validation"
        )
    )

    parser.add_argument(
        "--train",
        default="data/final/train_combined.csv",
        help="Training combined dataset.",
    )

    parser.add_argument(
        "--test",
        default="data/final/test_combined.csv",
        help="Testing combined dataset.",
    )

    parser.add_argument(
        "--metadata",
        default="data/final/tremor_metadata.csv",
        help="Tremor metadata CSV.",
    )

    parser.add_argument(
        "--outdir",
        default="data/final/validation",
        help="Directory for validation outputs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by validation.",
    )

    # ------------------------------------------------------------
    # RobustPDX
    # ------------------------------------------------------------
    #
    # The actual RobustPDX dataset has a known structure:
    #
    #   Session_ID
    #   status
    #   r1points
    #   r2points
    #   r3points
    #
    # Therefore we DO NOT ask the user for generic x/y/trace/group
    # column names anymore.
    #

    parser.add_argument(
        "--robustpdx",
        default=None,
        help=(
            "Path to the RobustPDX raw CSV "
            "(01_raw_dataset.csv). Optional."
        ),
    )

    args = parser.parse_args()

    # ============================================================
    # OUTPUT DIRECTORY
    # ============================================================

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # CONFIGURATION
    # ============================================================

    cfg = TremorConfig()

    # The configuration is used here for the expected
    # tremor frequencies and amplitudes during validation.
    #
    # It does NOT re-run tremor injection.

    print("=" * 70)
    print("TREMORSHIELD VALIDATION")
    print("=" * 70)

    # ============================================================
    # LOAD TREMORSHIELD DATA
    # ============================================================

    print(
        "\n[run_validation] Loading TREMORSHIELD datasets..."
    )

    train_df = pd.read_csv(
        args.train
    )

    test_df = pd.read_csv(
        args.test
    )

    meta_df = pd.read_csv(
        args.metadata
    )

    # Combine train and test only for validation.
    #
    # This does NOT merge participants back into the training
    # process. It simply gives the validation stage access to
    # all generated trials.

    rows_df = pd.concat(
        [
            train_df,
            test_df,
        ],
        ignore_index=True
    )

    # ============================================================
    # DATASET SUMMARY
    # ============================================================

    dataset_summary = {
        "train rows":
            f"{len(train_df):,}",

        "test rows":
            f"{len(test_df):,}",

        "total rows validated":
            f"{len(rows_df):,}",

        "trials in metadata":
            len(meta_df),

        "tremor trials in metadata":
            int(
                (
                    meta_df["tremor_status"] == 1
                ).sum()
            ),

        "clean trials in metadata":
            int(
                (
                    meta_df["tremor_status"] == 0
                ).sum()
            ),
    }

    print(
        f"\n[run_validation] "
        f"{dataset_summary}\n"
    )

    # ============================================================
    # 1. MATHEMATICAL / SIGNAL VALIDATION
    # ============================================================

    print("=" * 70)
    print("1. MATHEMATICAL / SIGNAL VALIDATION")
    print("=" * 70)

    signal_summary = run_signal_validation(
        rows_df,
        meta_df,
        cfg.frequencies_hz,
        cfg.amplitudes_px,
        outdir,
        seed=args.seed,
    )

    # ============================================================
    # 2a. OWN-DATA SPATIAL VALIDATION
    # ============================================================

    print("\n" + "=" * 70)
    print("2a. OWN-DATA SPATIAL VALIDATION")
    print("=" * 70)

    spatial_df = run_spatial_validation(
        rows_df,
        meta_df,
        outdir,
    )

    # ============================================================
    # 2b. ROBUSTPDX SPATIAL REFERENCE
    # ============================================================

    print("\n" + "=" * 70)
    print("2b. ROBUSTPDX SPATIAL REFERENCE")
    print("=" * 70)

    # ------------------------------------------------------------
    # RobustPDX is optional.
    #
    # If --robustpdx is not supplied, the validation function
    # will skip RobustPDX.
    # ------------------------------------------------------------

    robustpdx_colmap = RobustPDXColumnMap(
        path=args.robustpdx
    )

    robustpdx_df = run_robustpdx_validation(
        robustpdx_colmap,
        outdir,
    )

    # ============================================================
    # 3. STATISTICAL COMPARISON
    # ============================================================

    print("\n" + "=" * 70)
    print("3. STATISTICAL COMPARISON")
    print("=" * 70)

    comparison_df = run_statistical_comparison(
        spatial_df,
        robustpdx_df,
        outdir,
    )

    # ============================================================
    # 4. FINAL VALIDATION REPORT
    # ============================================================

    print("\n" + "=" * 70)
    print("4. VALIDATION REPORT")
    print("=" * 70)

    generate_report(
        outdir,
        dataset_summary,
        signal_summary,
        spatial_df,
        robustpdx_df,
        comparison_df,
        cfg,
    )

    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Outputs written to: "
        f"{outdir.resolve()}"
    )


if __name__ == "__main__":
    main()