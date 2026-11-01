"""
TREMORSHIELD — pipeline entry point.

Pipeline:

    1. Load original cleaned data
    2. Run idle stationary-row reconstruction
    3. Load idle-filled dataset
    4. Rename/anonymize participants
    5. Participant-level train/test split
    6. Training trial-level tremor/clean split
    7. Build training condition
    8. Build TEST-A clean condition
    9. Build TEST-B tremor condition
   10. Validate dataset
   11. Save outputs

Idle reconstruction is performed BEFORE participant splitting
and tremor injection.

The original source CSV is never modified.

Usage:

    python run_pipeline.py \
        --input data/raw/all_participants_cleaned.csv \
        --outdir data/final
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

from src.config import TremorConfig, TRIAL_KEYS
from src.data_loader import load_data
from src.anonymize import rename_participants
from src.splitting import (
    split_participants,
    split_training_trials,
    assign_test_trials,
    _trial_table,
)
from src.pipeline import build_condition
from src.validation import validate_dataset
from src.io_writer import save_outputs


# ============================================================
# IDLE RECONSTRUCTION CONFIGURATION
# ============================================================

IDLE_SCRIPT = "add_idle_rows.py"

IDLE_FILLED_FILE = os.path.join(
    "data",
    "final",
    "all_participant_cleaned_idle_filled.csv"
)


# ============================================================
# RUN IDLE RECONSTRUCTION
# ============================================================

def run_idle_reconstruction(
    input_file,
    output_file,
):
    """
    Run add_idle_rows.py before the main TREMORSHIELD pipeline.

    The existing add_idle_rows.py is a standalone script, so
    we execute it first and then load its generated output.

    The script itself performs its own safety checks:
        - 37-column schema
        - original-row preservation
        - stationary generated rows
        - no fake events
        - chronological insertion
        - idle-only additions
    """

    print("\n" + "=" * 70)
    print("STEP 2 — IDLE STATIONARY ROW RECONSTRUCTION")
    print("=" * 70)

    print(
        f"Input : {input_file}"
    )

    print(
        f"Output: {output_file}"
    )

    # --------------------------------------------------------
    # Verify the idle reconstruction script exists
    # --------------------------------------------------------

    if not os.path.exists(IDLE_SCRIPT):

        raise FileNotFoundError(
            f"Idle reconstruction script not found:\n"
            f"{IDLE_SCRIPT}"
        )

    # --------------------------------------------------------
    # Run the existing add_idle_rows.py
    # --------------------------------------------------------

    result = subprocess.run(
        [
            sys.executable,
            IDLE_SCRIPT,
        ],
        check=True,
    )

    # --------------------------------------------------------
    # Verify output was created
    # --------------------------------------------------------

    if not os.path.exists(output_file):

        raise FileNotFoundError(
            "Idle reconstruction completed, "
            "but expected output file was not created:\n"
            f"{output_file}"
        )

    print(
        "\n✓ Idle reconstruction completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "TREMORSHIELD synthetic tremor "
            "injection pipeline"
        )
    )

    parser.add_argument(
        "--input",
        default="data/raw/all_participants_cleaned.csv",
    )

    parser.add_argument(
        "--outdir",
        default="data/final",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--train-tremor-frac",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--test-tremor-frac",
        type=float,
        default=1.00,
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIGURATION
    # ========================================================

    cfg = TremorConfig(
        seed=args.seed,
        train_frac=args.train_frac,
        train_tremor_frac=args.train_tremor_frac,
        test_tremor_frac=args.test_tremor_frac,
    )

    # ========================================================
    # STEP 1 — LOAD ORIGINAL CLEANED DATA
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 1 — LOAD CLEANED DATA")
    print("=" * 70)

    print(
        f"Loading: {args.input}"
    )

    # We load here only to report the original size.
    # The actual dataframe used downstream will be the
    # idle-filled dataframe produced in Step 2.

    df_original = load_data(
        args.input
    )

    print(
        f"Original rows: "
        f"{len(df_original):,}"
    )

    # ========================================================
    # STEP 2 — ADD IDLE STATIONARY ROWS
    # ========================================================

    run_idle_reconstruction(
        args.input,
        IDLE_FILLED_FILE,
    )

    # ========================================================
    # LOAD IDLE-FILLED DATASET
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING IDLE-FILLED DATASET")
    print("=" * 70)

    df_idle_filled = load_data(
        IDLE_FILLED_FILE
    )

    print(
        f"Rows after idle reconstruction: "
        f"{len(df_idle_filled):,}"
    )

    rows_added = (
        len(df_idle_filled)
        - len(df_original)
    )

    print(
        f"Stationary rows added: "
        f"{rows_added:,}"
    )

    # ========================================================
    # STEP 3 — ANONYMIZE PARTICIPANTS
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 3 — ANONYMIZE PARTICIPANTS")
    print("=" * 70)

    df, _mapping = rename_participants(
        df_idle_filled,
        cfg.seed,
    )

    # ========================================================
    # STEP 4 — PARTICIPANT-LEVEL SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 4 — PARTICIPANT-LEVEL SPLIT")
    print("=" * 70)

    train_users, test_users, participant_split = (
        split_participants(
            df,
            cfg.seed,
            cfg.train_frac,
        )
    )

    df_train = df[
        df.participant_id.isin(train_users)
    ].copy()

    df_test = df[
        df.participant_id.isin(test_users)
    ].copy()

    print(
        f"Train participants: "
        f"{len(train_users)}"
    )

    print(
        f"Test participants : "
        f"{len(test_users)}"
    )

    print(
        f"Train rows: "
        f"{len(df_train):,}"
    )

    print(
        f"Test rows : "
        f"{len(df_test):,}"
    )

    # ========================================================
    # STEP 5 — TRAINING TRIAL SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 5 — TRAINING TRIAL SPLIT")
    print("=" * 70)

    train_assignment = split_training_trials(
        df_train,
        cfg.seed,
        cfg.train_tremor_frac,
    )

    # ========================================================
    # STEP 6 — BUILD TRAINING CONDITION
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 6 — BUILD TRAINING CONDITION")
    print("=" * 70)

    train_combined, train_meta = build_condition(
        df_train,
        train_assignment,
        cfg,
        seed_offset=cfg.seed + 100,
    )

    train_clean = train_combined[
        train_combined.tremor_status == 0
    ].reset_index(drop=True)

    train_tremor = train_combined[
        train_combined.tremor_status == 1
    ].reset_index(drop=True)

    # ========================================================
    # STEP 7 — TEST-A CLEAN
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 7 — BUILD TEST-A CLEAN")
    print("=" * 70)

    test_trials_all = _trial_table(
        df_test
    )

    test_trials_all["tremor_status"] = 0

    test_clean, test_clean_meta = build_condition(
        df_test,
        test_trials_all,
        cfg,
        seed_offset=cfg.seed + 200,
    )

    # ========================================================
    # STEP 8 — TEST-B TREMOR
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 8 — BUILD TEST-B TREMOR")
    print("=" * 70)

    tremor_selection = assign_test_trials(
        df_test,
        cfg.test_tremor_frac,
    )

    tremor_selection = tremor_selection.copy()

    tremor_selection["tremor_status"] = 1

    test_tremor, test_tremor_meta = build_condition(
        df_test,
        tremor_selection,
        cfg,
        seed_offset=cfg.seed + 300,
    )

    # ========================================================
    # COMBINE TEST CONDITIONS
    # ========================================================

    test_combined = pd.concat(
        [
            test_clean,
            test_tremor,
        ],
        ignore_index=True,
    )

    test_meta = pd.concat(
        [
            test_clean_meta,
            test_tremor_meta,
        ],
        ignore_index=True,
    )

    # ========================================================
    # STEP 9 — VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 9 — VALIDATION")
    print("=" * 70)

    validate_dataset(
        train_users,
        test_users,
        train_combined,
        test_combined,
        train_meta,
        test_meta,
        cfg,
        args.outdir,
    )

    tremor_metadata_all = pd.concat(
        [
            train_meta,
            test_meta,
        ],
        ignore_index=True,
    )

    # ========================================================
    # STEP 10 — SAVE OUTPUTS
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 10 — SAVE OUTPUTS")
    print("=" * 70)

    save_outputs(
        args.outdir,
        train_clean,
        train_tremor,
        train_combined,
        test_clean,
        test_tremor,
        test_combined,
        participant_split,
        tremor_metadata_all,
        cfg,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("TREMORSHIELD PIPELINE SUMMARY")
    print("=" * 70)

    print(
        f"Original input rows       : "
        f"{len(df_original):,}"
    )

    print(
        f"Idle-filled input rows   : "
        f"{len(df_idle_filled):,}"
    )

    print(
        f"Stationary rows added    : "
        f"{rows_added:,}"
    )

    print(
        f"Participants             : "
        f"{len(train_users)} train / "
        f"{len(test_users)} test "
        f"(of "
        f"{len(train_users) + len(test_users)} total)"
    )

    print(
        f"Training trials          : "
        f"{len(train_meta)} "
        f"("
        f"{(train_meta.tremor_status == 1).sum()} tremor, "
        f"{(train_meta.tremor_status == 0).sum()} clean)"
    )

    print(
        f"Test trials (distinct)   : "
        f"{test_meta[TRIAL_KEYS].drop_duplicates().shape[0]}"
    )

    print(
        f"Test metadata rows       : "
        f"{len(test_meta)} "
        f"(TEST-A clean + TEST-B tremor)"
    )

    print(
        f"train_combined           : "
        f"{len(train_combined):,} rows"
    )

    print(
        f"test_combined            : "
        f"{len(test_combined):,} rows"
    )

    print(
        f"Outputs                  : "
        f"{os.path.abspath(args.outdir)}"
    )

    print(
        "\nSource file was not modified."
    )

    print(
        "Idle reconstruction was completed "
        "before participant splitting and "
        "tremor injection."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()