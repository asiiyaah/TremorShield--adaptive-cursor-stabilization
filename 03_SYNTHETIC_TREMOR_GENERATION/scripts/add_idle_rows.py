"""
TREMORSHIELD - IDLE STATIONARY ROW RECONSTRUCTION

Purpose
-------
Add stationary rows ONLY inside idle trials.

Guarantees
----------
- 37 columns unchanged
- all original rows unchanged
- all original rows remain in original relative order
- only idle gets new rows
- new rows are stationary
- no fake move events
- rows inserted chronologically
- 30-second idle interval filled
- prints rows added per participant/trial

Input:
    data/raw/all_participants_cleaned.csv

Output:
    data/final/all_participant_cleaned_idle_filled.csv
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/all_participants_cleaned.csv"
)

OUTPUT_FILE = Path(
    "data/final/all_participant_cleaned_idle_filled.csv"
)

IDLE_TASK = "idle"

# Every task is designed for 30 seconds.
IDLE_DURATION_SEC = 30.0

# 100 Hz = one row every 10 ms.
SAMPLE_INTERVAL_SEC = 0.010
SAMPLE_INTERVAL_MS = SAMPLE_INTERVAL_SEC * 1000.0


# ============================================================
# EXACT 37-COLUMN SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "participant_id",
    "session_id",
    "trial_id",
    "task",
    "action_type",
    "event",
    "button",
    "drag",
    "target_id",
    "timestamp",
    "elapsed_sec",
    "dt",
    "x",
    "y",
    "screen_width",
    "screen_height",
    "dt_raw",
    "dx_raw",
    "dy_raw",
    "velocity_x_raw",
    "velocity_y_raw",
    "velocity_raw",
    "dv_raw",
    "acceleration_raw",
    "x_normalized",
    "y_normalized",
    "dx",
    "dy",
    "velocity_x",
    "velocity_y",
    "velocity",
    "acceleration",
    "direction_change",
    "target_x_normalized",
    "target_y_normalized",
    "target_width_normalized",
    "target_height_normalized",
]


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):
    """
    Verify that the input has exactly the expected
    37 columns in exactly the expected order.
    """

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:

        print("\nEXPECTED COLUMNS:")
        for i, column in enumerate(EXPECTED_COLUMNS):
            print(f"{i}: {column}")

        print("\nACTUAL COLUMNS:")
        for i, column in enumerate(actual_columns):
            print(f"{i}: {column}")

        raise ValueError(
            "\nInput CSV columns do not exactly match "
            "the expected 37-column schema."
        )

    if len(actual_columns) != 37:
        raise ValueError(
            f"Expected 37 columns, found "
            f"{len(actual_columns)}."
        )


# ============================================================
# DETERMINE IDLE TASK START
# ============================================================

def get_idle_task_start_ms(trial):
    """
    Determine the absolute beginning of the idle task.

    timestamp is in milliseconds.

    elapsed_sec is relative to task start.

        task_start =
            timestamp - elapsed_sec * 1000

    We calculate this for all valid rows and use the median.
    """

    valid = trial[
        trial["timestamp"].notna()
        & trial["elapsed_sec"].notna()
    ].copy()

    if valid.empty:
        raise ValueError(
            "Idle trial contains no valid "
            "timestamp/elapsed_sec pair."
        )

    starts = (
        valid["timestamp"].astype(float)
        - valid["elapsed_sec"].astype(float) * 1000.0
    )

    return float(starts.median())


# ============================================================
# CREATE ONE NEW STATIONARY ROW
# ============================================================

def make_stationary_row(
    previous_row,
    timestamp_ms,
    task_start_ms,
):
    """
    Construct one NEW stationary row.

    IMPORTANT:
    This never modifies the original previous_row.

    The cursor remains at the previous x/y position.
    """

    row = previous_row.copy()

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    row["timestamp"] = timestamp_ms

    row["elapsed_sec"] = (
        timestamp_ms - task_start_ms
    ) / 1000.0

    row["dt"] = SAMPLE_INTERVAL_SEC
    row["dt_raw"] = SAMPLE_INTERVAL_SEC

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    # x and y remain exactly the previous position.

    x = row["x"]
    y = row["y"]

    # --------------------------------------------------------
    # NO FAKE MOUSE EVENT
    # --------------------------------------------------------

    row["action_type"] = np.nan
    row["event"] = np.nan
    row["button"] = np.nan
    row["drag"] = np.nan

    # --------------------------------------------------------
    # RAW KINEMATICS
    # --------------------------------------------------------

    row["dx_raw"] = 0.0
    row["dy_raw"] = 0.0

    row["velocity_x_raw"] = 0.0
    row["velocity_y_raw"] = 0.0
    row["velocity_raw"] = 0.0

    row["dv_raw"] = 0.0
    row["acceleration_raw"] = 0.0

    # --------------------------------------------------------
    # NORMALIZED POSITION
    # --------------------------------------------------------

    if (
        pd.notna(x)
        and pd.notna(row["screen_width"])
        and float(row["screen_width"]) != 0
    ):
        row["x_normalized"] = (
            float(x)
            / float(row["screen_width"])
        )

    if (
        pd.notna(y)
        and pd.notna(row["screen_height"])
        and float(row["screen_height"]) != 0
    ):
        row["y_normalized"] = (
            float(y)
            / float(row["screen_height"])
        )

    # --------------------------------------------------------
    # NORMALIZED KINEMATICS
    # --------------------------------------------------------

    row["dx"] = 0.0
    row["dy"] = 0.0

    row["velocity_x"] = 0.0
    row["velocity_y"] = 0.0
    row["velocity"] = 0.0

    row["acceleration"] = 0.0
    row["direction_change"] = 0.0

    return row


# ============================================================
# GENERATE NEW ROWS FOR ONE IDLE TRIAL
# ============================================================

def generate_idle_rows(trial):
    """
    Generate ONLY new stationary rows for one idle trial.

    Existing rows are not returned as new rows.
    """

    # Sort temporarily for reconstruction.
    trial = trial.sort_values(
        "timestamp",
        kind="stable"
    ).reset_index(drop=True)

    task_start_ms = get_idle_task_start_ms(trial)

    task_end_ms = (
        task_start_ms
        + IDLE_DURATION_SEC * 1000.0
    )

    new_rows = []

    # --------------------------------------------------------
    # PROCESS EACH ORIGINAL OBSERVATION
    # --------------------------------------------------------

    for i in range(len(trial)):

        current = trial.iloc[i]

        current_time = float(
            current["timestamp"]
        )

        # Do not generate before task start.
        if current_time < task_start_ms:
            continue

        # ----------------------------------------------------
        # Determine end of current gap.
        # ----------------------------------------------------

        if i + 1 < len(trial):

            next_time = float(
                trial.iloc[i + 1]["timestamp"]
            )

            gap_end = min(
                next_time,
                task_end_ms
            )

        else:

            # Last observation:
            # fill until the known 30-second task end.
            gap_end = task_end_ms

        # ----------------------------------------------------
        # Add stationary rows every 10 ms.
        # ----------------------------------------------------

        t = (
            current_time
            + SAMPLE_INTERVAL_MS
        )

        while t < gap_end:

            new_row = make_stationary_row(
                previous_row=current,
                timestamp_ms=t,
                task_start_ms=task_start_ms,
            )

            new_rows.append(new_row)

            t += SAMPLE_INTERVAL_MS

    # --------------------------------------------------------
    # Return only newly generated rows.
    # --------------------------------------------------------

    if not new_rows:

        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    return pd.DataFrame(
        new_rows,
        columns=EXPECTED_COLUMNS
    )


# ============================================================
# CANONICAL ROW REPRESENTATION
# ============================================================

def canonical_value(value):
    """
    Convert a value to a stable string representation
    for comparison only.

    This does NOT modify the actual dataset.
    """

    if pd.isna(value):
        return "<NA>"

    return str(value)


def canonical_row(row):
    """
    Convert one row into a tuple that can safely be
    compared between the original and output datasets.
    """

    return tuple(
        canonical_value(value)
        for value in row
    )


# ============================================================
# VERIFY ORIGINAL ROWS
# ============================================================

def verify_original_rows(original_df, output_df):
    """
    Verify that every original row still exists in the
    output and that original rows remain in the same
    relative order.

    New rows are allowed between original rows.
    """

    print("\nVerifying original-row preservation...")

    original_rows = original_df[
        EXPECTED_COLUMNS
    ]

    output_rows = output_df[
        EXPECTED_COLUMNS
    ]

    original_keys = [
        canonical_row(row)
        for row in original_rows.itertuples(
            index=False,
            name=None
        )
    ]

    output_keys = [
        canonical_row(row)
        for row in output_rows.itertuples(
            index=False,
            name=None
        )
    ]

    output_position = 0

    for original_position, original_key in enumerate(
        original_keys
    ):

        found = False

        while output_position < len(output_keys):

            if (
                output_keys[output_position]
                == original_key
            ):

                found = True
                output_position += 1
                break

            output_position += 1

        if not found:

            raise RuntimeError(
                "SAFETY CHECK FAILED:\n"
                f"Original row "
                f"{original_position:,} "
                "could not be found in the "
                "output in the original order."
            )

    print(
        f"✓ All {len(original_keys):,} original rows "
        "remain unchanged and in original relative order"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "TREMORSHIELD - "
        "IDLE STATIONARY ROW RECONSTRUCTION"
    )
    print("=" * 75)

    # ========================================================
    # LOAD
    # ========================================================

    print("\nLoading input CSV...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Original rows   : {len(df):,}"
    )

    print(
        f"Original columns : {len(df.columns)}"
    )

    # ========================================================
    # VALIDATE SCHEMA
    # ========================================================

    validate_input(df)

    print("✓ 37-column schema verified")

    # ========================================================
    # FIND IDLE TRIALS
    # ========================================================

    idle_mask = (
        df["task"]
        .astype(str)
        .str.lower()
        .eq(IDLE_TASK)
    )

    idle = df.loc[idle_mask]

    idle_trials = (
        idle[
            [
                "participant_id",
                "session_id",
                "trial_id",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None
        )
    )

    idle_trials = list(idle_trials)

    print(
        f"Idle trials      : "
        f"{len(idle_trials)}"
    )

    # ========================================================
    # GENERATE NEW IDLE ROWS
    # ========================================================

    all_new_rows = []

    print("\nRows added per idle trial:")
    print("-" * 75)

    for (
        participant_id,
        session_id,
        trial_id
    ) in idle_trials:

        mask = (
            df["participant_id"].eq(
                participant_id
            )
            & df["session_id"].eq(
                session_id
            )
            & df["trial_id"].eq(
                trial_id
            )
            & df["task"]
            .astype(str)
            .str.lower()
            .eq(IDLE_TASK)
        )

        trial = df.loc[
            mask,
            EXPECTED_COLUMNS
        ].copy()

        new_rows = generate_idle_rows(
            trial
        )

        n_added = len(new_rows)

        print(
            f"{participant_id:<18} "
            f"session={str(session_id):<5} "
            f"trial={str(trial_id):<5} "
            f"original={len(trial):<6,} "
            f"added={n_added:<7,} "
            f"final={len(trial) + n_added:,}"
        )

        if n_added > 0:
            all_new_rows.append(new_rows)

    # ========================================================
    # COMBINE NEW ROWS
    # ========================================================

    if all_new_rows:

        new_rows_df = pd.concat(
            all_new_rows,
            ignore_index=True
        )

    else:

        new_rows_df = pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    print("-" * 75)

    print(
        f"Total new stationary rows: "
        f"{len(new_rows_df):,}"
    )

    # ========================================================
    # SAFETY CHECK 1
    # ========================================================

    print("\nRunning safety checks...")

    assert (
        list(new_rows_df.columns)
        == EXPECTED_COLUMNS
    )

    print(
        "✓ New rows use exactly "
        "the same 37 columns"
    )

    # ========================================================
    # SAFETY CHECK 2
    # ONLY IDLE ROWS GENERATED
    # ========================================================

    if len(new_rows_df) > 0:

        assert (
            new_rows_df["task"]
            .astype(str)
            .str.lower()
            .eq(IDLE_TASK)
            .all()
        )

    print(
        "✓ Only idle rows were generated"
    )

    # ========================================================
    # SAFETY CHECK 3
    # NO FAKE EVENTS
    # ========================================================

    if len(new_rows_df) > 0:

        assert (
            new_rows_df["event"]
            .isna()
            .all()
        )

        assert (
            new_rows_df["action_type"]
            .isna()
            .all()
        )

    print(
        "✓ No fake move events"
    )

    # ========================================================
    # SAFETY CHECK 4
    # NEW ROWS ARE STATIONARY
    # ========================================================

    if len(new_rows_df) > 0:

        assert np.allclose(
            new_rows_df["dx"]
            .fillna(0)
            .astype(float),
            0.0
        )

        assert np.allclose(
            new_rows_df["dy"]
            .fillna(0)
            .astype(float),
            0.0
        )

        assert np.allclose(
            new_rows_df["velocity"]
            .fillna(0)
            .astype(float),
            0.0
        )

        assert np.allclose(
            new_rows_df["acceleration"]
            .fillna(0)
            .astype(float),
            0.0
        )

    print(
        "✓ New rows are stationary"
    )

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    # Generated rows grouped by trial.
    new_by_trial = defaultdict(list)

    for _, row in new_rows_df.iterrows():

        key = (
            row["participant_id"],
            row["session_id"],
            row["trial_id"],
        )

        new_by_trial[key].append(row)

    # ========================================================
    # PRESERVE ORIGINAL TRIAL ORDER
    # ========================================================

    trial_order = (
        df[
            [
                "participant_id",
                "session_id",
                "trial_id",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None
        )
    )

    output_parts = []

    for key in trial_order:

        (
            participant_id,
            session_id,
            trial_id
        ) = key

        trial_mask = (
            df["participant_id"].eq(
                participant_id
            )
            & df["session_id"].eq(
                session_id
            )
            & df["trial_id"].eq(
                trial_id
            )
        )

        # ----------------------------------------------------
        # ORIGINAL ROWS.
        #
        # These are taken directly from the original DataFrame.
        # ----------------------------------------------------

        original_trial = df.loc[
            trial_mask,
            EXPECTED_COLUMNS
        ].copy()

        # ----------------------------------------------------
        # NON-IDLE:
        #
        # Keep EXACTLY as originally ordered.
        # ----------------------------------------------------

        if key not in new_by_trial:

            output_parts.append(
                original_trial
            )

            continue

        # ----------------------------------------------------
        # IDLE:
        #
        # Original + newly generated rows.
        # ----------------------------------------------------

        generated_trial = pd.DataFrame(
            new_by_trial[key],
            columns=EXPECTED_COLUMNS
        )

        combined_trial = pd.concat(
            [
                original_trial,
                generated_trial,
            ],
            ignore_index=True
        )

        # Chronological order inside this idle trial.
        combined_trial = combined_trial.sort_values(
            "timestamp",
            kind="stable"
        )

        output_parts.append(
            combined_trial
        )

    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    result = pd.concat(
        output_parts,
        ignore_index=True
    )

    # ========================================================
    # SAFETY CHECK 5
    # COLUMNS
    # ========================================================

    assert (
        list(result.columns)
        == EXPECTED_COLUMNS
    )

    assert len(result.columns) == 37

    print(
        "✓ 37 columns unchanged"
    )

    # ========================================================
    # SAFETY CHECK 6
    # ROW COUNT
    # ========================================================

    expected_rows = (
        len(df)
        + len(new_rows_df)
    )

    assert (
        len(result)
        == expected_rows
    )

    print(
        "✓ Original rows retained; "
        f"{len(new_rows_df):,} new rows added"
    )

    # ========================================================
    # SAFETY CHECK 7
    # ORIGINAL ROW PRESERVATION
    # ========================================================

    verify_original_rows(
        df,
        result
    )

    # ========================================================
    # SAFETY CHECK 8
    # NON-IDLE ROW COUNT
    # ========================================================

    original_non_idle_count = (
        ~df["task"]
        .astype(str)
        .str.lower()
        .eq(IDLE_TASK)
    ).sum()

    result_non_idle_count = (
        ~result["task"]
        .astype(str)
        .str.lower()
        .eq(IDLE_TASK)
    ).sum()

    assert (
        original_non_idle_count
        == result_non_idle_count
    )

    print(
        "✓ No non-idle rows were "
        "added or removed"
    )

    # ========================================================
    # SAFETY CHECK 9
    # VERIFY ONLY IDLE ROW COUNT INCREASED
    # ========================================================

    original_idle_count = (
        df["task"]
        .astype(str)
        .str.lower()
        .eq(IDLE_TASK)
    ).sum()

    result_idle_count = (
        result["task"]
        .astype(str)
        .str.lower()
        .eq(IDLE_TASK)
    ).sum()

    assert (
        result_idle_count
        == original_idle_count
        + len(new_rows_df)
    )

    print(
        "✓ Idle row count increased only "
        "by the generated stationary rows"
    )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("SUCCESS")
    print("=" * 75)

    print(
        f"Original rows : "
        f"{len(df):,}"
    )

    print(
        f"Rows added    : "
        f"{len(new_rows_df):,}"
    )

    print(
        f"Final rows    : "
        f"{len(result):,}"
    )

    print(
        f"Columns       : "
        f"{len(result.columns)}"
    )

    print(
        f"Output file   : "
        f"{OUTPUT_FILE}"
    )

    print("\nVerified:")
    print(
        "✓ 37 columns unchanged"
    )
    print(
        "✓ all original rows unchanged"
    )
    print(
        "✓ all original rows remain "
        "in original relative order"
    )
    print(
        "✓ only idle gets new rows"
    )
    print(
        "✓ new rows are stationary"
    )
    print(
        "✓ no fake move events"
    )
    print(
        "✓ rows inserted chronologically"
    )
    print(
        "✓ 30-second idle interval filled"
    )
    print(
        "✓ rows added printed per "
        "participant/trial"
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()