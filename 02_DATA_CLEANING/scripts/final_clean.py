import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 0. PROJECT PATHS
# ============================================================

# Project root = folder containing data/ and scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ORIGINAL DATA
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# FINAL folders
CLEANED_DIR = PROJECT_ROOT / "data" / "final clean"
COMBINED_DIR = PROJECT_ROOT / "data" / "final combine"

# Create folders automatically
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_FILE = (
    COMBINED_DIR / "all_participants_cleaned.csv"
)


# ============================================================
# 1. COLUMN DEFINITIONS
# ============================================================

TIME_COL = "timestamp"
X_COL = "x"
Y_COL = "y"

SCREEN_W = "screen_width"
SCREEN_H = "screen_height"

# These are important for keeping trials separate
GROUP_COLS = [
    "session_id",
    "trial_id"
]


# ============================================================
# 2. FIND ALL RAW CSV FILES
# ============================================================

input_files = sorted(
    RAW_DIR.glob("*.csv")
)

print("=" * 75)
print("TREMOR DATASET - CLEANING PIPELINE")
print("=" * 75)

print(f"\nRaw folder:")
print(RAW_DIR)

print(f"\nCSV files found: {len(input_files)}")

if len(input_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in {RAW_DIR}"
    )


# ============================================================
# 3. PARTICIPANT ANONYMIZATION
# ============================================================

# Each raw CSV represents one participant.
#
# Example:
#
# AISWARYA C_s01_ALL_TASKS.csv
#             ↓
# Participant_01
#
# The original participant name will NOT appear
# in the cleaned dataset.

participant_mapping = {}

for number, file in enumerate(input_files, start=1):

    participant_mapping[file.name] = (
        f"Participant_{number:02d}"
    )


# ============================================================
# FUNCTION TO CLEAN ONE PARTICIPANT
# ============================================================

def clean_participant_file(
    input_file,
    participant_id
):

    print("\n" + "-" * 75)
    print(f"Processing: {input_file.name}")
    print(f"Anonymous ID: {participant_id}")
    print("-" * 75)


    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(input_file)

    original_rows = len(df)

    print(
        f"Original rows: {original_rows}"
    )


    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        TIME_COL,
        X_COL,
        Y_COL,
        SCREEN_W,
        SCREEN_H,
        "session_id",
        "trial_id"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        print(
            "\nWARNING: File skipped."
        )

        print(
            "Missing columns:",
            missing
        )

        return None


    # ========================================================
    # ANONYMIZE PARTICIPANT
    # ========================================================

    # Remove original user_id completely.

    if "user_id" in df.columns:

        df = df.drop(
            columns=["user_id"]
        )

    # Add anonymous participant ID

    df.insert(
        0,
        "participant_id",
        participant_id
    )


    # ========================================================
    # CONVERT NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [
        TIME_COL,
        X_COL,
        Y_COL,
        SCREEN_W,
        SCREEN_H,
        "target_x",
        "target_y",
        "target_width",
        "target_height"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


    # ========================================================
    # 1. TIMESTAMP ORDERING
    # ========================================================

    # Order samples separately within each
    # session and trial.

    df = df.sort_values(
        by=GROUP_COLS + [TIME_COL]
    ).reset_index(drop=True)

    print(
        "✓ Timestamp ordering completed"
    )


    # ========================================================
    # 2. REMOVE EXACT DUPLICATE ROWS
    # ========================================================

    before = len(df)

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    print(
        f"✓ Exact duplicate rows removed: "
        f"{before - len(df)}"
    )


    # ========================================================
    # 3. INVALID VALUE CLEANING
    # ========================================================

    before = len(df)


    # --------------------------------------------------------
    # Remove missing essential values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            TIME_COL,
            X_COL,
            Y_COL,
            SCREEN_W,
            SCREEN_H,
            "session_id",
            "trial_id"
        ]
    )


    # --------------------------------------------------------
    # Remove infinite values
    # --------------------------------------------------------

    finite_mask = (
        np.isfinite(df[TIME_COL]) &
        np.isfinite(df[X_COL]) &
        np.isfinite(df[Y_COL]) &
        np.isfinite(df[SCREEN_W]) &
        np.isfinite(df[SCREEN_H])
    )

    df = df[finite_mask]


    # --------------------------------------------------------
    # Screen dimensions must be positive
    # --------------------------------------------------------

    df = df[
        (df[SCREEN_W] > 0) &
        (df[SCREEN_H] > 0)
    ]


    # --------------------------------------------------------
    # Coordinates cannot be negative
    # --------------------------------------------------------

    df = df[
        (df[X_COL] >= 0) &
        (df[Y_COL] >= 0)
    ]


    # --------------------------------------------------------
    # Coordinates cannot exceed screen boundaries
    # --------------------------------------------------------

    df = df[
        (df[X_COL] < df[SCREEN_W]) &
        (df[Y_COL] < df[SCREEN_H])
    ]


    df = df.reset_index(drop=True)


    print(
        f"✓ Invalid rows removed: "
        f"{before - len(df)}"
    )


    # ========================================================
    # RE-ORDER AFTER CLEANING
    # ========================================================

    df = df.sort_values(
        by=GROUP_COLS + [TIME_COL]
    ).reset_index(drop=True)


    # ========================================================
    # 4. FIRST DERIVED FEATURE CALCULATION
    #    FROM CLEANED RAW TRAJECTORY
    # ========================================================

    print(
        "✓ Calculating raw trajectory features..."
    )


    # --------------------------------------------------------
    # Calculate dt_raw
    # --------------------------------------------------------

    df["dt_raw"] = (
        df.groupby(GROUP_COLS)[TIME_COL]
          .diff()
    )


    # --------------------------------------------------------
    # Calculate elapsed_sec
    # --------------------------------------------------------

    # Elapsed time starts at 0 for each
    # session + trial and is based on the
    # timestamp after timestamp ordering.

    df["elapsed_sec"] = (
        df.groupby(GROUP_COLS)[TIME_COL]
          .transform(
              lambda x: x - x.iloc[0]
          )
    )


    # --------------------------------------------------------
    # Calculate dx and dy
    # --------------------------------------------------------

    df["dx_raw"] = (
        df.groupby(GROUP_COLS)[X_COL]
          .diff()
    )

    df["dy_raw"] = (
        df.groupby(GROUP_COLS)[Y_COL]
          .diff()
    )


    # --------------------------------------------------------
    # Calculate raw velocity
    # --------------------------------------------------------

    valid_dt = df["dt_raw"] > 0


    df["velocity_x_raw"] = np.nan
    df["velocity_y_raw"] = np.nan


    df.loc[
        valid_dt,
        "velocity_x_raw"
    ] = (
        df.loc[
            valid_dt,
            "dx_raw"
        ]
        /
        df.loc[
            valid_dt,
            "dt_raw"
        ]
    )


    df.loc[
        valid_dt,
        "velocity_y_raw"
    ] = (
        df.loc[
            valid_dt,
            "dy_raw"
        ]
        /
        df.loc[
            valid_dt,
            "dt_raw"
        ]
    )


    # Resultant velocity

    df["velocity_raw"] = np.sqrt(
        df["velocity_x_raw"] ** 2
        +
        df["velocity_y_raw"] ** 2
    )


    # --------------------------------------------------------
    # Calculate raw acceleration
    # --------------------------------------------------------

    df["dv_raw"] = (
        df.groupby(GROUP_COLS)[
            "velocity_raw"
        ].diff()
    )


    df["acceleration_raw"] = np.nan


    valid_acc = (
        (df["dt_raw"] > 0) &
        df["dv_raw"].notna()
    )


    df.loc[
        valid_acc,
        "acceleration_raw"
    ] = (
        df.loc[
            valid_acc,
            "dv_raw"
        ]
        /
        df.loc[
            valid_acc,
            "dt_raw"
        ]
    )


    print(
        "✓ Raw velocity calculated"
    )

    print(
        "✓ Raw acceleration calculated"
    )

    print(
        "✓ Elapsed time calculated"
    )


    # ========================================================
    # 5. RESOLUTION NORMALIZATION
    # ========================================================

    print(
        "✓ Performing resolution normalization..."
    )


    # --------------------------------------------------------
    # Normalize trajectory coordinates
    # --------------------------------------------------------

    df["x_normalized"] = (
        df[X_COL] /
        df[SCREEN_W]
    )

    df["y_normalized"] = (
        df[Y_COL] /
        df[SCREEN_H]
    )


    # --------------------------------------------------------
    # Normalize target coordinates
    # --------------------------------------------------------

    if "target_x" in df.columns:

        df["target_x_normalized"] = (
            df["target_x"] /
            df[SCREEN_W]
        )


    if "target_y" in df.columns:

        df["target_y_normalized"] = (
            df["target_y"] /
            df[SCREEN_H]
        )


    if "target_width" in df.columns:

        df["target_width_normalized"] = (
            df["target_width"] /
            df[SCREEN_W]
        )


    if "target_height" in df.columns:

        df["target_height_normalized"] = (
            df["target_height"] /
            df[SCREEN_H]
        )


    # ========================================================
    # 6. RECALCULATE FEATURES AFTER NORMALIZATION
    # ========================================================

    print(
        "✓ Recalculating features after normalization..."
    )


    # --------------------------------------------------------
    # Calculate dt again
    # --------------------------------------------------------

    df["dt"] = (
        df.groupby(GROUP_COLS)[TIME_COL]
          .diff()
    )


    # --------------------------------------------------------
    # Calculate normalized dx/dy
    # --------------------------------------------------------

    df["dx"] = (
        df.groupby(GROUP_COLS)[
            "x_normalized"
        ].diff()
    )

    df["dy"] = (
        df.groupby(GROUP_COLS)[
            "y_normalized"
        ].diff()
    )


    # --------------------------------------------------------
    # Calculate final velocity
    # --------------------------------------------------------

    valid_dt = df["dt"] > 0


    df["velocity_x"] = np.nan
    df["velocity_y"] = np.nan


    df.loc[
        valid_dt,
        "velocity_x"
    ] = (
        df.loc[
            valid_dt,
            "dx"
        ]
        /
        df.loc[
            valid_dt,
            "dt"
        ]
    )


    df.loc[
        valid_dt,
        "velocity_y"
    ] = (
        df.loc[
            valid_dt,
            "dy"
        ]
        /
        df.loc[
            valid_dt,
            "dt"
        ]
    )


    df["velocity"] = np.sqrt(
        df["velocity_x"] ** 2
        +
        df["velocity_y"] ** 2
    )


    # --------------------------------------------------------
    # Calculate final acceleration
    # --------------------------------------------------------

    df["dv"] = (
        df.groupby(GROUP_COLS)[
            "velocity"
        ].diff()
    )


    df["acceleration"] = np.nan


    valid_acc = (
        (df["dt"] > 0) &
        df["dv"].notna()
    )


    df.loc[
        valid_acc,
        "acceleration"
    ] = (
        df.loc[
            valid_acc,
            "dv"
        ]
        /
        df.loc[
            valid_acc,
            "dt"
        ]
    )


    # ========================================================
    # 7. RECALCULATE DIRECTION CHANGE
    # ========================================================

    df["angle"] = np.arctan2(
        df["dy"],
        df["dx"]
    )


    df["direction_change"] = (
        df.groupby(GROUP_COLS)[
            "angle"
        ].diff()
    )


    # Wrap angle to [-pi, pi]

    df["direction_change"] = (
        (
            df["direction_change"]
            + np.pi
        )
        %
        (2 * np.pi)
    ) - np.pi


    print(
        "✓ Final velocity calculated"
    )

    print(
        "✓ Final acceleration calculated"
    )

    print(
        "✓ Final direction change calculated"
    )


    # ========================================================
    # 8. FINAL COLUMN SELECTION
    # ========================================================

    # Existing final dataset columns are retained.
    #
    # ADDITIONALLY KEEP:
    #
    #   Raw coordinates:
    #       x
    #       y
    #
    #   Screen dimensions:
    #       screen_width
    #       screen_height
    #
    #   Time:
    #       timestamp
    #       elapsed_sec
    #       dt_raw
    #       dt
    #
    #   Raw/pre-normalization trajectory:
    #       dx_raw
    #       dy_raw
    #
    #   Raw/pre-normalization velocity:
    #       velocity_x_raw
    #       velocity_y_raw
    #       velocity_raw
    #
    #   Raw/pre-normalization acceleration:
    #       dv_raw
    #       acceleration_raw
    #
    # Existing normalized features are also retained.


    final_columns = [

        # ------------------------------------
        # Participant identification
        # ------------------------------------

        "participant_id",


        # ------------------------------------
        # Session / trial identification
        # ------------------------------------

        "session_id",
        "trial_id",


        # ------------------------------------
        # Task information
        # ------------------------------------

        "task",
        "action_type",
        "event",
        "button",
        "drag",
        "target_id",


        # ------------------------------------
        # Time
        # ------------------------------------

        "timestamp",
        "elapsed_sec",
        "dt",


        # ------------------------------------
        # RAW COORDINATES
        # ------------------------------------

        "x",
        "y",


        # ------------------------------------
        # SCREEN DIMENSIONS
        # ------------------------------------

        "screen_width",
        "screen_height",


        # ------------------------------------
        # RAW / PRE-NORMALIZATION FEATURES
        # ------------------------------------

        "dt_raw",
        "dx_raw",
        "dy_raw",

        "velocity_x_raw",
        "velocity_y_raw",
        "velocity_raw",

        "dv_raw",
        "acceleration_raw",


        # ------------------------------------
        # Normalized trajectory
        # ------------------------------------

        "x_normalized",
        "y_normalized",


        # ------------------------------------
        # Final displacement
        # ------------------------------------

        "dx",
        "dy",


        # ------------------------------------
        # Final velocity
        # ------------------------------------

        "velocity_x",
        "velocity_y",
        "velocity",


        # ------------------------------------
        # Final acceleration
        # ------------------------------------

        "acceleration",


        # ------------------------------------
        # Final direction
        # ------------------------------------

        "direction_change",


        # ------------------------------------
        # Normalized target information
        # ------------------------------------

        "target_x_normalized",
        "target_y_normalized",
        "target_width_normalized",
        "target_height_normalized"
    ]


    # Only keep columns that actually exist

    final_columns = [
        col
        for col in final_columns
        if col in df.columns
    ]


    final_df = df[
        final_columns
    ].copy()


    # ========================================================
    # 9. FINAL VALIDATION
    # ========================================================

    # Check normalized coordinates

    if "x_normalized" in final_df.columns:

        if not (
            final_df["x_normalized"].between(
                0, 1,
                inclusive="both"
            ).all()
        ):

            print(
                "WARNING: x_normalized "
                "contains values outside [0,1]"
            )


    if "y_normalized" in final_df.columns:

        if not (
            final_df["y_normalized"].between(
                0, 1,
                inclusive="both"
            ).all()
        ):

            print(
                "WARNING: y_normalized "
                "contains values outside [0,1]"
            )


    # Check final required columns

    print(
        f"Final rows: {len(final_df)}"
    )

    print(
        f"Final columns: {len(final_df.columns)}"
    )


    # ========================================================
    # 10. SAVE INDIVIDUAL CLEANED FILE
    # ========================================================

    output_file = (
        CLEANED_DIR /
        f"{participant_id}_cleaned.csv"
    )


    final_df.to_csv(
        output_file,
        index=False
    )


    print(
        f"✓ Saved: {output_file.name}"
    )


    return final_df


# ============================================================
# 11. PROCESS ALL PARTICIPANTS
# ============================================================

all_cleaned = []

successful = 0
failed = 0


for input_file in input_files:

    participant_id = (
        participant_mapping[
            input_file.name
        ]
    )

    try:

        cleaned_df = clean_participant_file(
            input_file,
            participant_id
        )

        if cleaned_df is not None:

            all_cleaned.append(
                cleaned_df
            )

            successful += 1

    except Exception as e:

        failed += 1

        print(
            "\nERROR:"
        )

        print(
            input_file.name
        )

        print(
            str(e)
        )


# ============================================================
# 12. COMBINE ALL CLEANED FILES
# ============================================================

print("\n" + "=" * 75)
print("COMBINING ALL CLEANED PARTICIPANTS")
print("=" * 75)


if len(all_cleaned) == 0:

    raise RuntimeError(
        "No files were successfully cleaned."
    )


combined_df = pd.concat(
    all_cleaned,
    ignore_index=True
)


# ============================================================
# 13. SORT COMBINED DATASET
# ============================================================

combined_df = combined_df.sort_values(
    by=[
        "participant_id",
        "session_id",
        "trial_id",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# 14. FINAL COMBINED AUDIT
# ============================================================

print(
    f"\nParticipants successfully processed: "
    f"{successful}"
)

print(
    f"Participants failed: {failed}"
)

print(
    f"Total final rows: "
    f"{len(combined_df)}"
)

print(
    f"Total final columns: "
    f"{len(combined_df.columns)}"
)


print(
    "\nFinal columns:"
)

for col in combined_df.columns:

    print(
        f"  {col}"
    )


print(
    "\nParticipant distribution:"
)

print(
    combined_df[
        "participant_id"
    ].value_counts().sort_index()
)


# ============================================================
# 15. SAVE COMBINED DATASET
# ============================================================

combined_df.to_csv(
    COMBINED_FILE,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)
print("CLEANING COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    f"Individual cleaned files : "
    f"{successful}"
)

print(
    f"Combined file             : "
    f"{COMBINED_FILE}"
)

print("=" * 75)
