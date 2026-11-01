

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# CONFIGURATION


WINDOW_SIZE_SEC = 0.500
STRIDE_SEC = 0.100

MIN_EVENTS = 5
MIN_WINDOW_DURATION_SEC = 0.200

TRIAL_KEYS = [
    "participant_id",
    "session_id",
    "trial_id",
]


# FINAL LABEL MAPPING

TASK_TO_FINAL_LABEL = {
    # Navigation
    "normal": "navigation",
    "fast": "navigation",
    "slow": "navigation",

    # Clicking
    "click": "clicking",
    "double_click": "clicking",

    "drag": "dragging",

    "precision": "target_selection",

    "target_selection": "target_selection",

    "idle": "idle",
}


FINAL_LABELS = [
    "navigation",
    "clicking",
    "dragging",
    "precision",
    "target_selection",
    "idle"
]


# REQUIRED SOURCE COLUMNS

REQUIRED_COLUMNS = [
    "participant_id",
    "session_id",
    "trial_id",
    "task",
    "timestamp",
    "event",
]


# OBSERVED FEATURE COLUMNS

OBSERVED_KINEMATIC_COLUMNS = [
    "dx_observed",
    "dy_observed",
    "vx_observed",
    "vy_observed",
    "velocity_observed",
    "acceleration_observed",
    "direction_change_observed",
]


# TIMESTAMP HANDLING

def add_elapsed_seconds(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create elapsed_sec relative to the beginning of one trial.

    Handles timestamps represented approximately in seconds
    or milliseconds.
    """

    df = trial_df.copy()

    timestamp = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    if timestamp.isna().all():
        raise ValueError(
            "Timestamp column contains no valid numeric values."
        )

    df["_timestamp_numeric"] = timestamp

    df = df.sort_values(
        "_timestamp_numeric",
        kind="mergesort",
    ).copy()

    t = df["_timestamp_numeric"].to_numpy(
        dtype=float
    )

    positive_dt = np.diff(t)
    positive_dt = positive_dt[positive_dt > 0]

    if len(positive_dt) == 0:
        scale = 1.0
    else:
        median_dt = float(
            np.median(positive_dt)
        )

        # ~0.016 sec -> seconds
        # ~16 ms     -> milliseconds
        scale = 1000.0 if median_dt > 1.0 else 1.0

    df["elapsed_sec"] = (
        df["_timestamp_numeric"]
        - df["_timestamp_numeric"].iloc[0]
    ) / scale

    df = df.drop(
        columns=["_timestamp_numeric"]
    )

    return df


# SAFE NUMERIC HELPERS

def numeric_array(
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:

    if column not in df.columns:
        return np.array([], dtype=float)

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    return values[np.isfinite(values)]


def safe_mean(values):
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def safe_std(values):
    if len(values) == 0:
        return np.nan
    return float(np.std(values))


def safe_median(values):
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def safe_percentile(values, percentile):
    if len(values) == 0:
        return np.nan
    return float(
        np.percentile(values, percentile)
    )


def safe_min(values):
    if len(values) == 0:
        return np.nan
    return float(np.min(values))


def safe_max(values):
    if len(values) == 0:
        return np.nan
    return float(np.max(values))


# WINDOW CREATION

def create_trial_windows(
    trial_df: pd.DataFrame,
    window_size_sec: float = WINDOW_SIZE_SEC,
    stride_sec: float = STRIDE_SEC,
    min_events: int = MIN_EVENTS,
    min_duration_sec: float = MIN_WINDOW_DURATION_SEC,
) -> Tuple[List[pd.DataFrame], List[Dict]]:

    if trial_df.empty:
        return [], []

    df = add_elapsed_seconds(
        trial_df
    )

    df = df.reset_index(
        drop=False
    ).rename(
        columns={
            "index": "_original_row_index"
        }
    )

    t = df["elapsed_sec"].to_numpy(
        dtype=float
    )

    if len(t) == 0:
        return [], []

    trial_start = float(t.min())
    trial_end = float(t.max())

    duration = trial_end - trial_start

    if duration < min_duration_sec:

        if len(df) >= min_events:

            metadata = {
                "window_start_sec": trial_start,
                "window_end_sec": trial_end,
                "window_duration_sec": duration,
                "n_events": len(df),
            }

            return [df.copy()], [metadata]

        return [], []

    windows = []
    metadata_rows = []

    starts = np.arange(
        trial_start,
        trial_end + stride_sec,
        stride_sec,
    )

    for start in starts:

        end = (
            start
            + window_size_sec
        )

        mask = (
            (df["elapsed_sec"] >= start)
            &
            (df["elapsed_sec"] < end)
        )

        window = df.loc[mask].copy()

        if window.empty:
            continue

        actual_start = float(
            window["elapsed_sec"].min()
        )

        actual_end = float(
            window["elapsed_sec"].max()
        )

        actual_duration = (
            actual_end
            - actual_start
        )

        if len(window) < min_events:
            continue

        if (
            actual_duration
            < min_duration_sec
        ):
            continue

        windows.append(window)

        metadata_rows.append(
            {
                "window_start_sec": float(
                    start
                ),
                "window_end_sec": float(
                    end
                ),
                "window_duration_sec": float(
                    actual_duration
                ),
                "n_events": int(
                    len(window)
                ),
            }
        )

    return (
        windows,
        metadata_rows,
    )


# FEATURE EXTRACTION

def extract_window_features(
    window: pd.DataFrame,
) -> Dict:

    features = {}

    # Basic temporal information

    elapsed = pd.to_numeric(
        window["elapsed_sec"],
        errors="coerce",
    )

    elapsed = elapsed[
        np.isfinite(elapsed)
    ]

    if len(elapsed) > 0:

        duration = (
            float(elapsed.max())
            - float(elapsed.min())
        )

    else:

        duration = np.nan

    features[
        "window_duration_sec"
    ] = duration

    features[
        "n_events"
    ] = int(len(window))

    # Event statistics

    if "event" in window.columns:

        events = (
            window["event"]
            .astype(str)
            .str.lower()
        )

        features[
            "n_move_events"
        ] = int(
            (events == "move").sum()
        )

        features[
            "n_press_events"
        ] = int(
            (events == "press").sum()
        )

        features[
            "n_release_events"
        ] = int(
            (events == "release").sum()
        )

        features[
            "n_double_click_events"
        ] = int(
            (
                events
                == "double_click"
            ).sum()
        )

    else:

        features[
            "n_move_events"
        ] = 0

        features[
            "n_press_events"
        ] = 0

        features[
            "n_release_events"
        ] = 0

        features[
            "n_double_click_events"
        ] = 0

    # Velocity

    velocity = numeric_array(
        window,
        "velocity_observed",
    )

    features[
        "velocity_mean"
    ] = safe_mean(velocity)

    features[
        "velocity_std"
    ] = safe_std(velocity)

    features[
        "velocity_median"
    ] = safe_median(velocity)

    features[
        "velocity_p10"
    ] = safe_percentile(
        velocity,
        10,
    )

    features[
        "velocity_p90"
    ] = safe_percentile(
        velocity,
        90,
    )

    features[
        "velocity_max"
    ] = safe_max(velocity)

    # Velocity X/Y

    vx = numeric_array(
        window,
        "vx_observed",
    )

    vy = numeric_array(
        window,
        "vy_observed",
    )

    features[
        "vx_mean"
    ] = safe_mean(vx)

    features[
        "vx_std"
    ] = safe_std(vx)

    features[
        "vy_mean"
    ] = safe_mean(vy)

    features[
        "vy_std"
    ] = safe_std(vy)

    # Displacement increments

    dx = numeric_array(
        window,
        "dx_observed",
    )

    dy = numeric_array(
        window,
        "dy_observed",
    )

    features[
        "dx_mean"
    ] = safe_mean(dx)

    features[
        "dx_std"
    ] = safe_std(dx)

    features[
        "dy_mean"
    ] = safe_mean(dy)

    features[
        "dy_std"
    ] = safe_std(dy)

    # Acceleration

    acceleration = numeric_array(
        window,
        "acceleration_observed",
    )

    features[
        "acceleration_mean"
    ] = safe_mean(acceleration)

    features[
        "acceleration_std"
    ] = safe_std(acceleration)

    features[
        "acceleration_median"
    ] = safe_median(acceleration)

    features[
        "acceleration_p90"
    ] = safe_percentile(
        acceleration,
        90,
    )

    features[
        "acceleration_max"
    ] = safe_max(acceleration)

    # Direction changes
    

    direction_change = numeric_array(
        window,
        "direction_change_observed",
    )

    features[
        "direction_change_mean"
    ] = safe_mean(direction_change)

    features[
        "direction_change_std"
    ] = safe_std(direction_change)

    features[
        "direction_change_median"
    ] = safe_median(
        direction_change
    )

    features[
        "direction_change_p90"
    ] = safe_percentile(
        direction_change,
        90,
    )

    # Path length

    if (
        len(dx) > 0
        and len(dy) > 0
    ):

        n = min(
            len(dx),
            len(dy),
        )

        step_distance = np.sqrt(
            dx[:n] ** 2
            + dy[:n] ** 2
        )

        path_length = float(
            np.sum(step_distance)
        )

    else:

        path_length = np.nan

    features[
        "path_length"
    ] = path_length

    # Net displacement

    if (
        len(dx) > 0
        and len(dy) > 0
    ):

        net_dx = float(
            np.sum(dx)
        )

        net_dy = float(
            np.sum(dy)
        )

        net_displacement = float(
            np.sqrt(
                net_dx ** 2
                + net_dy ** 2
            )
        )

    else:

        net_displacement = np.nan

    features[
        "net_displacement"
    ] = net_displacement

    # Path straightness
    #
    # 1 = approximately straight
    # smaller = more curved/irregular

    if (
        np.isfinite(path_length)
        and np.isfinite(net_displacement)
        and path_length > 0
    ):

        features[
            "path_straightness"
        ] = (
            net_displacement
            / path_length
        )

    else:

        features[
            "path_straightness"
        ] = np.nan

    # Pause ratio

    if len(velocity) > 0:

        
        positive_velocity = velocity[
            velocity > 0
        ]

        if len(positive_velocity) > 0:

            threshold = (
                0.10
                * np.median(
                    positive_velocity
                )
            )

            pause_ratio = float(
                np.mean(
                    velocity
                    <= threshold
                )
            )

        else:

            pause_ratio = 1.0

    else:

        pause_ratio = np.nan

    features[
        "pause_ratio"
    ] = pause_ratio

    # Coordinate spread
    #

    x = numeric_array(
        window,
        "observed_x",
    )

    y = numeric_array(
        window,
        "observed_y",
    )

    features[
        "x_std"
    ] = safe_std(x)

    features[
        "y_std"
    ] = safe_std(y)

    if len(x) > 0:

        features[
            "x_range"
        ] = float(
            np.max(x)
            - np.min(x)
        )

    else:

        features[
            "x_range"
        ] = np.nan

    if len(y) > 0:

        features[
            "y_range"
        ] = float(
            np.max(y)
            - np.min(y)
        )

    else:

        features[
            "y_range"
        ] = np.nan

    return features


# PROCESS ONE DATASET

def process_dataset(
    input_csv: str,
    output_dir: str,
    split_name: str,
) -> None:

    input_path = Path(input_csv)
    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 75)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 75)

    df = pd.read_csv(
        input_path
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # Required columns

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )

    # Check tasks

    tasks = (
        df["task"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    print()
    print("Original tasks:")
    print(
        tasks.value_counts(
            dropna=False
        )
    )

    unknown_tasks = sorted(
        set(tasks.unique())
        - set(TASK_TO_FINAL_LABEL.keys())
    )

    if unknown_tasks:
        raise ValueError(
            "Unknown task labels found: "
            f"{unknown_tasks}"
        )

    # Deterministic ordering

    df = df.sort_values(
        TRIAL_KEYS + ["timestamp"],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    # Window storage
    

    ml_rows = []
    event_window_rows = []
    window_index_rows = []

    total_trials = 0
    trials_with_windows = 0

    total_windows = 0
    excluded_idle_windows = 0

    grouped = df.groupby(
        TRIAL_KEYS,
        sort=False,
        dropna=False,
    )

    # TRIAL LOOP

    for trial_key, trial_df in grouped:

        total_trials += 1

        trial_df = trial_df.copy()

        task = (
            str(
                trial_df["task"].iloc[0]
            )
            .lower()
            .strip()
        )

        final_label = (
            TASK_TO_FINAL_LABEL[
                task
            ]
            if task in TASK_TO_FINAL_LABEL
            else None
        )

        windows, metadata = (
            create_trial_windows(
                trial_df
            )
        )

        if not windows:
            continue

        trials_with_windows += 1

        participant_id = (
            trial_df[
                "participant_id"
            ].iloc[0]
        )

        session_id = (
            trial_df[
                "session_id"
            ].iloc[0]
        )

        trial_id = (
            trial_df[
                "trial_id"
            ].iloc[0]
        )

        # Tremor status

        if "tremor_status" in trial_df.columns:

            tremor_status = int(
                trial_df[
                    "tremor_status"
                ].iloc[0]
            )

        elif (
            "tremor_applied"
            in trial_df.columns
        ):

            tremor_status = int(
                trial_df[
                    "tremor_applied"
                ].iloc[0]
            )

        else:

            tremor_status = np.nan

        # Window loop

        for local_idx, (
            window,
            meta,
        ) in enumerate(
            zip(
                windows,
                metadata,
            )
        ):

            window_id = (
                f"{split_name}_"
                f"{participant_id}_"
                f"{session_id}_"
                f"{trial_id}_"
                f"W{local_idx:04d}"
            )

            total_windows += 1

            # Event-level window data

            event_window = (
                window.copy()
            )

            event_window[
                "window_id"
            ] = window_id

            event_window[
                "window_task"
            ] = task

            event_window[
                "final_label"
            ] = (
                final_label
                if final_label is not None
                else "excluded"
            )

            event_window[
                "window_tremor_status"
            ] = tremor_status

            event_window[
                "window_start_sec"
            ] = meta[
                "window_start_sec"
            ]

            event_window[
                "window_end_sec"
            ] = meta[
                "window_end_sec"
            ]

            event_window[
                "window_duration_sec"
            ] = meta[
                "window_duration_sec"
            ]

            event_window[
                "window_n_events"
            ] = meta[
                "n_events"
            ]

            event_window_rows.append(
                event_window
            )

            

            

          

            features = (
                extract_window_features(
                    window
                )
            )

           

            row = {
                "window_id": window_id,

                "split": split_name,

                "participant_id":
                    participant_id,

                "session_id":
                    session_id,

                "trial_id":
                    trial_id,

                # Original task retained
                # for traceability.
                "original_task":
                    task,

                # Final five-class label.
                "label":
                    final_label,

                "tremor_status":
                    tremor_status,

                "window_start_sec":
                    meta[
                        "window_start_sec"
                    ],

                "window_end_sec":
                    meta[
                        "window_end_sec"
                    ],
            }

            # Add extracted features.
            row.update(
                features
            )

            ml_rows.append(
                row
            )

            # ------------------------------------------------
            # Window index
            # ------------------------------------------------

            window_index_rows.append(
                {
                    "window_id":
                        window_id,

                    "split":
                        split_name,

                    "participant_id":
                        participant_id,

                    "session_id":
                        session_id,

                    "trial_id":
                        trial_id,

                    "original_task":
                        task,

                    "label":
                        final_label,

                    "tremor_status":
                        tremor_status,

                    "window_start_sec":
                        meta[
                            "window_start_sec"
                        ],

                    "window_end_sec":
                        meta[
                            "window_end_sec"
                        ],

                    "window_duration_sec":
                        meta[
                            "window_duration_sec"
                        ],

                    "n_events":
                        meta[
                            "n_events"
                        ],
                }
            )

    # CREATE DATAFRAMES

    if not ml_rows:

        raise RuntimeError(
            f"No ML windows generated "
            f"for {split_name}."
        )

    ml_df = pd.DataFrame(
        ml_rows
    )

    event_df = pd.concat(
        event_window_rows,
        ignore_index=True,
    )

    index_df = pd.DataFrame(
        window_index_rows
    )

    # SAVE COMPLETE EVENT-LEVEL WINDOWS

    event_path = (
        output_path
        / f"{split_name}_windowed_events.csv"
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    # SAVE ML DATASET

    ml_path = (
        output_path
        / f"{split_name}_ml_features.csv"
    )

    ml_df.to_csv(
        ml_path,
        index=False,
    )

    # SAVE WINDOW INDEX

    index_path = (
        output_path
        / f"{split_name}_window_index.csv"
    )

    index_df.to_csv(
        index_path,
        index=False,
    )

    # TREMOR / CLEAN ML DATASETS

    tremor_ml = ml_df[
        ml_df[
            "tremor_status"
        ] == 1
    ].copy()

    clean_ml = ml_df[
        ml_df[
            "tremor_status"
        ] == 0
    ].copy()

    tremor_path = (
        output_path
        / f"{split_name}_tremor_ml_features.csv"
    )

    clean_path = (
        output_path
        / f"{split_name}_clean_ml_features.csv"
    )

    tremor_ml.to_csv(
        tremor_path,
        index=False,
    )

    clean_ml.to_csv(
        clean_path,
        index=False,
    )

    # PRINT SUMMARY

    print()
    print("WINDOWING + FEATURE SUMMARY")
    print("-" * 75)

    print(
        f"Trials processed       : "
        f"{total_trials:,}"
    )

    print(
        f"Trials with windows    : "
        f"{trials_with_windows:,}"
    )

    print(
        f"Total windows          : "
        f"{total_windows:,}"
    )

    print(
        f"ML windows             : "
        f"{len(ml_df):,}"
    )

    print(
        f"Idle/excluded windows  : "
        f"{excluded_idle_windows:,}"
    )

    print()
    print("Final label distribution:")
    print(
        ml_df[
            "label"
        ].value_counts()
    )

    print()
    print("Tremor / clean:")
    print(
        ml_df[
            "tremor_status"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print("Files written:")
    print(
        f"  {event_path}"
    )

    print(
        f"  {ml_path}"
    )

    print(
        f"  {index_path}"
    )

    print(
        f"  {tremor_path}"
    )

    print(
        f"  {clean_path}"
    )

    print("=" * 75)


# MAIN

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "TREMORSHIELD windowing + "
            "feature extraction"
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train_combined.csv",
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test_combined.csv",
    )

    parser.add_argument(
        "--outdir",
        default="data/final/windows",
        help="Output directory",
    )

    parser.add_argument(
        "--window-ms",
        type=float,
        default=500.0,
        help="Window length in milliseconds.",
    )

    parser.add_argument(
        "--stride-ms",
        type=float,
        default=100.0,
        help="Window stride in milliseconds.",
    )

    parser.add_argument(
        "--min-events",
        type=int,
        default=5,
        help="Minimum events per window.",
    )

    parser.add_argument(
        "--min-duration-ms",
        type=float,
        default=200.0,
        help="Minimum window duration.",
    )

    args = parser.parse_args()

    # Global configuration.
    global WINDOW_SIZE_SEC
    global STRIDE_SEC
    global MIN_EVENTS
    global MIN_WINDOW_DURATION_SEC

    WINDOW_SIZE_SEC = (
        args.window_ms / 1000.0
    )

    STRIDE_SEC = (
        args.stride_ms / 1000.0
    )

    MIN_EVENTS = (
        args.min_events
    )

    MIN_WINDOW_DURATION_SEC = (
        args.min_duration_ms / 1000.0
    )

   

    process_dataset(
        input_csv=args.train,
        output_dir=args.outdir,
        split_name="train",
    )

 

    process_dataset(
        input_csv=args.test,
        output_dir=args.outdir,
        split_name="test",
    )

    print()
    print("=" * 75)
    print(
        "TREMORSHIELD WINDOWING + "
        "FEATURE EXTRACTION COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()




