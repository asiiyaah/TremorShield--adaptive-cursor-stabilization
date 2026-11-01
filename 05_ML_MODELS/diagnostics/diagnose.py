import pandas as pd
import numpy as np


TRAIN_PATH = "data/final/windows/train_windowed_events.csv"
TEST_PATH = "data/final/windows/test_windowed_events.csv"


def analyze(path, name):

    print()
    print("=" * 75)
    print(f"{name.upper()} TRIAL ANALYSIS")
    print("=" * 75)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Use the actual column names from windowing.py
    # --------------------------------------------------------

    # window_task = original task name
    # final_label = final context label

    required = [
        "participant_id",
        "session_id",
        "trial_id",
        "window_task",
        "final_label",
        "event",
        "button",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Normalize event/button values
    # --------------------------------------------------------

    df["event_clean"] = (
        df["event"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["button_clean"] = (
        df["button"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Event indicators
    # --------------------------------------------------------

    df["is_move"] = (
        df["event_clean"] == "move"
    )

    df["is_press"] = (
        df["event_clean"] == "press"
    )

    df["is_release"] = (
        df["event_clean"] == "release"
    )

    df["is_double_click"] = (
        df["event_clean"]
        == "double_click"
    )

    df["is_left_press"] = (
        df["is_press"]
        &
        (
            df["button_clean"]
            == "left"
        )
    )

    # --------------------------------------------------------
    # TRIAL-LEVEL aggregation
    # --------------------------------------------------------

    trial = (
        df.groupby(
            [
                "participant_id",
                "session_id",
                "trial_id",
                "window_task",
                "final_label",
            ],
            dropna=False,
        )
        .agg(
            n_rows=(
                "event_clean",
                "size",
            ),

            move_events=(
                "is_move",
                "sum",
            ),

            press_events=(
                "is_press",
                "sum",
            ),

            release_events=(
                "is_release",
                "sum",
            ),

            double_click_events=(
                "is_double_click",
                "sum",
            ),

            left_press_events=(
                "is_left_press",
                "sum",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    trial["move_ratio"] = (
        trial["move_events"]
        /
        trial["n_rows"]
    )

    trial["press_ratio"] = (
        trial["press_events"]
        /
        trial["n_rows"]
    )

    # --------------------------------------------------------
    # Summary by final label
    # --------------------------------------------------------

    summary = (
        trial
        .groupby(
            "final_label"
        )[
            [
                "n_rows",
                "move_events",
                "press_events",
                "release_events",
                "double_click_events",
                "left_press_events",
                "move_ratio",
                "press_ratio",
            ]
        ]
        .agg(
            [
                "mean",
                "median",
            ]
        )
    )

    print()
    print("=" * 75)
    print("TRIAL-LEVEL SUMMARY")
    print("=" * 75)

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # Percentage of trials containing events
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "PERCENTAGE OF TRIALS "
        "CONTAINING EVENT TYPES"
    )
    print("=" * 75)

    for label in [
        "clicking",
        "precision",
        "dragging",
        "target_selection",
        "navigation",
        "idle",
    ]:

        subset = trial[
            trial["final_label"]
            == label
        ]

        if len(subset) == 0:
            continue

        print()
        print(
            label.upper()
        )

        print(
            "Trials:",
            len(subset),
        )

        print(
            "With movement:",
            f"{100 * (subset['move_events'] > 0).mean():.2f}%"
        )

        print(
            "With button press:",
            f"{100 * (subset['press_events'] > 0).mean():.2f}%"
        )

        print(
            "With double click:",
            f"{100 * (subset['double_click_events'] > 0).mean():.2f}%"
        )

    # --------------------------------------------------------
    # Clicking vs precision
    # --------------------------------------------------------

    cp = trial[
        trial["final_label"].isin(
            [
                "clicking",
                "precision",
            ]
        )
    ].copy()

    print()
    print("=" * 75)
    print(
        "CLICKING VS PRECISION "
        "— TRIAL LEVEL"
    )
    print("=" * 75)

    if len(cp) > 0:

        print(
            cp.groupby(
                "final_label"
            )[
                [
                    "n_rows",
                    "move_events",
                    "press_events",
                    "release_events",
                    "double_click_events",
                    "move_ratio",
                    "press_ratio",
                ]
            ]
            .mean()
            .T
            .to_string()
        )

    # --------------------------------------------------------
    # Original task vs final label
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "ORIGINAL TASK -> FINAL LABEL"
    )
    print("=" * 75)

    task_label_counts = (
        trial.groupby(
            [
                "window_task",
                "final_label",
            ]
        )
        .size()
    )

    print(
        task_label_counts.to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = (
        "data/final/"
        f"{name.lower()}_trial_context.csv"
    )

    trial.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"Saved: {output}"
    )


# ============================================================
# MAIN
# ============================================================

analyze(
    TRAIN_PATH,
    "TRAIN",
)

analyze(
    TEST_PATH,
    "TEST",
)