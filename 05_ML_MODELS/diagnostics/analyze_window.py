import pandas as pd
from pathlib import Path


TRAIN_FILE = Path("data/final/windows/train_windowed_events.csv")
TEST_FILE = Path("data/final/windows/test_windowed_events.csv")


def analyze(path, name):

    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    df = pd.read_csv(path, low_memory=False)

    print(f"Rows loaded: {len(df):,}")

    # ------------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------------
    required = [
        "participant_id",
        "session_id",
        "trial_id",
        "window_id",
        "window_task",
        "final_label",
        "event",
        "button"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print("\nMissing columns:")
        for c in missing:
            print(" ", c)
        return

    # ------------------------------------------------------------
    # Keep one record per window
    # ------------------------------------------------------------
    window_groups = [
        "participant_id",
        "session_id",
        "trial_id",
        "window_id"
    ]

    rows = []

    for keys, g in df.groupby(window_groups, sort=False):

        participant_id, session_id, trial_id, window_id = keys

        label = g["final_label"].iloc[0]
        task = g["window_task"].iloc[0]

        events = g["event"].fillna("").astype(str)
        buttons = g["button"].fillna("").astype(str)

        n_events = len(g)

        n_move = (events == "move").sum()
        n_press = (events == "press").sum()
        n_release = (events == "release").sum()
        n_double = (events == "double_click").sum()

        n_left_press = (
            (events == "press") &
            (buttons == "left")
        ).sum()

        rows.append({
            "participant_id": participant_id,
            "session_id": session_id,
            "trial_id": trial_id,
            "window_id": window_id,
            "window_task": task,
            "final_label": label,
            "n_events": n_events,
            "n_move": n_move,
            "n_press": n_press,
            "n_release": n_release,
            "n_double_click": n_double,
            "n_left_press": n_left_press
        })

    w = pd.DataFrame(rows)

    print(f"Unique windows: {len(w):,}")

    # ------------------------------------------------------------
    # Clicking vs precision
    # ------------------------------------------------------------

    subset = w[w["final_label"].isin(["clicking", "precision"])].copy()

    print("\n" + "-" * 75)
    print("CLICKING VS PRECISION — WINDOW LEVEL")
    print("-" * 75)

    summary = subset.groupby("final_label").agg(
        windows=("window_id", "count"),
        mean_events=("n_events", "mean"),
        mean_moves=("n_move", "mean"),
        mean_presses=("n_press", "mean"),
        mean_releases=("n_release", "mean"),
        mean_double_clicks=("n_double_click", "mean"),
        mean_left_presses=("n_left_press", "mean")
    )

    print(summary.to_string())

    # ------------------------------------------------------------
    # Percentage of windows containing each event
    # ------------------------------------------------------------

    print("\n" + "-" * 75)
    print("PERCENTAGE OF WINDOWS CONTAINING EVENTS")
    print("-" * 75)

    for label in ["clicking", "precision"]:

        x = subset[subset["final_label"] == label]

        print(f"\n{label.upper()}")
        print(f"Windows: {len(x):,}")

        print(
            f"With movement: "
            f"{(x['n_move'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"With press: "
            f"{(x['n_press'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"With release: "
            f"{(x['n_release'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"With double click: "
            f"{(x['n_double_click'] > 0).mean() * 100:.2f}%"
        )

        print(
            f"With left press: "
            f"{(x['n_left_press'] > 0).mean() * 100:.2f}%"
        )

    # ------------------------------------------------------------
    # Distribution of number of presses per window
    # ------------------------------------------------------------

    print("\n" + "-" * 75)
    print("PRESS COUNT DISTRIBUTION")
    print("-" * 75)

    for label in ["clicking", "precision"]:

        x = subset[subset["final_label"] == label]

        print(f"\n{label.upper()}")

        print(
            x["n_press"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # ------------------------------------------------------------
    # Double-click windows
    # ------------------------------------------------------------

    print("\n" + "-" * 75)
    print("DOUBLE-CLICK WINDOWS")
    print("-" * 75)

    dc = subset[subset["n_double_click"] > 0]

    print(
        dc["final_label"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output = Path(
        f"data/final/{name.lower().replace(' ', '_')}_window_context.csv"
    )

    w.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":

    analyze(
        TRAIN_FILE,
        "TRAIN"
    )

    analyze(
        TEST_FILE,
        "TEST"
    )