

import pandas as pd

df = pd.read_csv("data/final/train_combined.csv")

idle = df[
    (df["participant_id"] == "U1") &
    (df["session_id"] == 1) &
    (df["trial_id"] == 9)
].copy()

print("=" * 70)
print("U1 IDLE TRIAL")
print("=" * 70)

cols = [
    "participant_id",
    "session_id",
    "trial_id",
    "task",
    "event",
    "timestamp",
    "x_normalized",
    "y_normalized",
    "observed_x",
    "observed_y",
    "ground_truth_x",
    "ground_truth_y"
]

cols = [c for c in cols if c in idle.columns]

print(idle[cols].to_string(index=False))
print("\nTIME DIFFERENCES")

idle = idle.sort_values("timestamp").copy()

idle["dt"] = idle["timestamp"].diff()

print(idle["dt"].describe())

print("\nLargest gaps:")
print(
    idle[
        ["timestamp", "dt"]
    ]
    .sort_values("dt", ascending=False)
    .head(20)
    .to_string(index=False)
)