

import os
import pandas as pd

from .config import REQUIRED_COLUMNS, VALID_TASKS


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required column(s): "
            f"{missing}. Refusing to proceed with assumptions about "
            "column naming. Update REQUIRED_COLUMNS or fix the input file."
        )

    unexpected_tasks = set(df["task"].unique()) - VALID_TASKS
    if unexpected_tasks:
        raise ValueError(
            f"Found task label(s) not in the expected set {sorted(VALID_TASKS)}: "
            f"{sorted(unexpected_tasks)}. Refusing to proceed silently."
        )

    print(f"[load_data] Loaded {len(df):,} rows, {df['participant_id'].nunique()} "
          f"participants, {df.shape[1]} columns from {path}")
    return df
