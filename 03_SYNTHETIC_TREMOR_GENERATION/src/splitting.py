

from typing import Tuple

import numpy as np
import pandas as pd

from .config import TRIAL_KEYS


def _trial_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (user_id, session_id, trial_id) with its task label."""
    return df.groupby(TRIAL_KEYS, as_index=False)["task"].first()


def split_participants(df: pd.DataFrame, seed: int,
                        train_frac: float = 0.80) -> Tuple[set, set, pd.DataFrame]:
   
    rng = np.random.default_rng(seed + 1)  # decorrelate from the rename shuffle
    users = np.array(sorted(df["participant_id"].unique(), key=lambda u: int(u[1:])))
    shuffled = rng.permutation(users)

    n_train = int(round(len(shuffled) * train_frac))
    n_train = max(1, min(n_train, len(shuffled) - 1))  # keep both splits non-empty

    train_users = set(shuffled[:n_train].tolist())
    test_users = set(shuffled[n_train:].tolist())

    assert train_users.isdisjoint(test_users), "train/test participant overlap!"

    split_df = pd.DataFrame(
        [{"participant_id": u, "split": "train"} for u in sorted(train_users, key=lambda u: int(u[1:]))]
        + [{"participant_id": u, "split": "test"} for u in sorted(test_users, key=lambda u: int(u[1:]))]
    )

    print(f"[split_participants] {len(train_users)} train / {len(test_users)} "
          f"test participants (target {train_frac:.0%}/{1-train_frac:.0%}, "
          f"seed={seed+1})")
    return train_users, test_users, split_df

#to split training trials into tremor vs clean
def split_training_trials(df_train: pd.DataFrame, seed: int,
                           tremor_frac: float = 0.70) -> pd.DataFrame:
   
    rng = np.random.default_rng(seed + 2)
    trials = _trial_table(df_train)

    tremor_flags = np.zeros(len(trials), dtype=int)

    #This prevents one task from accidentally getting almost all tremor while another gets almost none.
    for task, idx in trials.groupby("task").groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_tremor = int(round(len(idx) * tremor_frac))
        tremor_flags[idx[:n_tremor]] = 1

    trials = trials.copy()
    trials["tremor_status"] = tremor_flags

    pct = trials["tremor_status"].mean() * 100
    print(f"[split_training_trials] {len(trials)} training trials -> "
          f"{trials['tremor_status'].sum()} tremor ({pct:.1f}%), "
          f"{(trials['tremor_status'] == 0).sum()} clean, seed={seed+2}")
    return trials


def assign_test_trials(df_test: pd.DataFrame, tremor_frac: float = 1.00) -> pd.DataFrame:
    
    trials = _trial_table(df_test)
    n_select = int(round(len(trials) * tremor_frac))
    selected = trials.sample(n=n_select, random_state=0) if n_select < len(trials) else trials
    print(f"[assign_test_trials] {len(trials)} test trials; "
          f"{len(selected)} will also get a synthetic-tremor rendering "
          f"(TEST-B). All {len(trials)} remain available untouched as TEST-A.")
    return selected[TRIAL_KEYS + ["task"]].copy()
