
from typing import List, Tuple
import hashlib
import numpy as np
import pandas as pd

from .config import TRIAL_KEYS, TremorConfig
from .injection import inject_tremor_into_trial
from .features import recalculate_features


def build_condition(df_split: pd.DataFrame, trial_assignment: pd.DataFrame,
                     cfg: TremorConfig, seed_offset: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    assignment = trial_assignment.set_index(TRIAL_KEYS)

    out_rows: List[pd.DataFrame] = []
    meta_rows: List[dict] = []

    for key, trial_df in df_split.groupby(TRIAL_KEYS, sort=False):
        if key not in assignment.index:
            continue  # trial not selected for this condition (e.g. TEST-B subset)
        tremor_status = int(assignment.loc[key, "tremor_status"])

        # Per-trial seed: reproducible, independent of groupby iteration order.
        seed_string = f"{seed_offset}|{key[0]}|{key[1]}|{key[2]}"
        trial_seed = int(hashlib.sha256(seed_string.encode("utf-8")).hexdigest()[:8],
    16
)
        rng = np.random.default_rng(trial_seed)

        if tremor_status == 1:
            freq = rng.choice(cfg.frequencies_hz)
            amplitude = rng.choice(cfg.amplitudes_px)
            phase = rng.uniform(0, 2 * np.pi)
        else:
            freq = amplitude = phase = None

        trial_df = inject_tremor_into_trial(
            trial_df, freq, amplitude, phase, rng, tremor_status
        )
        trial_df = recalculate_features(trial_df)
        out_rows.append(trial_df)

        meta_rows.append({
            "participant_id": key[0], "session_id": key[1], "trial_id": key[2],
            "task": trial_df["task"].iloc[0],
            "tremor_status": tremor_status,
            "tremor_frequency_hz": freq if freq is not None else np.nan,
            "tremor_amplitude_px": amplitude if amplitude is not None else np.nan,
            "tremor_phase_rad": phase if phase is not None else np.nan,
            "n_rows": len(trial_df),
            "random_seed": trial_seed,
        })

    rows_df = pd.concat(out_rows, ignore_index=True) if out_rows else df_split.iloc[0:0].copy()
    meta_df = pd.DataFrame(meta_rows)
    return rows_df, meta_df
