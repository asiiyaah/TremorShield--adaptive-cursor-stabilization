

import json
import os

import pandas as pd

from .config import TremorConfig


def save_outputs(outdir: str,
                  train_clean: pd.DataFrame, train_tremor: pd.DataFrame, train_combined: pd.DataFrame,
                  test_clean: pd.DataFrame, test_tremor: pd.DataFrame, test_combined: pd.DataFrame,
                  participant_split: pd.DataFrame,
                  tremor_metadata: pd.DataFrame, cfg: TremorConfig) -> None:
    os.makedirs(outdir, exist_ok=True)

    files = {
        "train_clean.csv": train_clean,
        "train_synthetic_tremor.csv": train_tremor,
        "train_combined.csv": train_combined,
        "test_clean.csv": test_clean,
        "test_synthetic_tremor.csv": test_tremor,
        "test_combined.csv": test_combined,
        "participant_split.csv": participant_split,
        "tremor_metadata.csv": tremor_metadata,
    }
    for name, d in files.items():
        path = os.path.join(outdir, name)
        d.to_csv(path, index=False)
        print(f"[save_outputs] wrote {path} ({len(d):,} rows)")

    config_path = os.path.join(outdir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    print(f"[save_outputs] wrote {config_path}")
