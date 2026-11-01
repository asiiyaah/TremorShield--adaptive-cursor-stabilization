

import numpy as np
import pandas as pd

from .tremor_model import generate_tremor, _unit_normals


def inject_tremor_into_trial(trial_df: pd.DataFrame, freq, amplitude_px, phase,
                              rng: np.random.Generator,
                              tremor_status: int) -> pd.DataFrame:
    
    trial_df = trial_df.sort_values("elapsed_sec").reset_index(drop=True)

    gt_x_norm = trial_df["x_normalized"].to_numpy(dtype=float)
    gt_y_norm = trial_df["y_normalized"].to_numpy(dtype=float)
    
    W = trial_df["screen_width"].to_numpy(dtype=float)
    H = trial_df["screen_height"].to_numpy(dtype=float)
    if np.any(W <= 0) or np.any(H <= 0):
        raise ValueError("screen_width and screen_height must be positive.")
    gt_x_px = gt_x_norm * W
    gt_y_px = gt_y_norm * H
    obs_x_norm = gt_x_norm.copy()
    obs_y_norm = gt_y_norm.copy()

    trial_df["ground_truth_x"] = gt_x_norm
    trial_df["ground_truth_y"] = gt_y_norm
    trial_df["tremor_status"] = tremor_status

    if tremor_status == 1:
        dt = trial_df["dt"].fillna(0.0).to_numpy(dtype=float)
        t_local = trial_df["elapsed_sec"].to_numpy(dtype=float)

        nx, ny = _unit_normals(gt_x_px, gt_y_px, dt)
        s = generate_tremor(dt, freq, amplitude_px, phase, t_local, rng)

        move_mask = trial_df["event"].isin(["move", "press", "release", "double_click"]).to_numpy()
        obs_x_norm[move_mask] = gt_x_norm[move_mask] + s[move_mask] * nx[move_mask]/W[move_mask]
        obs_y_norm[move_mask] = gt_y_norm[move_mask] + s[move_mask] * ny[move_mask]/H[move_mask]
        # obs_x[move_mask] = gt_x[move_mask] + s[move_mask] * nx[move_mask]
        # obs_y[move_mask] = gt_y[move_mask] + s[move_mask] * ny[move_mask]
        
        #clipping (not sure)
        # obs_x_norm = np.clip(obs_x_norm, 0.0, 1.0)
        # obs_y_norm = np.clip(obs_y_norm, 0.0, 1.0)


        trial_df["tremor_frequency_hz"] = freq
        trial_df["tremor_amplitude_px"] = amplitude_px
        trial_df["tremor_phase_rad"] = phase
    else:
        trial_df["tremor_frequency_hz"] = np.nan
        trial_df["tremor_amplitude_px"] = np.nan
        trial_df["tremor_phase_rad"] = np.nan

    trial_df["observed_x"] = obs_x_norm
    trial_df["observed_y"] = obs_y_norm
    return trial_df
