

import numpy as np
import pandas as pd

from .config import EPS_DT


def recalculate_features(trial_df: pd.DataFrame) -> pd.DataFrame:
    trial_df = trial_df.reset_index(drop=True)
    n = len(trial_df)
    x = trial_df["observed_x"].to_numpy(dtype=float)
    y = trial_df["observed_y"].to_numpy(dtype=float)
    dt = trial_df["dt"].fillna(0.0).to_numpy(dtype=float)
    dt_safe = np.where(dt > EPS_DT, dt, EPS_DT)

    dx = np.zeros(n)
    dy = np.zeros(n)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]

    vx = np.zeros(n)
    vy = np.zeros(n)
    vx[1:] = dx[1:] / dt_safe[1:]
    vy[1:] = dy[1:] / dt_safe[1:]

    velocity = np.sqrt(vx ** 2 + vy ** 2)

    acceleration = np.zeros(n)
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / dt_safe[1:]

    heading = np.arctan2(vy, vx)
    direction_change = np.zeros(n)
    dh = heading[1:] - heading[:-1]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    direction_change[1:] = np.abs(dh)

    dx[0] = dy[0] = vx[0] = vy[0] = velocity[0] = acceleration[0] = direction_change[0] = np.nan

    trial_df["dx_observed"] = dx
    trial_df["dy_observed"] = dy
    trial_df["vx_observed"] = vx
    trial_df["vy_observed"] = vy
    trial_df["velocity_observed"] = velocity
    trial_df["acceleration_observed"] = acceleration
    trial_df["direction_change_observed"] = direction_change
    return trial_df
