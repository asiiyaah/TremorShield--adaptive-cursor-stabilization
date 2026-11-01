

from typing import Tuple

import numpy as np
import pandas as pd

from .config import EPS_DT, EPS_SPEED, OU_TAU_SEC, OU_SIGMA, OU_LEVEL, OU_MIN, OU_MAX


def generate_tremor(dt: np.ndarray, freq: float, amplitude_px: float,
                     phase: float, t_local: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    
    n = len(dt) 
    r = np.empty(n, dtype=float)
    r[0] = np.clip(rng.uniform(0.3, 0.9), OU_MIN, OU_MAX)
    theta = 1.0 / OU_TAU_SEC
    for i in range(1, n):
        dt_i = max(dt[i], EPS_DT)
        drift = theta * (OU_LEVEL - r[i - 1]) * dt_i
        diffusion = OU_SIGMA * np.sqrt(dt_i) * rng.normal()
        r[i] = np.clip(r[i - 1] + drift + diffusion, OU_MIN, OU_MAX)

    s = amplitude_px * r * np.sin(2 * np.pi * freq * t_local + phase)
    return s


def _unit_normals(x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    n = len(x)
    vx = np.zeros(n)
    vy = np.zeros(n)
    dt_safe = np.where(dt > EPS_DT, dt, EPS_DT)
    vx[1:] = (x[1:] - x[:-1]) / dt_safe[1:]
    vy[1:] = (y[1:] - y[:-1]) / dt_safe[1:]

    speed = np.sqrt(vx ** 2 + vy ** 2)
    valid = speed > EPS_SPEED

    nx = np.full(n, np.nan)
    ny = np.full(n, np.nan)
    nx[valid] = -vy[valid] / speed[valid]
    ny[valid] = vx[valid] / speed[valid]

    nx_s = pd.Series(nx).ffill().bfill()
    ny_s = pd.Series(ny).ffill().bfill()
    nx = nx_s.to_numpy().copy()  # .copy(): to_numpy() can return a read-only view
    ny = ny_s.to_numpy().copy()

    still_nan = np.isnan(nx)
    if still_nan.any():
        nx[still_nan] = 0.0
        ny[still_nan] = 1.0

    return nx, ny
