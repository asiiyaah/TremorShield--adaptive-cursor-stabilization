

from dataclasses import dataclass #to create configuration objects
from typing import Tuple

# Full column set expected in all_cleaned_data.csv. load_data() raises a
# clear error (never silently proceeds) if any of these are missing.
# REQUIRED_COLUMNS = [
#     "master_row_id", "timestamp", "elapsed_sec", "dt", "user_id",
#     "session_id", "trial_id", "task", "action_type", "event", "button",
#     "drag", "x", "y", "dx", "dy", "velocity", "acceleration",
#     "direction_change", "target_x", "target_y", "target_width",
#     "target_height", "target_id", "screen_width", "screen_height",
#     "dt_original", "dx_original", "dy_original", "velocity_original",
#     "acceleration_original", "direction_change_original",
#     "dt_recalculated", "dx_recalculated", "dy_recalculated",
#     "vx_recalculated", "vy_recalculated", "velocity_recalculated",
#     "acceleration_recalculated", "direction_change_recalculated_audit",
#     "vx", "vy", "source_file",
# ]
REQUIRED_COLUMNS = [ 
    "participant_id",
    "dx",
    "dy",
    "velocity_x",
    "velocity_y",
    "velocity",
    "acceleration",
    "direction_change",
    "timestamp",
    "elapsed_sec",
    "dt",
    "x",
    "y",
    "dt_raw",
    "dx_raw",
    "dy_raw",
    "velocity_x_raw",
    "velocity_y_raw",
    "velocity_raw",
    "dv_raw",
    "acceleration_raw",
    "session_id",
    "trial_id",
    "task",
    "action_type",
    "event",
    "button",
    "drag",
    "x_normalized",
    "y_normalized",
    "target_x_normalized",
    "target_y_normalized",
    "target_width_normalized",
    "target_height_normalized",
    "target_id",
    "screen_width",
    "screen_height",]




# The Random Forest intent/context target. tremor_status is metadata, NOT
# part of this label set (see tremor_model.py docstring, point 2 in chat).
VALID_TASKS = {
    "normal", "fast", "slow", "click", "double_click", "drag",
    "precision", "target_selection", "idle",
}

TRIAL_KEYS = ["participant_id", "session_id", "trial_id"]

EPS_DT = 1e-4        # if dt=0, to avoid divide-by-zero
EPS_SPEED = 1e-6      # if speed<EPS_SPEED  movement direction is considered undefined to prevent dividing by zero when computing direction change


#OU (Ornstein–Uhlenbeck) process is one way to generate this smoothly changing random amplitude.
OU_TAU_SEC = 0.20     # TAU = how quickly the amplitude moves around.ie,Small value → changes quickly
OU_SIGMA = 0.55       # controls randomness of the amplitude. Large value → more randomness
OU_LEVEL = 0.55       # center/target around which the envelope fluctuates.
OU_MIN, OU_MAX = 0.05, 1.00  # clip range for r(t)
COORDINATE_COLUMNS = ("x_normalized", "y_normalized")
COORDINATE_SYSTEM = "normalized"

@dataclass
class TremorConfig:
    seed: int = 42
    train_frac: float = 0.80          
    train_tremor_frac: float = 0.70   
    test_tremor_frac: float = 1.00    
                                      
    frequencies_hz: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)
    amplitudes_px: Tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
