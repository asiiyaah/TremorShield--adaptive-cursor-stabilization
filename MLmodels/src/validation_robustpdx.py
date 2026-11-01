


from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# ROBUSTPDX COLUMN CONFIGURATION
# ============================================================

@dataclass
class RobustPDXColumnMap:
    """
    Column names used by the actual RobustPDX raw dataset.
    """

    path: Optional[str] = None

    session_col: str = "Session_ID"
    status_col: str = "status"

    round1_points_col: str = "r1points"
    round2_points_col: str = "r2points"
    round3_points_col: str = "r3points"

    # These are keyboard reaction-time fields.
    # They are intentionally not used for mouse trajectory
    # frequency validation.
    round1_timestamp_col: str = "timestamps.round1"
    round2_timestamp_col: str = "timestamps.round2"
    round3_timestamp_col: str = "timestamps.round3"


# ============================================================
# LOAD ROBUSTPDX
# ============================================================

def load_robustpdx(
    colmap: RobustPDXColumnMap
) -> Optional[pd.DataFrame]:
    """
    Load the actual RobustPDX raw CSV.

    Returns
    -------
    pd.DataFrame
        Raw RobustPDX data if available.

    None
        If the file cannot be loaded or required columns are missing.
    """

    if not colmap.path:
        print(
            "[validation_robustpdx] "
            "No RobustPDX path configured. "
            "Skipping RobustPDX validation."
        )
        return None

    path = Path(colmap.path)

    if not path.exists():
        print(
            "[validation_robustpdx] "
            f"RobustPDX file not found: {path}"
        )
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(
            "[validation_robustpdx] "
            f"Could not read RobustPDX CSV: {exc}"
        )
        return None

    required_columns = [
        colmap.session_col,
        colmap.status_col,
        colmap.round1_points_col,
        colmap.round2_points_col,
        colmap.round3_points_col,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        print(
            "[validation_robustpdx] "
            f"Missing required columns: {missing}"
        )

        print(
            "[validation_robustpdx] "
            "Available columns:"
        )
        print(list(df.columns))

        return None

    print(
        "[validation_robustpdx] "
        f"Loaded {len(df):,} RobustPDX records."
    )

    groups = sorted(
        df[colmap.status_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    print(
        "[validation_robustpdx] "
        f"Clinical groups: {groups}"
    )

    return df


# ============================================================
# PARSE ONE POINT
# ============================================================

def _parse_single_point(
    raw_point: str,
    round_name: str,
    point_index: int
) -> Optional[dict]:
    """
    Parse one encoded RobustPDX point.

    The RobustPDX point strings use 'NEXT' as the separator
    between consecutive points.

    The point representation contains X/Y information and
    additional task-specific information.

    This parser extracts the first two coordinate values as X
    and Y, and then interprets the remaining information
    according to the round.

    For r1:
        X
        Y
        inside/outside
        centerline distance

    For r2/r3:
        X
        Y
        inside/outside

    Returns None when the point cannot be parsed safely.
    """

    if raw_point is None:
        return None

    if pd.isna(raw_point):
        return None

    raw_point = str(raw_point).strip()

    if not raw_point:
        return None

    # --------------------------------------------------------
    # The encoded point format uses "s" between fields.
    #
    # Example structure:
    #
    #     XsYsinside/outside...
    #
    # We first split the point into fields.
    # --------------------------------------------------------

    fields = raw_point.split("s")

    if len(fields) < 2:
        return None

    # --------------------------------------------------------
    # X coordinate
    # --------------------------------------------------------

    try:
        x = float(fields[0])
    except (ValueError, TypeError):
        return None

    # --------------------------------------------------------
    # Y coordinate
    # --------------------------------------------------------

    try:
        y = float(fields[1])
    except (ValueError, TypeError):
        return None

    result = {
        "round": round_name,
        "point_index": point_index,
        "x": x,
        "y": y,
        "inside_target": np.nan,
        "distance_from_centerline_px": np.nan,
        "raw_point": raw_point,
    }

    # --------------------------------------------------------
    # Find boolean inside/outside field
    # --------------------------------------------------------

    boolean_index = None

    for i, field in enumerate(fields[2:], start=2):

        value = field.strip().lower()

        if value == "true":
            result["inside_target"] = True
            boolean_index = i
            break

        if value == "false":
            result["inside_target"] = False
            boolean_index = i
            break

    # --------------------------------------------------------
    # r1 contains centerline-distance information.
    #
    # We look for the numeric field immediately after the
    # inside/outside flag.
    # --------------------------------------------------------

    if round_name == "r1" and boolean_index is not None:

        if boolean_index + 1 < len(fields):

            distance_text = fields[
                boolean_index + 1
            ].strip()

            try:
                result[
                    "distance_from_centerline_px"
                ] = float(distance_text)

            except (ValueError, TypeError):
                pass

    return result


# ============================================================
# PARSE POINT SEQUENCE
# ============================================================

def parse_point_sequence(
    point_string,
    round_name: str
) -> list:
    """
    Parse a complete r1/r2/r3 point sequence.

    Points are separated by the token:

        NEXT
    """

    if pd.isna(point_string):
        return []

    point_string = str(point_string).strip()

    if not point_string:
        return []

    raw_points = point_string.split("NEXT")

    parsed = []

    for point_index, raw_point in enumerate(raw_points):

        raw_point = raw_point.strip()

        if not raw_point:
            continue

        point = _parse_single_point(
            raw_point=raw_point,
            round_name=round_name,
            point_index=point_index,
        )

        if point is not None:
            parsed.append(point)

    return parsed


# ============================================================
# STANDARDIZE ROBUSTPDX DATASET
# ============================================================

def standardize_robustpdx(
    df: pd.DataFrame,
    colmap: RobustPDXColumnMap
) -> pd.DataFrame:
    """
    Convert the encoded RobustPDX trajectory fields into one
    standardized point-level dataframe.

    Output columns:

        session_id
        status
        round
        point_index
        x
        y
        inside_target
        distance_from_centerline_px
        raw_point
    """

    all_points = []

    for _, row in df.iterrows():

        session_id = row[
            colmap.session_col
        ]

        status = str(
            row[colmap.status_col]
        ).strip().lower()

        # ----------------------------------------------------
        # Round 1: straight-line
        # ----------------------------------------------------

        r1_points = parse_point_sequence(
            row[colmap.round1_points_col],
            "r1",
        )

        for point in r1_points:

            point["session_id"] = session_id
            point["status"] = status

            all_points.append(point)

        # ----------------------------------------------------
        # Round 2: sine-wave
        # ----------------------------------------------------

        r2_points = parse_point_sequence(
            row[colmap.round2_points_col],
            "r2",
        )

        for point in r2_points:

            point["session_id"] = session_id
            point["status"] = status

            all_points.append(point)

        # ----------------------------------------------------
        # Round 3: spiral
        # ----------------------------------------------------

        r3_points = parse_point_sequence(
            row[colmap.round3_points_col],
            "r3",
        )

        for point in r3_points:

            point["session_id"] = session_id
            point["status"] = status

            all_points.append(point)

    if not all_points:

        return pd.DataFrame(
            columns=[
                "session_id",
                "status",
                "round",
                "point_index",
                "x",
                "y",
                "inside_target",
                "distance_from_centerline_px",
                "raw_point",
            ]
        )

    standardized = pd.DataFrame(
        all_points
    )

    standardized = standardized[
        [
            "session_id",
            "status",
            "round",
            "point_index",
            "x",
            "y",
            "inside_target",
            "distance_from_centerline_px",
            "raw_point",
        ]
    ]

    return standardized


# ============================================================
# PATH LENGTH
# ============================================================

def _path_length(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Calculate total distance travelled along a trajectory.
    """

    if len(x) < 2:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)

    distance = np.sqrt(
        dx ** 2 +
        dy ** 2
    )

    return float(
        np.sum(distance)
    )


# ============================================================
# DIRECTION CHANGES
# ============================================================

def _direction_changes(
    x: np.ndarray,
    y: np.ndarray,
    threshold_rad: float = np.deg2rad(15)
) -> int:
    """
    Count meaningful changes in movement direction.

    Changes smaller than 15 degrees are ignored so that tiny
    numerical fluctuations do not become direction changes.
    """

    if len(x) < 3:
        return 0

    dx = np.diff(x)
    dy = np.diff(y)

    heading = np.arctan2(
        dy,
        dx
    )

    dh = np.diff(
        heading
    )

    # Correct angle wraparound.
    dh = (
        (dh + np.pi)
        % (2 * np.pi)
    ) - np.pi

    return int(
        np.sum(
            np.abs(dh) > threshold_rad
        )
    )


# ============================================================
# SPATIAL FEATURES FOR EACH TRACE
# ============================================================

def compute_robustpdx_features(
    standardized_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute spatial features for every RobustPDX trace.

    A trace is identified by:

        Session_ID + round

    Features:

        n_points
        path_length_px
        x_std_px
        y_std_px
        bbox_width_px
        bbox_height_px
        bbox_area_px2
        direction_changes
        percent_inside_target

    For r1 only:

        centerline_deviation_mean_px
        centerline_deviation_rms_px
        centerline_deviation_p95_px
        centerline_deviation_max_px
    """

    if standardized_df.empty:
        return pd.DataFrame()

    results = []

    grouped = standardized_df.groupby(
        [
            "session_id",
            "round"
        ],
        sort=False
    )

    for (
        session_id,
        round_name
    ), group_df in grouped:

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        x = group_df[
            "x"
        ].to_numpy(
            dtype=float
        )

        y = group_df[
            "y"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # Keep only finite coordinates
        # ----------------------------------------------------

        finite = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        x = x[finite]
        y = y[finite]

        if len(x) < 3:
            continue

        status = str(
            group_df["status"].iloc[0]
        )

        # ----------------------------------------------------
        # Basic spatial features
        # ----------------------------------------------------

        path_length = _path_length(
            x,
            y
        )

        direction_changes = _direction_changes(
            x,
            y
        )

        bbox_width = float(
            x.max() - x.min()
        )

        bbox_height = float(
            y.max() - y.min()
        )

        bbox_area = (
            bbox_width *
            bbox_height
        )

        # ----------------------------------------------------
        # Percentage of points inside target
        # ----------------------------------------------------

        inside_values = group_df[
            "inside_target"
        ]

        inside_values = inside_values[
            inside_values.notna()
        ]

        if len(inside_values) > 0:

            percent_inside = (
                100.0 *
                inside_values.astype(bool).mean()
            )

        else:

            percent_inside = np.nan

        # ----------------------------------------------------
        # Base result
        # ----------------------------------------------------

        features = {

            "session_id":
                session_id,

            "status":
                status,

            "round":
                round_name,

            "n_points":
                len(x),

            "path_length_px":
                path_length,

            "x_std_px":
                float(x.std()),

            "y_std_px":
                float(y.std()),

            "bbox_width_px":
                bbox_width,

            "bbox_height_px":
                bbox_height,

            "bbox_area_px2":
                bbox_area,

            "direction_changes":
                direction_changes,

            "percent_inside_target":
                percent_inside,

        }

        # ----------------------------------------------------
        # r1-specific centerline deviation
        # ----------------------------------------------------

        if round_name == "r1":

            # We need to keep the same finite-coordinate rows
            # when extracting the centerline distances.

            valid_group = group_df[
                np.isfinite(
                    group_df["x"]
                )
                &
                np.isfinite(
                    group_df["y"]
                )
            ]

            distance = valid_group[
                "distance_from_centerline_px"
            ].to_numpy(
                dtype=float
            )

            distance = distance[
                np.isfinite(distance)
            ]

            if len(distance) > 0:

                features.update(
                    {
                        "centerline_deviation_mean_px":
                            float(
                                distance.mean()
                            ),

                        "centerline_deviation_rms_px":
                            float(
                                np.sqrt(
                                    np.mean(
                                        distance ** 2
                                    )
                                )
                            ),

                        "centerline_deviation_p95_px":
                            float(
                                np.percentile(
                                    distance,
                                    95
                                )
                            ),

                        "centerline_deviation_max_px":
                            float(
                                distance.max()
                            ),
                    }
                )

            else:

                features.update(
                    {
                        "centerline_deviation_mean_px":
                            np.nan,

                        "centerline_deviation_rms_px":
                            np.nan,

                        "centerline_deviation_p95_px":
                            np.nan,

                        "centerline_deviation_max_px":
                            np.nan,
                    }
                )

        else:

            features.update(
                {
                    "centerline_deviation_mean_px":
                        np.nan,

                    "centerline_deviation_rms_px":
                        np.nan,

                    "centerline_deviation_p95_px":
                        np.nan,

                    "centerline_deviation_max_px":
                        np.nan,
                }
            )

        results.append(
            features
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# ROBUSTPDX GROUP SUMMARY
# ============================================================

def summarize_robustpdx_groups(
    feature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Produce descriptive summaries separately for each
    RobustPDX clinical group.

    Example groups:

        pd
        nonpd

    These labels describe the RobustPDX reference population
    only.
    """

    if feature_df.empty:
        return pd.DataFrame()

    numeric_columns = [
        "n_points",
        "path_length_px",
        "x_std_px",
        "y_std_px",
        "bbox_width_px",
        "bbox_height_px",
        "bbox_area_px2",
        "direction_changes",
        "percent_inside_target",
        "centerline_deviation_mean_px",
        "centerline_deviation_rms_px",
        "centerline_deviation_p95_px",
        "centerline_deviation_max_px",
    ]

    available_columns = [
        column
        for column in numeric_columns
        if column in feature_df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    summary = (
        feature_df
        .groupby(
            "status"
        )[available_columns]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    return summary


# ============================================================
# MAIN ROBUSTPDX VALIDATION
# ============================================================

def run_robustpdx_validation(
    colmap: RobustPDXColumnMap,
    outdir: str
) -> Optional[pd.DataFrame]:
    """
    Run the complete RobustPDX spatial validation.

    Outputs:

        robustpdx_standardized_points.csv
        robustpdx_feature_summary.csv
        robustpdx_group_summary.csv

    Returns:

        Per-trace feature dataframe.
    """

    outdir = Path(
        outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 1 — Load raw RobustPDX
    # --------------------------------------------------------

    df = load_robustpdx(
        colmap
    )

    if df is None:

        pd.DataFrame(
            {
                "note": [
                    "RobustPDX was not available "
                    "for this validation run."
                ]
            }
        ).to_csv(
            outdir /
            "robustpdx_feature_summary.csv",
            index=False
        )

        return None

    # --------------------------------------------------------
    # STEP 2 — Parse trajectory point strings
    # --------------------------------------------------------

    standardized = standardize_robustpdx(
        df,
        colmap
    )

    standardized_path = (
        outdir /
        "robustpdx_standardized_points.csv"
    )

    standardized.to_csv(
        standardized_path,
        index=False
    )

    print(
        "[validation_robustpdx] "
        f"Wrote standardized points: "
        f"{standardized_path}"
    )

    print(
        "[validation_robustpdx] "
        f"Parsed {len(standardized):,} points."
    )

    # --------------------------------------------------------
    # STEP 3 — Calculate per-trace spatial features
    # --------------------------------------------------------

    features = compute_robustpdx_features(
        standardized
    )

    feature_path = (
        outdir /
        "robustpdx_feature_summary.csv"
    )

    features.to_csv(
        feature_path,
        index=False
    )

    print(
        "[validation_robustpdx] "
        f"Wrote feature summary: "
        f"{feature_path}"
    )

    print(
        "[validation_robustpdx] "
        f"Analysed {len(features)} traces."
    )

    # --------------------------------------------------------
    # STEP 4 — Clinical-group descriptive summary
    # --------------------------------------------------------

    group_summary = summarize_robustpdx_groups(
        features
    )

    group_summary_path = (
        outdir /
        "robustpdx_group_summary.csv"
    )

    group_summary.to_csv(
        group_summary_path
    )

    print(
        "[validation_robustpdx] "
        f"Wrote group summary: "
        f"{group_summary_path}"
    )

    # --------------------------------------------------------
    # STEP 5 — Basic logging
    # --------------------------------------------------------

    if not features.empty:

        statuses = sorted(
            features[
                "status"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        rounds = sorted(
            features[
                "round"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        print(
            "[validation_robustpdx] "
            f"Clinical groups analysed: {statuses}"
        )

        print(
            "[validation_robustpdx] "
            f"Tracing rounds analysed: {rounds}"
        )

    return features