import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


TRAIN_PATH = "data/final/windows/train_ml_features.csv"
TEST_PATH = "data/final/windows/test_ml_features.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("LOADING DATA")
print("=" * 75)

train = pd.read_csv(
    TRAIN_PATH,
    low_memory=False,
)

test = pd.read_csv(
    TEST_PATH,
    low_memory=False,
)

print(f"Train windows: {len(train):,}")
print(f"Test windows : {len(test):,}")


# ============================================================
# ONLY CLICKING + PRECISION
# ============================================================

LABELS = [
    "clicking",
    "precision",
]

train = train[
    train["label"].isin(LABELS)
].copy()

test = test[
    test["label"].isin(LABELS)
].copy()


# ============================================================
# SORT TEMPORALLY
# ============================================================

SORT_COLUMNS = [
    "participant_id",
    "session_id",
    "trial_id",
    "window_start_sec",
]

train = train.sort_values(
    SORT_COLUMNS
).reset_index(drop=True)

test = test.sort_values(
    SORT_COLUMNS
).reset_index(drop=True)


# ============================================================
# FEATURES
# ============================================================

EXCLUDE = [
    "window_id",
    "split",
    "participant_id",
    "session_id",
    "trial_id",
    "original_task",
    "label",
    "tremor_status",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
    "n_press_events",
    "n_release_events",
    "n_double_click_events",
]

FEATURES = [
    col
    for col in train.columns
    if (
        col not in EXCLUDE
        and pd.api.types.is_numeric_dtype(
            train[col]
        )
    )
]

print()
print("=" * 75)
print("FEATURES")
print("=" * 75)

print(f"Number of current-window features: {len(FEATURES)}")


# ============================================================
# CREATE PREVIOUS-WINDOW FEATURES
# ============================================================

def add_previous_window_features(
    df,
    features,
):

    result = df.copy()

    # Group by trial so that a previous window
    # never comes from another trial.

    grouped = result.groupby(
        [
            "participant_id",
            "session_id",
            "trial_id",
        ],
        sort=False,
    )

    for feature in features:

        previous = grouped[
            feature
        ].shift(1)

        result[
            feature + "_prev"
        ] = previous

    return result


print()
print("=" * 75)
print("ADDING TEMPORAL CONTEXT")
print("=" * 75)

train_temporal = (
    add_previous_window_features(
        train,
        FEATURES,
    )
)

test_temporal = (
    add_previous_window_features(
        test,
        FEATURES,
    )
)


# ============================================================
# TEMPORAL FEATURES
# ============================================================

TEMPORAL_FEATURES = (
    FEATURES
    +
    [
        feature + "_prev"
        for feature in FEATURES
    ]
)

print(
    f"Total temporal features: "
    f"{len(TEMPORAL_FEATURES)}"
)


# ============================================================
# REMOVE FIRST WINDOW OF EACH TRIAL
# ============================================================

# The first window has no previous-window information.

train_temporal = train_temporal.dropna(
    subset=[
        feature + "_prev"
        for feature in FEATURES
    ],
    how="all",
)

test_temporal = test_temporal.dropna(
    subset=[
        feature + "_prev"
        for feature in FEATURES
    ],
    how="all",
)


# ============================================================
# X / Y
# ============================================================

X_train = train_temporal[
    TEMPORAL_FEATURES
].copy()

X_test = test_temporal[
    TEMPORAL_FEATURES
].copy()

y_train = train_temporal[
    "label"
]

y_test = test_temporal[
    "label"
]


# ============================================================
# IMPUTATION
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
)


# ============================================================
# RANDOM FOREST
# ============================================================

print()
print("=" * 75)
print("TRAINING TEMPORAL RANDOM FOREST")
print("=" * 75)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICT
# ============================================================

pred = model.predict(
    X_test
)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    pred,
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        pred,
    )
)

macro_f1 = f1_score(
    y_test,
    pred,
    average="macro",
)

weighted_f1 = f1_score(
    y_test,
    pred,
    average="weighted",
)


print()
print("=" * 75)
print("TEMPORAL RESULTS")
print("=" * 75)

print(
    f"Accuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Balanced Accuracy : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1       : "
    f"{weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        pred,
        labels=[
            "clicking",
            "precision",
        ],
        digits=4,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

cm = confusion_matrix(
    y_test,
    pred,
    labels=[
        "clicking",
        "precision",
    ],
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_clicking",
        "Actual_precision",
    ],
    columns=[
        "Predicted_clicking",
        "Predicted_precision",
    ],
)

print(
    cm_df.to_string()
)


# ============================================================
# PRECISION ANALYSIS
# ============================================================

precision_mask = (
    y_test == "precision"
)

precision_total = int(
    precision_mask.sum()
)

precision_correct = int(
    np.sum(
        pred[precision_mask]
        == "precision"
    )
)

precision_as_clicking = int(
    np.sum(
        pred[precision_mask]
        == "clicking"
    )
)

print()
print("=" * 75)
print("PRECISION ANALYSIS")
print("=" * 75)

print(
    f"Actual precision windows : "
    f"{precision_total}"
)

print(
    f"Correctly predicted      : "
    f"{precision_correct}"
)

print(
    f"Predicted as clicking    : "
    f"{precision_as_clicking}"
)

print(
    f"Precision recall         : "
    f"{precision_correct / precision_total:.4f}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": TEMPORAL_FEATURES,
        "importance":
            model.feature_importances_,
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False,
    )
)

print()
print("=" * 75)
print("TOP 30 TEMPORAL FEATURES")
print("=" * 75)

print(
    importance
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "data/final/"
    "temporal_context_results.txt"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "TEMPORAL CLICKING VS PRECISION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Accuracy          : {accuracy:.4f}\n"
    )

    f.write(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}\n"
    )

    f.write(
        f"Macro F1          : {macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}\n\n"
    )

    f.write(
        classification_report(
            y_test,
            pred,
            labels=[
                "clicking",
                "precision",
            ],
            digits=4,
        )
    )

    f.write(
        "\n\nCONFUSION MATRIX\n"
    )

    f.write(
        cm_df.to_string()
    )

    f.write(
        "\n\nTOP FEATURES\n"
    )

    f.write(
        importance
        .head(30)
        .to_string(
            index=False
        )
    )

print()
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    f"Results saved to: {output_path}"
)