"""
FAST TASK CLASSIFICATION - 4 MODELS (Optimized for speed)
Dataset: ML Features (train_ml_features.csv, test_ml_features.csv)
Target: label (task performed by user)
Models: Linear Regression, KNN, Logistic Regression (Fast), Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score
)
import html
import re

print("="*80)
print("LOADING DATASETS")
print("="*80)

train_df = pd.read_csv("data/robustpdx/train_ml_features.csv")
test_df = pd.read_csv("data/robustpdx/test_ml_features.csv")

def clean_column_name(col):
    col = html.unescape(str(col))
    col = col.replace("**", "").strip()
    col = re.sub(r"^\*+", "", col)
    return col.strip()

train_df.columns = [clean_column_name(c) for c in train_df.columns]
test_df.columns = [clean_column_name(c) for c in test_df.columns]

print(f"\nTraining dataset shape: {train_df.shape}")
print(f"Testing dataset shape : {test_df.shape}")

TARGET = "label"

train_df = train_df.dropna(subset=[TARGET]).copy()
test_df = test_df.dropna(subset=[TARGET]).copy()

train_df[TARGET] = train_df[TARGET].astype(str).str.strip()
test_df[TARGET] = test_df[TARGET].astype(str).str.strip()

print("\n" + "="*80)
print("TASK DISTRIBUTION")
print("="*80)

train_distribution = train_df[TARGET].value_counts().sort_index()
test_distribution = test_df[TARGET].value_counts().sort_index()

distribution_table = pd.DataFrame({
    "Train Count": train_distribution,
    "Train %": (train_distribution / len(train_df) * 100),
    "Test Count": test_distribution,
    "Test %": (test_distribution / len(test_df) * 100)
}).fillna(0)

print("\nTask distribution:")
print(distribution_table.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

train_distribution.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title("Training Dataset: Task Distribution")
axes[0].set_xlabel("Task")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis='x', rotation=45)

test_distribution.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title("Testing Dataset: Task Distribution")
axes[1].set_xlabel("Task")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("task_distribution.png", dpi=100, bbox_inches='tight')
plt.close()

print("\nTask distribution plot saved: task_distribution.png")

print("\n" + "="*80)
print("FEATURE PREPARATION")
print("="*80)

excluded_columns = [
    "window_id", "split", "participant_id", "session_id", "trial_id",
    "original_task", "label", "tremor_status", "window_start_sec",
    "window_end_sec"
]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
    and col in test_df.columns
    and pd.api.types.is_numeric_dtype(train_df[col])
    and pd.api.types.is_numeric_dtype(test_df[col])
]

X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan)
X_test = test_df[feature_columns].replace([np.inf, -np.inf], np.nan)

y_train = train_df[TARGET]
y_test = test_df[TARGET]

print(f"Training windows : {len(X_train)}")
print(f"Testing windows  : {len(X_test)}")
print(f"Input features   : {len(feature_columns)}")
print(f"Train classes    : {sorted(y_train.unique())}")
print(f"Test classes     : {sorted(y_test.unique())}")

distribution_table.to_csv("task_distribution_train_test.csv")

print("\n" + "="*80)
print("DEFINING MODELS - FAST VERSION")
print("="*80)

models_config = {
    "Linear Regression (Logistic)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ))
    ]),
    "Fast Linear Classifier": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(
            max_iter=500,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results_summary = []

for model_name, model in models_config.items():

    print("\n" + "="*80)
    print(f"TRAINING: {model_name}")
    print("="*80)

    model.fit(X_train, y_train)
    print(f"[OK] Training completed")

    y_pred = model.predict(X_test)
    print(f"[OK] Predictions completed")

    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Performance Indicator": [
            "Accuracy", "Balanced Accuracy", "Macro Precision", "Macro Recall",
            "Macro F1-Score", "Weighted Precision", "Weighted Recall",
            "Weighted F1-Score", "Matthews Correlation Coefficient", "Cohen's Kappa"
        ],
        "Score": [
            accuracy, balanced_acc, precision_macro, recall_macro, f1_macro,
            precision_weighted, recall_weighted, f1_weighted, mcc, kappa
        ]
    })

    print("\n" + "-"*80)
    print(f"PERFORMANCE METRICS FOR {model_name}")
    print("-"*80)
    print(metrics_df.to_string(index=False, formatters={"Score": "{:.4f}".format}))
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"Weighted F1-Score: {f1_weighted*100:.2f}%")

    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(set(y_test.unique()) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f"{model_name}: Confusion Matrix")
    axes[0].set_xlabel("Predicted Task")
    axes[0].set_ylabel("Actual Task")

    cm_normalized = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=axes[1], cbar_kws={'label': 'Recall'})
    axes[1].set_title(f"{model_name}: Normalized Confusion Matrix (Recall)")
    axes[1].set_xlabel("Predicted Task")
    axes[1].set_ylabel("Actual Task")

    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{safe_name}_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved: {safe_name}_confusion_matrix.png")

    if model_name in ["Random Forest", "Linear Regression (Logistic)", "Fast Linear Classifier"]:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, 'feature_importances_'):
            importance_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": classifier.feature_importances_
            }).sort_values("Importance", ascending=False)
        elif hasattr(classifier, 'coef_'):
            importance_df = pd.DataFrame({
                "Feature": feature_columns,
                "Importance": np.abs(classifier.coef_[0]) if classifier.coef_.shape[0] == 1 else np.mean(np.abs(classifier.coef_), axis=0)
            }).sort_values("Importance", ascending=False)
        else:
            importance_df = None

        if importance_df is not None:
            print("\n" + "-"*80)
            print("TOP 15 IMPORTANT FEATURES")
            print("-"*80)
            print(importance_df.head(15).to_string(index=False))

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", palette="viridis")
            plt.title(f"{model_name}: Top 15 Feature Importances")
            plt.tight_layout()
            plt.savefig(f"{safe_name}_feature_importance.png", dpi=100, bbox_inches='tight')
            plt.close()
            print(f"\nFeature importance plot saved: {safe_name}_feature_importance.png")

            importance_df.to_csv(f"{safe_name}_feature_importance.csv", index=False)

    predictions_df = pd.DataFrame({
        "Actual_Task": y_test.reset_index(drop=True),
        "Predicted_Task": pd.Series(y_pred)
    })
    predictions_df["Correct"] = (predictions_df["Actual_Task"] == predictions_df["Predicted_Task"])
    predictions_df.to_csv(f"{safe_name}_task_predictions.csv", index=False)

    metrics_df.to_csv(f"{safe_name}_task_metrics.csv", index=False)

    results_summary.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Macro F1": f1_macro,
        "Weighted F1": f1_weighted,
        "MCC": mcc,
        "Kappa": kappa
    })

print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

summary_df = pd.DataFrame(results_summary)
print("\n")
print(summary_df.to_string(index=False, formatters={
    col: "{:.4f}".format for col in summary_df.columns if col != "Model"
}))

summary_df.to_csv("all_models_task_classification_summary.csv", index=False)

print("\n" + "="*80)
print("FILES SAVED")
print("="*80)
print("[OK] task_distribution_train_test.csv")
print("[OK] task_distribution.png")
print("[OK] all_models_task_classification_summary.csv")
print("\nFor each model:")
print("  • {model}_confusion_matrix.png")
print("  • {model}_feature_importance.png (if applicable)")
print("  • {model}_feature_importance.csv (if applicable)")
print("  • {model}_task_predictions.csv")
print("  • {model}_task_metrics.csv")

print("\n" + "="*80)
print("TASK CLASSIFICATION TRAINING COMPLETED")
print("="*80)
