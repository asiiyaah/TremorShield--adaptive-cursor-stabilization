"""
ML Training Pipeline for RobustPDX Parkinson's Disease Dataset

This script trains multiple classification models on the RobustPDX dataset
to predict Parkinson's Disease status from behavioral and motor task features.

Usage:
    python train.py

Output:
    - Trained models and scalers in models/robustpdx/
    - Evaluation metrics and visualizations in models/robustpdx/
    - Model comparison report in models/robustpdx/model_comparison.json
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.ml_utils import (
    evaluate_classifier, plot_confusion_matrix, plot_roc_curve,
    save_model, save_metrics, save_model_comparison
)


def load_robustpdx_data(data_path: str = "data/robustpdx/03_feature_extracted_dataset.csv"):
    """
    Load and validate RobustPDX dataset.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[load_robustpdx_data] Loaded {len(df):,} samples with {df.shape[1]} columns")

    # Validate target column
    if "Parkinson's Disease status" not in df.columns:
        raise ValueError("Target column 'Parkinson's Disease status' not found")

    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """
    Prepare data: clean, split into train/val/test, and scale.

    Args:
        df: Input dataframe with target 'Parkinson's Disease status'
        test_size: Proportion for test set (0.15 = 15%)
        val_size: Proportion for validation set within training data (0.15 = 15% of 85%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler, label_encoder)
    """

    print("\n[prepare_data] Preparing dataset...")

    # Extract target and features
    target_col = "Parkinson's Disease status"
    y = df[target_col].copy()

    # Encode labels: nonpd -> 0, pd -> 1
    label_map = {"nonpd": 0, "pd": 1, "suspectedpd": 2}
    y_encoded = y.map(label_map)

    if y_encoded.isna().any():
        raise ValueError(f"Unknown label values in target. Found: {y.unique()}")

    print(f"[prepare_data] Target distribution: {y.value_counts().to_dict()}")

    # Select feature columns (exclude identifiers and target)
    exclude_cols = ["Session ID", target_col, "Parkinson's Disease status"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    # Identify categorical columns and encode them
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    if categorical_cols:
        print(f"[prepare_data] Encoding {len(categorical_cols)} categorical columns")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    feature_cols = list(X.columns)
    print(f"[prepare_data] Using {len(feature_cols)} features (after encoding)")

    # Handle missing values: forward fill then backward fill, then drop remaining
    print(f"[prepare_data] Missing values before cleaning: {X.isna().sum().sum()}")
    X = X.fillna(X.mean(numeric_only=True))
    remaining_na = X.isna().sum().sum()
    if remaining_na > 0:
        print(f"[prepare_data] Warning: {remaining_na} values still NA after mean imputation, dropping rows")
        valid_idx = ~X.isna().any(axis=1)
        X = X[valid_idx].reset_index(drop=True)
        y_encoded = y_encoded[valid_idx].reset_index(drop=True)

    print(f"[prepare_data] Final dataset size: {len(X):,} samples")

    # Train/test split (80/20 stratified by class)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=seed, stratify=y_encoded
    )

    # Train/val split (85/15 of the training set, stratified by class)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=seed, stratify=y_temp
    )

    print(f"[prepare_data] Train set: {len(X_train):,} samples")
    print(f"[prepare_data] Validation set: {len(X_val):,} samples")
    print(f"[prepare_data] Test set: {len(X_test):,} samples")
    print(f"[prepare_data] Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"[prepare_data] Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # Scale features (fit on training set only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to dataframes for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

    print("[prepare_data] Data scaling complete")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            feature_cols, scaler, label_map)


def train_models(X_train, X_val, X_test, y_train, y_val, y_test, output_dir: str, scaler=None, seed: int = 42):
    """
    Train multiple models and evaluate on validation/test sets.
    """

    os.makedirs(output_dir, exist_ok=True)

    models_config = [
        {
            "name": "Linear Regression",
            "model": LinearRegression(),
        },
        {
            "name": "KNN",
            "model": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        },
        {
            "name": "SVM",
            "model": SVC(kernel='rbf', C=1.0, random_state=seed, probability=True),
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1, class_weight='balanced'),
        },
    ]

    results = []

    print("\n[train_models] Starting model training...")

    for config in models_config:
        model_name = config["name"]
        model = config["model"]

        print(f"\n[train_models] Training {model_name}...")

        # Train on training set
        model.fit(X_train, y_train)

        # Predict on validation and test sets
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # For Linear Regression (continuous output), round to nearest class
        if model_name == "Linear Regression":
            y_val_pred = np.round(y_val_pred).astype(int)
            y_test_pred = np.round(y_test_pred).astype(int)
            # Clip to valid class range (0, 1, 2)
            y_val_pred = np.clip(y_val_pred, 0, 2)
            y_test_pred = np.clip(y_test_pred, 0, 2)

        # Get probabilities for ROC-AUC (if available)
        try:
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_val_proba = None
            y_test_proba = None

        # Evaluate
        val_metrics = evaluate_classifier(y_val, y_val_pred, y_val_proba, f"{model_name} (Validation)")
        test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_proba, f"{model_name} (Test)")

        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        if test_metrics.get('roc_auc') is not None:
            print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

        # Save model
        model_key = model_name.lower().replace(" ", "_")
        feature_names = list(X_train.columns)
        save_model(model, scaler, feature_names, output_dir, model_key)

        # Save metrics
        save_metrics(val_metrics, os.path.join(output_dir, f"{model_key}_metrics_val.json"))
        save_metrics(test_metrics, os.path.join(output_dir, f"{model_key}_metrics_test.json"))

        # Save visualizations
        plot_confusion_matrix(y_test, y_test_pred, model_name,
                             os.path.join(output_dir, f"{model_key}_confusion_matrix.png"))

        if y_test_proba is not None:
            plot_roc_curve(y_test, y_test_proba, model_name,
                          os.path.join(output_dir, f"{model_key}_roc_curve.png"))

        # Collect results for comparison
        comparison_row = {
            "Model": model_name,
            "Val_Accuracy": float(val_metrics['accuracy']),
            "Test_Accuracy": float(test_metrics['accuracy']),
            "Test_Precision": float(test_metrics['precision']),
            "Test_Recall": float(test_metrics['recall']),
            "Test_F1": float(test_metrics['f1']),
            "Test_Macro_F1": float(test_metrics['macro_f1']),
            "Test_Weighted_F1": float(test_metrics['weighted_f1']),
        }

        if test_metrics.get('roc_auc') is not None:
            comparison_row["Test_ROC_AUC"] = float(test_metrics['roc_auc'])

        results.append(comparison_row)

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
            }).sort_values('importance', ascending=False)

            importance_path = os.path.join(output_dir, f"{model_key}_feature_importance.csv")
            feature_importance_df.to_csv(importance_path, index=False)
            print(f"  Top 5 features: {feature_importance_df.head(5)['feature'].tolist()}")

    # Save model comparison
    comparison_df = save_model_comparison(results, os.path.join(output_dir, "model_comparison.json"))

    print("\n[train_models] Model Comparison Table:")
    print(comparison_df.to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ML models on RobustPDX dataset")
    parser.add_argument("--data", default="data/robustpdx/03_feature_extracted_dataset.csv",
                       help="Path to RobustPDX dataset")
    parser.add_argument("--output", default="models/robustpdx",
                       help="Output directory for models and metrics")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--test-size", type=float, default=0.15,
                       help="Test set size (default 0.15)")
    parser.add_argument("--val-size", type=float, default=0.15,
                       help="Validation set size within training (default 0.15)")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TREMOR SHIELD — ML TRAINING PIPELINE (RobustPDX Dataset)")
    print("=" * 70)

    # Load data
    df = load_robustpdx_data(args.data)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler, label_map = (
        prepare_data(df, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    )

    # Train models
    results = train_models(X_train, X_val, X_test, y_train, y_val, y_test, args.output, scaler, args.seed)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {os.path.abspath(args.output)}")
    print(f"Models saved: {len(results)}")
    print(f"Configuration:")
    print(f"  Dataset: {args.data}")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Random seed: {args.seed}")
    print(f"\nBest model by test accuracy: {max(results, key=lambda x: x['Test_Accuracy'])['Model']}")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    main()
