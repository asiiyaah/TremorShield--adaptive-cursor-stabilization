"""
Example: Load a trained model and make predictions on new data.

This script demonstrates how to:
1. Load a saved model and its preprocessing pipeline
2. Prepare new data in the same format as training
3. Make predictions

Usage:
    python inference.py --model gradient_boosting --input sample_data.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json


def load_model_pipeline(model_dir: str, model_name: str):
    """Load trained model, scaler, and metadata."""
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"[load_model_pipeline] Loaded {metadata['model_type']}")
    print(f"[load_model_pipeline] Expected features: {len(metadata['feature_names'])}")

    return model, scaler, metadata


def prepare_input(df: pd.DataFrame, feature_names: list, scaler):
    """Prepare input data to match training format."""

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # One-hot encode categorical columns (matching training)
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Select only features that were used in training
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        print(f"[prepare_input] Adding {len(missing_features)} missing categorical features (setting to 0)")
        for feature in missing_features:
            df[feature] = 0

    # Select only features that were used in training, in correct order
    df = df[feature_names]

    # Scale features using the fitted scaler
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=feature_names)

    return df_scaled


def predict(model, X, model_metadata: dict):
    """Make predictions on prepared data."""

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Map predictions back to original labels
    label_map_reverse = {0: "Non-PD", 1: "PD", 2: "Suspected PD"}

    predictions_df = pd.DataFrame({
        'predicted_class_id': y_pred,
        'predicted_class': [label_map_reverse.get(p, f"class_{p}") for p in y_pred],
        'confidence_nonpd': y_pred_proba[:, 0],
        'confidence_pd': y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else np.nan,
        'confidence_suspected': y_pred_proba[:, 2] if y_pred_proba.shape[1] > 2 else np.nan,
    })

    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Make predictions using trained models")
    parser.add_argument("--model", default="knn",
                       choices=["linear_regression", "knn", "svm", "random_forest"],
                       help="Which model to use")
    parser.add_argument("--model-dir", default="models/robustpdx",
                       help="Directory containing trained models")
    parser.add_argument("--input", default="data/robustpdx/03_feature_extracted_dataset.csv",
                       help="Input CSV file with samples to predict")
    parser.add_argument("--output", default="predictions.csv",
                       help="Output CSV file for predictions")
    parser.add_argument("--limit", type=int, default=10,
                       help="Number of samples to predict (default 10, 0 for all)")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TREMOR SHIELD — INFERENCE EXAMPLE")
    print("=" * 70)

    # Load model
    print(f"\n[main] Loading model: {args.model}")
    model, scaler, metadata = load_model_pipeline(args.model_dir, args.model)

    # Load data
    print(f"[main] Loading input data: {args.input}")
    df = pd.read_csv(args.input)
    print(f"[main] Loaded {len(df)} samples")

    if args.limit > 0:
        df = df.head(args.limit)
        print(f"[main] Using first {args.limit} samples")

    # Remove target if present (for inference on unlabeled data)
    if "Parkinson's Disease status" in df.columns:
        df = df.drop(columns=["Parkinson's Disease status"])

    # Remove Session ID if present (it's an identifier, not a feature)
    if "Session ID" in df.columns:
        session_ids = df["Session ID"].copy()
        df = df.drop(columns=["Session ID"])
    else:
        session_ids = None

    # Prepare data
    print("[main] Preparing data...")
    X_prepared = prepare_input(df, metadata['feature_names'], scaler)

    # Make predictions
    print("[main] Making predictions...")
    predictions = predict(model, X_prepared, metadata)

    # Add session ID if available
    if session_ids is not None:
        predictions.insert(0, 'Session_ID', session_ids.values)

    # Save results
    predictions.to_csv(args.output, index=False)
    print(f"[main] Saved predictions to {args.output}")

    # Display sample
    print("\n[main] Sample predictions:")
    print(predictions.head(10).to_string())

    print("\n" + "=" * 70 + "\n")

    return predictions


if __name__ == "__main__":
    main()
