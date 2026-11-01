"""
ML utilities for model training, evaluation, and serialization.
"""

import json
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import joblib


def evaluate_classifier(y_true, y_pred, y_pred_proba=None, model_name: str = "Model") -> Dict[str, Any]:
    """
    Evaluate a binary classifier and return metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC-AUC)
        model_name: Name of the model for display

    Returns:
        Dictionary of metrics
    """
    num_classes = len(np.unique(y_true))
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0, average="weighted" if num_classes > 2 else None)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, average="weighted" if num_classes > 2 else None)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0, average="weighted" if num_classes > 2 else None)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_pred_proba is not None:
        try:
            if num_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
        except Exception as e:
            print(f"[evaluate_classifier] ROC-AUC calculation failed: {e}")
            metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = [f"class_{i}" for i in range(num_classes)]

    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["per_class_metrics"] = class_report

    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    """
    Create and save a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    num_classes = cm.shape[0]
    if num_classes == 2:
        classes = ['Non-PD', 'PD']
    elif num_classes == 3:
        classes = ['Non-PD', 'PD', 'Suspected PD']
    else:
        classes = [f'Class {i}' for i in range(num_classes)]

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_title(f'Confusion Matrix - {model_name}')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"[plot_confusion_matrix] Saved to {save_path}")

    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name: str = "Model", save_path: str = None):
    """
    Create and save a ROC curve plot (binary classification only).
    """
    try:
        num_classes = len(np.unique(y_true))
        if num_classes != 2:
            print(f"[plot_roc_curve] Skipping ROC curve for {num_classes}-class problem")
            return

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"[plot_roc_curve] Saved to {save_path}")

        plt.close()
    except Exception as e:
        print(f"[plot_roc_curve] Error: {e}")


def save_model(model, scaler, feature_names: list, output_dir: str, model_name: str = "model"):
    """
    Save trained model, scaler, and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    scaler_path = os.path.join(output_dir, f"{model_name}_scaler.joblib")
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": type(model).__name__,
        "feature_names": feature_names,
        "num_features": len(feature_names),
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[save_model] Saved model to {model_path}")
    print(f"[save_model] Saved scaler to {scaler_path}")
    print(f"[save_model] Saved metadata to {metadata_path}")

    return model_path, scaler_path, metadata_path


def load_model(output_dir: str, model_name: str = "model"):
    """
    Load trained model, scaler, and metadata.
    """
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    scaler_path = os.path.join(output_dir, f"{model_name}_scaler.joblib")
    metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return model, scaler, metadata


def save_metrics(metrics: dict, output_path: str):
    """
    Save metrics to JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[save_metrics] Saved to {output_path}")


def save_model_comparison(comparisons: list, output_path: str):
    """
    Save model comparison table to JSON and CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # JSON format
    with open(output_path.replace('.json', '.json'), 'w') as f:
        json.dump(comparisons, f, indent=2)

    # CSV format
    df = pd.DataFrame(comparisons)
    csv_path = output_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)

    print(f"[save_model_comparison] Saved to {output_path}")
    print(f"[save_model_comparison] Saved to {csv_path}")

    return df
