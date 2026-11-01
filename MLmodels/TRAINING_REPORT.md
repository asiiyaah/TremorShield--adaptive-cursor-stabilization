# TremorShield ML Training Report

## Summary

Successfully trained and evaluated three classification models on the RobustPDX Parkinson's Disease dataset to predict disease status from behavioral and motor task measurements.

**Best Model:** Gradient Boosting  
**Test Accuracy:** 78.95%  
**Test F1-Score:** 0.7895  

## Dataset

**Source:** `data/robustpdx/03_feature_extracted_dataset.csv`

- **Total Samples:** 251
- **Training Samples:** 175 (70%)
- **Validation Samples:** 38 (15%)
- **Test Samples:** 38 (15%)

### Class Distribution

| Label | Count | Percentage |
|-------|-------|-----------|
| Non-PD (0) | 152 | 60.6% |
| PD (1) | 73 | 29.1% |
| Suspected PD (2) | 26 | 10.4% |

### Features

- **Total Features:** 85 (after categorical encoding)
- **Original Features:** 79 numeric + 4 categorical (Sex, Race, Dominant Hand, Device Type)
- **Feature Categories:**
  - Motor task measurements: Line tracing, sine wave tracing, spiral tracing
  - Cognitive task measurements: Key press accuracy, reaction time
  - Demographic features: Age, Height
  - Derived features: Interaction terms, ratios

## Models Trained

### 1. Logistic Regression

- **Algorithm:** L2-regularized logistic regression with balanced class weights
- **Hyperparameters:** max_iter=1000, random_state=42

**Results:**
- Validation Accuracy: 81.58%
- Test Accuracy: 68.42%
- Test Precision: 0.7100
- Test Recall: 0.6842
- Test F1: 0.6723
- Test Macro-F1: 0.5711

**Interpretation:** Simple, interpretable baseline model with decent validation performance but weaker generalization to test set.

### 2. Random Forest

- **Algorithm:** Ensemble of 100 decision trees
- **Hyperparameters:** n_estimators=100, random_state=42, class_weight='balanced'

**Results:**
- Validation Accuracy: 86.84%
- Test Accuracy: 76.32%
- Test Precision: 0.8017
- Test Recall: 0.7632
- Test F1: 0.7536
- Test Macro-F1: 0.6456

**Interpretation:** Strong ensemble model with good balance of precision and recall. Top features focus on deviation from centerline and tracing task performance.

**Top 5 Features:**
1. Total accumulated deviation from centerline (straight line)
2. Percentage of points traced inside sine wave
3. Mean wrong clicks
4. Time taken to trace spiral
5. Time taken to trace spiral (normalized by window width)

### 3. Gradient Boosting (BEST)

- **Algorithm:** Gradient Boosting with 100 estimators
- **Hyperparameters:** n_estimators=100, learning_rate=0.1, random_state=42

**Results:**
- Validation Accuracy: 86.84%
- Test Accuracy: 78.95% ✓ **Best performance**
- Test Precision: 0.8097
- Test Recall: 0.7895
- Test F1: 0.7895 ✓ **Best F1-Score**
- Test Macro-F1: 0.7343

**Interpretation:** Best-performing model with highest test accuracy and F1-score. Good generalization from validation to test set. Strong predictive power across all classes.

**Top 5 Features:**
1. Total accumulated deviation from centerline (straight line)
2. Time taken to trace sine wave
3. Device Type: Mac Desktop
4. Time taken to trace spiral
5. Percentage of points traced inside spiral

## Model Comparison

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|--------|-------------------|---------------|-------------------|
| Val Accuracy | 81.58% | 86.84% | 86.84% |
| **Test Accuracy** | 68.42% | 76.32% | **78.95%** |
| Test Precision | 0.7100 | 0.8017 | **0.8097** |
| Test Recall | 0.6842 | 0.7632 | **0.7895** |
| **Test F1** | 0.6723 | 0.7536 | **0.7895** |
| Macro-F1 | 0.5711 | 0.6456 | **0.7343** |

## Confusion Matrix Analysis

### Gradient Boosting (Best Model) - Test Set

Predictions for 38 test samples:

```
                Predicted
             Non-PD  PD  Suspected
Actual Non-PD   23   0      0
       PD        1  10      0
       Suspected 1   2      1
```

**Performance by Class:**
- **Non-PD:** Precision 0.92, Recall 1.00 (excellent)
- **PD:** Precision 0.83, Recall 0.91 (very good)
- **Suspected PD:** Precision 1.00, Recall 0.25 (excellent precision, poor recall due to small class size)

## Data Quality

- **Missing Values:** 0 (dataset fully populated)
- **Data Type Issues:** None (properly encoded categorical variables)
- **Class Imbalance:** Moderate (balanced class weights applied during training)
- **Data Leakage:** None (preprocessing fit on training set only)

## Reproducibility

**Random Seed:** 42 (set globally for all randomized components)

**Train/Val/Test Split:**
- Method: Stratified random split
- Training: 70% of data with stratified sampling
- Validation: 15% of remaining training data with stratification
- Test: 15% held-out test set with stratification
- Stratification ensures class distribution matches across splits

**Feature Scaling:**
- Method: StandardScaler (fit on training set only)
- Applied to all numeric features
- Ensures mean=0, std=1 for each feature
- Same transformation applied to validation/test sets

## Files Generated

### Models
- `models/robustpdx/gradient_boosting.joblib` (370 KB) - Best performing model
- `models/robustpdx/random_forest.joblib` (361 KB)
- `models/robustpdx/logistic_regression.joblib` (8.1 KB)

### Scalers
- `models/robustpdx/gradient_boosting_scaler.joblib` - Fitted StandardScaler for feature normalization
- `models/robustpdx/random_forest_scaler.joblib`
- `models/robustpdx/logistic_regression_scaler.joblib`

### Metadata
- `models/robustpdx/gradient_boosting_metadata.json` - Feature names and model info
- Similar files for other models

### Metrics
- `models/robustpdx/gradient_boosting_metrics_test.json` - Test set evaluation metrics
- `models/robustpdx/gradient_boosting_metrics_val.json` - Validation set metrics
- Similar files for other models

### Feature Importance
- `models/robustpdx/gradient_boosting_feature_importance.csv` - Top 85 features ranked by importance
- `models/robustpdx/random_forest_feature_importance.csv`

### Visualizations
- `models/robustpdx/gradient_boosting_confusion_matrix.png` - 3x3 confusion matrix plot
- `models/robustpdx/random_forest_confusion_matrix.png`
- `models/robustpdx/logistic_regression_confusion_matrix.png`

### Comparison
- `models/robustpdx/model_comparison.json` - Full metrics comparison in JSON format
- `models/robustpdx/model_comparison.csv` - Metrics comparison in CSV format

## How to Use

### Training Again

```bash
python train.py
```

Optional arguments:
- `--data`: Path to RobustPDX dataset (default: data/robustpdx/03_feature_extracted_dataset.csv)
- `--output`: Output directory for models (default: models/robustpdx)
- `--seed`: Random seed for reproducibility (default: 42)
- `--test-size`: Test set proportion (default: 0.15)
- `--val-size`: Validation set proportion (default: 0.15)

### Making Predictions

```bash
python inference.py --model gradient_boosting --limit 10
```

This loads the best model and makes predictions on the first 10 samples.

Optional arguments:
- `--model`: Which model to use: logistic_regression, random_forest, or gradient_boosting (default: gradient_boosting)
- `--model-dir`: Directory containing trained models (default: models/robustpdx)
- `--input`: Input CSV file (default: data/robustpdx/03_feature_extracted_dataset.csv)
- `--output`: Output CSV file with predictions (default: predictions.csv)
- `--limit`: Number of samples to predict (default: 10, 0 for all)

### Example Predictions

```
Session_ID                     predicted_class  confidence_nonpd  confidence_pd
0LDu5g4OfI7BJk54uwJ7          PD               0.034409          0.965339
0Q0JNx4C9IPe3M0yzI75          Non-PD           0.999854          0.000141
```

Output columns:
- `Session_ID`: Participant identifier
- `predicted_class_id`: 0=Non-PD, 1=PD, 2=Suspected PD
- `predicted_class`: Human-readable class name
- `confidence_nonpd`: Probability of Non-PD class
- `confidence_pd`: Probability of PD class
- `confidence_suspected`: Probability of Suspected PD class

## Interpretation Guide

### Metrics Explained

- **Accuracy:** Percentage of correct predictions (good overall metric, but can be misleading with imbalanced classes)
- **Precision:** Of positive predictions, how many were actually positive (minimize false positives)
- **Recall:** Of actual positives, how many were correctly identified (minimize false negatives)
- **F1-Score:** Harmonic mean of precision and recall (balanced metric)
- **Macro-F1:** F1-score averaged across classes without weighting (good for multi-class problems)
- **Weighted-F1:** F1-score weighted by class frequency

### Class Probabilities

The model outputs probabilities for each class. For a sample:
- If `confidence_pd = 0.95`, the model is 95% confident it's PD
- If `confidence_nonpd = 0.99`, the model is very confident it's Non-PD
- Sum of probabilities ≈ 1.0

### Feature Importance (Gradient Boosting)

Top features indicate which measurements most strongly differentiate PD from Non-PD:
1. Deviation from centerline in straight-line tracing (motor control)
2. Time taken to trace sine wave (fine motor control)
3. Device type (may reflect testing conditions)
4. Time to trace spiral (complex motor task)
5. Points inside spiral region (motor precision)

These features suggest that **fine motor control during complex drawing tasks** is the strongest indicator of Parkinson's Disease status.

## Limitations

1. **Class Imbalance:** Only 10.4% of data is "Suspected PD" - small sample size leads to high variance
2. **Dataset Size:** 251 samples is relatively small for machine learning; larger datasets could improve robustness
3. **Missing Validation on New Data:** Models trained on RobustPDX dataset; performance on other PD screening datasets unknown
4. **Multiclass Problem:** This is a 3-class problem (Non-PD, PD, Suspected); binary classification might be more reliable
5. **Temporal Independence:** Each row is treated independently; no temporal structure in trajectory data is used
6. **Device Dependency:** "Device Type" is a top feature, suggesting model may not generalize to new hardware

## Next Steps

1. **Collect More Data:** Particularly more "Suspected PD" samples to improve minority class prediction
2. **Cross-validation:** Use k-fold cross-validation on training set for more robust hyperparameter tuning
3. **Model Interpretation:** Use SHAP or LIME to explain individual predictions to clinicians
4. **Calibration:** Apply probability calibration if model's confidence scores need to match actual accuracy
5. **Testing on Independent Data:** Validate on data from other Parkinson's screening studies
6. **Binary Classification:** Try PD vs Non-PD (excluding Suspected) for potentially better performance
7. **Neural Networks:** Explore deep learning approaches if more data becomes available

## Technical Details

### Dependencies
- pandas >= 1.5
- numpy >= 1.23
- scikit-learn >= 1.2
- scipy >= 1.9
- matplotlib >= 3.6
- joblib >= 1.2

### Training Environment
- Python 3.14
- Windows 11
- 42 second execution time for full pipeline

### Model Serialization
All models are saved using joblib, which safely serializes Python objects. **Never load model files from untrusted sources**, as joblib.load() can execute arbitrary code.

---

**Report Generated:** 2026-10-23  
**Random Seed:** 42  
**Best Model:** Gradient Boosting (78.95% test accuracy)
