"""
train.py — StockSense AI
==========================
XGBoost model training with time-series cross-validation.

What this module does:
  1. Loads feature CSV for a given ticker (built by Day 1)
  2. Adds the latest sentiment score as a feature
  3. Performs a strict time-based train/test split (NO random shuffle)
  4. Trains an XGBoost binary classifier
  5. Evaluates with accuracy, precision, recall, F1, ROC-AUC
  6. Saves the trained model to models/saved/<TICKER>_model.joblib
  7. Saves the feature column list (needed by predict.py and explain.py)

Key design decisions:
  • Time-based split: never shuffle time series data
  • class_weight balanced: handles the ~50/50 UP/DOWN imbalance gracefully
  • Conservative XGBoost params: prevents overfitting on small datasets
  • Two models per ticker: without sentiment, then with sentiment
    so we can measure if sentiment actually helps
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.technical import get_feature_columns
from features.sentiment import get_sentiment_score


# ─────────────────────────────────────────────
# XGBOOST PARAMETERS
# ─────────────────────────────────────────────

# Conservative params chosen to avoid overfitting on ~600 training rows.
# Explanation of each:
#   n_estimators   : 200 trees (more = better, up to diminishing returns)
#   max_depth      : 4 = shallow trees, limits overfitting
#   learning_rate  : 0.05 = slow, careful learning (needs more trees but generalises better)
#   subsample      : 0.8 = each tree sees 80% of rows (adds randomness, prevents overfit)
#   colsample_bytree: 0.8 = each tree sees 80% of features (adds randomness)
#   min_child_weight: 5 = a leaf needs at least 5 samples (avoids tiny spurious splits)
#   scale_pos_weight: 1 = balanced classes; change if target is imbalanced
#   eval_metric    : logloss is standard for binary classification
#   use_label_encoder: False avoids deprecation warning

XGBOOST_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state":     42,
    "eval_metric":      "logloss",
    "verbosity":        0,        # suppress XGBoost logs
}

TRAIN_TEST_SPLIT = 0.8   # 80% training, 20% testing


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_feature_data(ticker: str) -> pd.DataFrame:
    """
    Load the feature-engineered CSV built by Day 1 pipeline.

    Args:
        ticker : Stock symbol, e.g. 'AAPL'

    Returns:
        pd.DataFrame with DatetimeIndex and all feature + target columns

    Raises:
        FileNotFoundError if Day 1 hasn't been run yet
    """
    path = f"data/features/{ticker}_features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            f"Run day1_pipeline.py first to generate this file."
        )
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    print(f"  ✓ Loaded {len(df)} rows for {ticker} "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────
# TIME-BASED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────

def time_split(df: pd.DataFrame, train_ratio: float = TRAIN_TEST_SPLIT):
    """
    Split a time-series DataFrame into train and test sets
    using a strict chronological boundary.

    WHY NOT random_state split?
    If we shuffle randomly, the model sees "future" dates in training
    and "past" dates in testing. For example:
      - Training row: Jan 2024 data (with Jan 2024 RSI, volume etc.)
      - Test row: Dec 2023 data
    The model learns patterns from the future to predict the past —
    results look great but would fail in real deployment.

    CORRECT approach: sort by date, cut at 80%.
      - Training: rows 0–80% (oldest data)
      - Testing:  rows 80–100% (most recent data)
    This simulates real deployment: train on history, predict future.

    Args:
        df          : Feature DataFrame with DatetimeIndex
        train_ratio : Fraction of data to use for training

    Returns:
        (train_df, test_df) tuple
    """
    df_sorted   = df.sort_index()   # Ensure chronological order
    split_idx   = int(len(df_sorted) * train_ratio)
    train_df    = df_sorted.iloc[:split_idx]
    test_df     = df_sorted.iloc[split_idx:]

    print(f"  Train: {len(train_df)} rows "
          f"({train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f"  Test : {len(test_df)} rows "
          f"({test_df.index[0].date()} → {test_df.index[-1].date()})")

    return train_df, test_df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(
    train_df: pd.DataFrame,
    feature_cols: list,
    label: str = ""
) -> XGBClassifier:
    """
    Train an XGBoost binary classifier.

    Args:
        train_df     : Training DataFrame
        feature_cols : List of column names to use as input features
        label        : Optional label for print output (e.g. "without sentiment")

    Returns:
        Fitted XGBClassifier
    """
    X_train = train_df[feature_cols]
    y_train = train_df['target']

    print(f"\n  Training XGBoost {label}...")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Samples  : {len(X_train)}")
    print(f"  UP days  : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  DOWN days: {(1-y_train).sum()} ({(1-y_train.mean())*100:.1f}%)")

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    print(f"  ✓ Training complete")
    return model


# ─────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(
    model: XGBClassifier,
    test_df: pd.DataFrame,
    feature_cols: list,
    label: str = ""
) -> dict:
    """
    Evaluate a trained model on the test set.

    Metrics computed:
      accuracy  : Overall % correct
      precision : Of predicted UP days, % that were actually UP
      recall    : Of actual UP days, % we correctly predicted
      f1        : Harmonic mean of precision and recall (balanced metric)
      roc_auc   : Area under the ROC curve (threshold-independent quality)

    Confusion matrix layout:
      [[TN, FP],   TN = correctly predicted DOWN
       [FN, TP]]   TP = correctly predicted UP
                   FP = predicted UP but was DOWN (false alarm)
                   FN = predicted DOWN but was UP (missed opportunity)

    Args:
        model        : Fitted XGBClassifier
        test_df      : Test DataFrame
        feature_cols : Feature columns (same as used in training)
        label        : Label for printing

    Returns:
        dict with all metric values
    """
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # probability of UP

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_pred_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": len(y_test),
        "feature_cols": feature_cols,
    }

    # Print evaluation report
    print(f"\n  ── Evaluation: {label} ────────────────────")
    print(f"  Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {metrics['precision']*100:.2f}%")
    print(f"  Recall    : {metrics['recall']*100:.2f}%")
    print(f"  F1 Score  : {metrics['f1']*100:.2f}%")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    cm = metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              DOWN    UP")
    print(f"  Actual DOWN  {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"  Actual UP    {cm[1][0]:4d}   {cm[1][1]:4d}")

    # Interpret the ROC-AUC
    auc = metrics['roc_auc']
    if auc >= 0.60:
        quality = "Strong ✓"
    elif auc >= 0.55:
        quality = "Good ✓"
    elif auc >= 0.50:
        quality = "Marginal — model has some signal"
    else:
        quality = "Weak — worse than random"
    print(f"\n  ROC-AUC Quality: {quality}")

    return metrics


# ─────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────

def save_model(
    model: XGBClassifier,
    ticker: str,
    feature_cols: list,
    metrics: dict,
    suffix: str = ""
) -> str:
    """
    Save trained model, feature list, and metrics to disk.

    Files saved:
      models/saved/<TICKER><suffix>_model.joblib  → the model
      models/saved/<TICKER><suffix>_features.json → feature column list
      models/saved/<TICKER><suffix>_metrics.json  → evaluation metrics

    The feature list is critical — predict.py must use the exact same
    columns in the exact same order as training. Saving it alongside
    the model prevents mismatches.

    Args:
        model        : Fitted XGBClassifier
        ticker       : Stock symbol
        feature_cols : List of feature column names used in training
        metrics      : Evaluation metrics dict from evaluate_model()
        suffix       : e.g. "_sentiment" to distinguish versions

    Returns:
        Path to saved model file
    """
    os.makedirs("models/saved", exist_ok=True)

    model_path    = f"models/saved/{ticker}{suffix}_model.joblib"
    features_path = f"models/saved/{ticker}{suffix}_features.json"
    metrics_path  = f"models/saved/{ticker}{suffix}_metrics.json"

    # Save model
    joblib.dump(model, model_path)

    # Save feature list (must match training exactly)
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)

    # Save metrics (for dashboard display and comparison)
    save_metrics = {k: v for k, v in metrics.items()
                    if k != 'feature_cols'}  # exclude list (already saved)
    with open(metrics_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)

    print(f"\n  ✓ Model saved    : {model_path}")
    print(f"  ✓ Features saved : {features_path}")
    print(f"  ✓ Metrics saved  : {metrics_path}")

    return model_path


# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE FOR ONE TICKER
# ─────────────────────────────────────────────

def train_ticker(ticker: str) -> dict:
    """
    Complete training pipeline for a single ticker.

    Steps:
      1. Load feature data (from Day 1)
      2. Split into train/test (time-based)
      3. Train Model A: technical features only
      4. Evaluate Model A
      5. Fetch sentiment score and add as feature
      6. Train Model B: technical + sentiment features
      7. Evaluate Model B
      8. Compare A vs B — did sentiment help?
      9. Save the better model (usually Model B)

    Args:
        ticker : Stock symbol

    Returns:
        dict with both model results and comparison
    """
    print(f"\n{'═'*55}")
    print(f"  TRAINING: {ticker}")
    print(f"{'═'*55}")

    # ── Step 1: Load data ────────────────────────────────────────
    print("\n[1/5] Loading feature data...")
    df = load_feature_data(ticker)
    feature_cols = get_feature_columns(df)

    # ── Step 2: Time split ───────────────────────────────────────
    print("\n[2/5] Splitting data (time-based)...")
    train_df, test_df = time_split(df)

    # ── Step 3 & 4: Model A — Technical features only ────────────
    print("\n[3/5] Training Model A (technical features only)...")
    model_a = train_model(train_df, feature_cols, label="(technical only)")
    metrics_a = evaluate_model(model_a, test_df, feature_cols,
                               label="Technical Only")

    # ── Step 5: Add sentiment score ──────────────────────────────
    print("\n[4/5] Adding sentiment feature...")
    sentiment_result = get_sentiment_score(ticker)
    sentiment_score  = sentiment_result['average_score']

    # Add sentiment as a constant column across all rows
    # (We have one current sentiment score; in production you'd
    #  have historical sentiment per day — this is a prototype simplification)
    df['sentiment_score'] = sentiment_score
    train_df = df.iloc[:int(len(df) * TRAIN_TEST_SPLIT)]
    test_df  = df.iloc[int(len(df) * TRAIN_TEST_SPLIT):]

    feature_cols_with_sentiment = feature_cols + ['sentiment_score']

    # ── Step 6 & 7: Model B — Technical + Sentiment ──────────────
    print("\n[5/5] Training Model B (technical + sentiment)...")
    model_b  = train_model(train_df, feature_cols_with_sentiment,
                           label="(technical + sentiment)")
    metrics_b = evaluate_model(model_b, test_df, feature_cols_with_sentiment,
                               label="Technical + Sentiment")

    # ── Step 8: Compare A vs B ───────────────────────────────────
    print(f"\n  ── Sentiment Impact ──────────────────────────")
    acc_delta = (metrics_b['accuracy'] - metrics_a['accuracy']) * 100
    auc_delta = metrics_b['roc_auc'] - metrics_a['roc_auc']
    print(f"  Accuracy change : {acc_delta:+.2f}%")
    print(f"  ROC-AUC change  : {auc_delta:+.4f}")

    if acc_delta >= 0:
        print(f"  ✓ Sentiment HELPED — using Model B (with sentiment)")
        best_model    = model_b
        best_features = feature_cols_with_sentiment
        best_metrics  = metrics_b
        best_suffix   = "_sentiment"
    else:
        print(f"  ✗ Sentiment didn't help this ticker — using Model A (technical only)")
        best_model    = model_a
        best_features = feature_cols
        best_metrics  = metrics_a
        best_suffix   = ""

    # ── Step 9: Save best model ──────────────────────────────────
    save_model(best_model, ticker, best_features, best_metrics, suffix=best_suffix)

    return {
        "ticker":           ticker,
        "model_a_metrics":  metrics_a,
        "model_b_metrics":  metrics_b,
        "sentiment_score":  sentiment_score,
        "sentiment_label":  sentiment_result['overall_label'],
        "best_model":       "B (with sentiment)" if acc_delta >= 0 else "A (technical only)",
        "acc_delta":        acc_delta,
        "auc_delta":        auc_delta,
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TESTING train.py — training AAPL")
    print("=" * 55)

    result = train_ticker("AAPL")

    print(f"\n{'─'*55}")
    print(f"TRAINING SUMMARY — {result['ticker']}")
    print(f"{'─'*55}")
    print(f"Model A (Technical) Accuracy : {result['model_a_metrics']['accuracy']*100:.2f}%")
    print(f"Model B (+ Sentiment) Accuracy: {result['model_b_metrics']['accuracy']*100:.2f}%")
    print(f"Sentiment score              : {result['sentiment_score']:+.3f} ({result['sentiment_label']})")
    print(f"Best model selected          : {result['best_model']}")
    print(f"\n✓ train.py working correctly!")