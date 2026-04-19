"""
explain.py — StockSense AI
============================
SHAP-based explainability for the XGBoost model.

What SHAP does:
  For every single prediction, SHAP calculates each feature's
  contribution to pushing the model toward UP or toward DOWN.

  Example output for an AAPL UP prediction:
    RSI_normalised       : +0.82  ← strongly pushed toward UP
    MACD_bullish         : +0.61  ← pushed toward UP
    BB_position          : +0.23  ← slightly toward UP
    SMA_crossover        : -0.15  ← pushed toward DOWN
    volume_ratio         : -0.08  ← slightly toward DOWN
    ...

  Positive SHAP value → feature pushed prediction toward UP
  Negative SHAP value → feature pushed prediction toward DOWN

Why this matters:
  Without SHAP, the model is a black box. With SHAP, a user can see:
  "RSI being at 72 was the main reason the model predicted DOWN."
  This is our key differentiator from other stock tools.

Two outputs this module produces:
  1. Global feature importance: which features matter most on average
     across all predictions (for model insight / portfolio slide)
  2. Local explanation: SHAP values for ONE specific prediction
     (for the dashboard's "Why this prediction?" panel)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.technical import build_features, get_feature_columns
from features.sentiment import get_sentiment_score


# ─────────────────────────────────────────────
# LOAD MODEL AND DATA
# ─────────────────────────────────────────────

def load_model_and_data(ticker: str):
    """
    Load the saved model, feature list, and test data for a ticker.

    Args:
        ticker : Stock symbol

    Returns:
        (model, feature_cols, X_test) tuple
        X_test is the last 20% of data (test set) as a DataFrame
    """
    # Find the right model file (sentiment version preferred)
    model = None
    feature_cols = None

    for suffix in ["_sentiment", ""]:
        model_path    = f"models/saved/{ticker}{suffix}_model.joblib"
        features_path = f"models/saved/{ticker}{suffix}_features.json"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            with open(features_path, 'r') as f:
                feature_cols = json.load(f)
            print(f"  ✓ Loaded: {model_path}")
            break

    if model is None:
        raise FileNotFoundError(
            f"No model found for {ticker}. Run models/train.py first."
        )

    # Load and rebuild the feature dataset
    features_csv = f"data/features/{ticker}_features.csv"
    df = pd.read_csv(features_csv, index_col='Date', parse_dates=True)

    # Add sentiment column if the model uses it
    if 'sentiment_score' in feature_cols:
        sentiment = get_sentiment_score(ticker)
        df['sentiment_score'] = sentiment['average_score']

    # Use the test set (last 20%) for global SHAP analysis
    split_idx = int(len(df) * 0.8)
    X_test = df.iloc[split_idx:][feature_cols]

    return model, feature_cols, X_test


# ─────────────────────────────────────────────
# GLOBAL FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def get_global_importance(ticker: str) -> dict:
    """
    Compute SHAP-based global feature importance across the test set.

    This tells us: on average, which features drive predictions most?
    Used for the "Model Insights" section of the dashboard and
    for understanding whether our feature engineering worked.

    Args:
        ticker : Stock symbol

    Returns:
        dict with:
          feature_names         : list of feature names
          mean_abs_shap_values  : list of mean |SHAP| per feature
          top_features          : top 10 features sorted by importance
    """
    print(f"  Computing global SHAP importance for {ticker}...")

    model, feature_cols, X_test = load_model_and_data(ticker)

    # Create SHAP TreeExplainer — optimised for tree-based models (XGBoost)
    # Much faster than the generic KernelExplainer
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP value per feature
    # |SHAP| = importance regardless of direction
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create sorted importance list
    importance_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print(f"  ✓ Top 5 most important features:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']:<30} {row['importance']:.4f}")

    return {
        'feature_names':        list(importance_df['feature']),
        'mean_abs_shap_values': list(importance_df['importance'].round(4)),
        'top_features':         importance_df.head(10).to_dict('records'),
    }


# ─────────────────────────────────────────────
# LOCAL EXPLANATION FOR ONE PREDICTION
# ─────────────────────────────────────────────

def get_local_explanation(ticker: str, top_n: int = 10) -> dict:
    """
    Compute SHAP values for the LATEST prediction (the one shown
    on the dashboard).

    This answers: "For today's AAPL prediction, why did the model
    say UP? Which features contributed most?"

    Args:
        ticker : Stock symbol
        top_n  : Number of features to return (default 10)

    Returns:
        dict with:
          shap_values     : dict {feature_name: shap_value}
                            positive = pushed toward UP
                            negative = pushed toward DOWN
          base_value      : the model's default prediction before any features
          top_positive    : top features pushing toward UP
          top_negative    : top features pushing toward DOWN
          expected_value  : base_value (alias for clarity)
    """
    print(f"  Computing local SHAP explanation for {ticker}...")

    model, feature_cols, _ = load_model_and_data(ticker)

    # Get latest feature row (same as predict.py)
    raw = yf.Ticker(ticker).history(period="6mo")
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

    featured = build_features(raw)

    # Add sentiment if needed
    if 'sentiment_score' in feature_cols:
        sentiment = get_sentiment_score(ticker)
        featured['sentiment_score'] = sentiment['average_score']

    X_latest = featured.iloc[[-1]][feature_cols]

    # SHAP explanation for this single row
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_latest)

    # shap_values is shape (1, n_features) — get the first (only) row
    values_1d    = shap_values[0]
    base_value   = float(explainer.expected_value)

    # Build feature → SHAP value mapping
    shap_dict = {
        feature_cols[i]: round(float(values_1d[i]), 4)
        for i in range(len(feature_cols))
    }

    # Sort by absolute SHAP value (most impactful first)
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    # Separate positive (toward UP) and negative (toward DOWN)
    top_positive = [(k, v) for k, v in sorted_shap if v > 0][:5]
    top_negative = [(k, v) for k, v in sorted_shap if v < 0][:5]

    print(f"  ✓ Base value (expected): {base_value:.4f}")
    print(f"  Top positive contributors (→ UP):")
    for feat, val in top_positive[:3]:
        print(f"      {feat:<30} {val:+.4f}")
    print(f"  Top negative contributors (→ DOWN):")
    for feat, val in top_negative[:3]:
        print(f"      {feat:<30} {val:+.4f}")

    return {
        'shap_values':     shap_dict,
        'sorted_shap':     sorted_shap[:top_n],
        'base_value':      base_value,
        'expected_value':  base_value,
        'top_positive':    top_positive,
        'top_negative':    top_negative,
        'feature_values':  {
            col: round(float(X_latest[col].iloc[0]), 4)
            for col in feature_cols
        },
    }


# ─────────────────────────────────────────────
# SAVE SHAP IMPORTANCE TO JSON
# ─────────────────────────────────────────────

def save_global_importance(ticker: str, importance: dict) -> None:
    """
    Save global feature importance to disk so the dashboard
    can load it without recomputing SHAP every time.

    Args:
        ticker     : Stock symbol
        importance : Output of get_global_importance()
    """
    os.makedirs("models/saved", exist_ok=True)
    path = f"models/saved/{ticker}_shap_importance.json"

    # Convert to serialisable format
    save_data = {
        'top_features': importance['top_features'],
        'all_features': [
            {'feature': name, 'importance': val}
            for name, val in zip(
                importance['feature_names'],
                importance['mean_abs_shap_values']
            )
        ]
    }

    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"  ✓ SHAP importance saved: {path}")


# ─────────────────────────────────────────────
# FULL EXPLAIN PIPELINE FOR ONE TICKER
# ─────────────────────────────────────────────

def run_explain_pipeline(ticker: str) -> dict:
    """
    Run the complete explainability pipeline for one ticker.

    Steps:
      1. Compute global feature importance (across test set)
      2. Save importance to disk
      3. Compute local explanation for latest prediction

    Args:
        ticker : Stock symbol

    Returns:
        dict with both global and local SHAP results
    """
    print(f"\n{'─'*55}")
    print(f"  SHAP EXPLAINABILITY: {ticker}")
    print(f"{'─'*55}")

    print("\n[1/2] Computing global feature importance...")
    global_imp = get_global_importance(ticker)
    save_global_importance(ticker, global_imp)

    print("\n[2/2] Computing local explanation for latest prediction...")
    local_exp = get_local_explanation(ticker)

    return {
        'ticker':      ticker,
        'global':      global_imp,
        'local':       local_exp,
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TESTING explain.py")
    print("=" * 55)

    result = run_explain_pipeline("AAPL")

    print(f"\n{'─'*55}")
    print(f"EXPLAIN RESULT SUMMARY")
    print(f"{'─'*55}")

    print("\nTop 5 Global Features (most important on average):")
    for feat in result['global']['top_features'][:5]:
        print(f"  {feat['feature']:<30} {feat['importance']:.4f}")

    print("\nLocal Explanation (today's prediction):")
    print(f"  Base value (model default): {result['local']['base_value']:.4f}")
    print(f"  Top 3 features pushing UP:")
    for feat, val in result['local']['top_positive'][:3]:
        print(f"    {feat:<30} {val:+.4f}")
    print(f"  Top 3 features pushing DOWN:")
    for feat, val in result['local']['top_negative'][:3]:
        print(f"    {feat:<30} {val:+.4f}")

    print(f"\n✓ explain.py working correctly!")