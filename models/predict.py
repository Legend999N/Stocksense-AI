"""
predict.py — StockSense AI
============================
Inference module — loads a saved model and makes predictions
on new data for any ticker.

What this module does:
  1. Loads the saved model and feature list from disk
  2. Builds a fresh feature row for the latest available data
  3. Returns: direction (UP/DOWN), confidence score, risk level

This is what the Streamlit dashboard calls on Day 3.
It's kept separate from train.py so the dashboard never
accidentally re-trains the model on every page load.

Usage:
    from models.predict import predict_ticker
    result = predict_ticker("AAPL")
    print(result['direction'])    # "UP" or "DOWN"
    print(result['confidence'])   # e.g. 0.71
    print(result['risk_level'])   # "Low", "Medium", "High"
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.technical import build_features
from features.sentiment import get_sentiment_score


# ─────────────────────────────────────────────
# LOAD SAVED MODEL
# ─────────────────────────────────────────────

def load_model(ticker: str):
    """
    Load the saved model and its feature list from disk.

    Tries to load the sentiment version first (_sentiment_model.joblib),
    falls back to the technical-only version if not found.

    Args:
        ticker : Stock symbol

    Returns:
        (model, feature_cols) tuple

    Raises:
        FileNotFoundError if no model exists for this ticker
    """
    # Try sentiment version first (usually the better model)
    for suffix in ["_sentiment", ""]:
        model_path    = f"models/saved/{ticker}{suffix}_model.joblib"
        features_path = f"models/saved/{ticker}{suffix}_features.json"

        if os.path.exists(model_path) and os.path.exists(features_path):
            model = joblib.load(model_path)
            with open(features_path, 'r') as f:
                feature_cols = json.load(f)
            print(f"  ✓ Loaded model: {model_path} ({len(feature_cols)} features)")
            return model, feature_cols

    raise FileNotFoundError(
        f"No trained model found for {ticker}.\n"
        f"Run models/train.py first to train the model."
    )


# ─────────────────────────────────────────────
# BUILD LATEST FEATURE ROW
# ─────────────────────────────────────────────

def get_latest_features(ticker: str, feature_cols: list) -> pd.DataFrame:
    """
    Fetch the most recent stock data and build a feature row
    for today's prediction.

    We download 6 months of data (enough for all rolling windows
    like SMA_50 and MACD to be fully computed) but only use the
    LAST ROW for prediction.

    Args:
        ticker       : Stock symbol
        feature_cols : Feature columns the model expects

    Returns:
        Single-row DataFrame with all required features
    """
    print(f"  Fetching latest data for {ticker}...")

    # Download enough history to compute all indicators correctly
    # SMA_50 needs 50 days min, MACD needs 26 + 9 = 35 days,
    # so 6 months (~126 trading days) is more than enough
    raw = yf.Ticker(ticker).history(period="6mo")

    if raw.empty:
        raise ValueError(f"Could not fetch recent data for {ticker}")

    # Clean the data (same as fetch_data.py)
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

    # Build all features using the Day 1 pipeline
    featured = build_features(raw)

    # Take only the most recent row (today's data)
    latest_row = featured.iloc[[-1]]  # Keep as DataFrame (not Series)

    # Add sentiment if the model uses it
    if 'sentiment_score' in feature_cols:
        sentiment = get_sentiment_score(ticker)
        latest_row = latest_row.copy()
        latest_row['sentiment_score'] = sentiment['average_score']

    # Verify all required features are present
    missing = [col for col in feature_cols if col not in latest_row.columns]
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    return latest_row[feature_cols]


# ─────────────────────────────────────────────
# RISK LEVEL CALCULATOR
# ─────────────────────────────────────────────

def calculate_risk_level(ticker: str) -> str:
    """
    Calculate stock risk level based on ATR (Average True Range)
    as a percentage of price — the 'ATR_normalised' feature.

    ATR_normalised = daily price range / closing price
    This tells us: on average, how much does this stock move daily
    as a % of its price?

    Thresholds:
      < 1.5%  → Low risk    (stable stocks like MSFT, JNJ)
      1.5–3%  → Medium risk (typical for large-cap tech)
      > 3%    → High risk   (volatile stocks like TSLA, crypto)

    Args:
        ticker : Stock symbol

    Returns:
        "Low" | "Medium" | "High"
    """
    try:
        raw = yf.Ticker(ticker).history(period="3mo")
        if raw.empty:
            return "Medium"  # default if can't calculate

        raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

        featured = build_features(raw)
        atr_norm = featured['ATR_normalised'].iloc[-1]

        if atr_norm < 0.015:
            return "Low"
        elif atr_norm < 0.03:
            return "Medium"
        else:
            return "High"

    except Exception:
        return "Medium"


# ─────────────────────────────────────────────
# GENERATE NATURAL LANGUAGE EXPLANATION
# ─────────────────────────────────────────────

def generate_explanation(
    direction: str,
    confidence: float,
    rsi: float,
    macd_bullish: int,
    bb_position: float,
    sentiment_score: float,
    volume_ratio: float,
) -> str:
    """
    Generate a plain-English explanation of the prediction.

    This is one of our key differentiators — explaining WHY
    the model made its decision in terms a non-technical user
    can understand.

    Args:
        direction       : "UP" or "DOWN"
        confidence      : Model confidence (0.5 to 1.0)
        rsi             : RSI value (0-100)
        macd_bullish    : 1 if MACD > Signal, 0 otherwise
        bb_position     : Bollinger Band position (0-1, >1 = above upper band)
        sentiment_score : Sentiment score (-1 to +1)
        volume_ratio    : Today's volume vs 20-day average

    Returns:
        Human-readable explanation string
    """
    signals = []

    # RSI interpretation
    if rsi > 70:
        signals.append(f"RSI at {rsi:.0f} signals overbought conditions (bearish)")
    elif rsi < 30:
        signals.append(f"RSI at {rsi:.0f} signals oversold conditions (bullish)")
    else:
        signals.append(f"RSI at {rsi:.0f} is in neutral territory")

    # MACD interpretation
    if macd_bullish:
        signals.append("MACD is in a bullish crossover (upward momentum)")
    else:
        signals.append("MACD is in a bearish crossover (downward momentum)")

    # Bollinger Bands interpretation
    if bb_position > 1.0:
        signals.append("price is above the upper Bollinger Band (extended)")
    elif bb_position < 0.0:
        signals.append("price is below the lower Bollinger Band (potential bounce)")
    elif bb_position > 0.8:
        signals.append("price is near the upper Bollinger Band")
    elif bb_position < 0.2:
        signals.append("price is near the lower Bollinger Band")

    # Volume interpretation
    if volume_ratio > 1.5:
        signals.append(f"volume is {volume_ratio:.1f}x average (high conviction)")
    elif volume_ratio < 0.7:
        signals.append(f"volume is low ({volume_ratio:.1f}x average, low conviction)")

    # Sentiment interpretation
    if sentiment_score > 0.1:
        signals.append(f"news sentiment is positive ({sentiment_score:+.2f})")
    elif sentiment_score < -0.1:
        signals.append(f"news sentiment is negative ({sentiment_score:+.2f})")

    # Build the explanation
    conf_pct  = confidence * 100
    signal_str = "; ".join(signals[:3])  # limit to top 3 for readability

    explanation = (
        f"Model predicts {direction} with {conf_pct:.0f}% confidence. "
        f"Key signals: {signal_str}."
    )

    return explanation


# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_ticker(ticker: str) -> dict:
    """
    Make a prediction for a single ticker using the saved model.

    This is the main function called by the Streamlit dashboard.

    Args:
        ticker : Stock symbol

    Returns:
        dict with:
          ticker          : str
          direction       : "UP" or "DOWN"
          confidence      : float (0.5–1.0, model's certainty)
          risk_level      : "Low" | "Medium" | "High"
          explanation     : str (plain-English reason)
          latest_rsi      : float (for dashboard display)
          latest_macd_bullish: int
          latest_bb_position : float
          sentiment_score : float
          prediction_date : str (the date of the last data point)
    """
    print(f"\n  Making prediction for {ticker}...")

    # Load model and features
    model, feature_cols = load_model(ticker)

    # Get latest features
    X_latest = get_latest_features(ticker, feature_cols)

    # Make prediction
    direction_code = model.predict(X_latest)[0]
    probabilities  = model.predict_proba(X_latest)[0]

    direction  = "UP" if direction_code == 1 else "DOWN"
    # Confidence = probability of the predicted class
    confidence = float(probabilities[direction_code])

    # Calculate risk level
    risk_level = calculate_risk_level(ticker)

    # Extract key indicator values for the explanation
    # (these come from the feature row, not the model)
    latest_features_df = get_latest_features(ticker, feature_cols)

    def safe_get(col, default=0.0):
        return float(latest_features_df[col].iloc[0]) if col in latest_features_df.columns else default

    rsi          = safe_get('RSI', 50.0)
    macd_bullish = int(safe_get('MACD_bullish', 0))
    bb_position  = safe_get('BB_position', 0.5)
    volume_ratio = safe_get('volume_ratio', 1.0)
    sentiment    = safe_get('sentiment_score', 0.0)

    # Generate explanation
    explanation = generate_explanation(
        direction, confidence, rsi, macd_bullish,
        bb_position, sentiment, volume_ratio
    )

    # Get prediction date (last date in the data)
    raw = yf.Ticker(ticker).history(period="5d")
    prediction_date = str(raw.index[-1].date()) if not raw.empty else "Unknown"

    result = {
        "ticker":              ticker,
        "direction":           direction,
        "confidence":          round(confidence, 4),
        "risk_level":          risk_level,
        "explanation":         explanation,
        "latest_rsi":          round(rsi, 2),
        "latest_macd_bullish": macd_bullish,
        "latest_bb_position":  round(bb_position, 4),
        "sentiment_score":     round(sentiment, 4),
        "volume_ratio":        round(volume_ratio, 4),
        "prediction_date":     prediction_date,
    }

    print(f"  ✓ Prediction: {direction} | "
          f"Confidence: {confidence*100:.1f}% | "
          f"Risk: {risk_level}")

    return result


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TESTING predict.py")
    print("=" * 55)

    result = predict_ticker("AAPL")

    print(f"\n{'─'*55}")
    print(f"PREDICTION RESULT")
    print(f"{'─'*55}")
    print(f"  Ticker         : {result['ticker']}")
    print(f"  Direction      : {result['direction']}")
    print(f"  Confidence     : {result['confidence']*100:.1f}%")
    print(f"  Risk Level     : {result['risk_level']}")
    print(f"  RSI            : {result['latest_rsi']}")
    print(f"  MACD Bullish   : {bool(result['latest_macd_bullish'])}")
    print(f"  BB Position    : {result['latest_bb_position']:.3f}")
    print(f"  Sentiment      : {result['sentiment_score']:+.3f}")
    print(f"  Date           : {result['prediction_date']}")
    print(f"\n  Explanation:")
    print(f"  {result['explanation']}")
    print(f"\n✓ predict.py working correctly!")