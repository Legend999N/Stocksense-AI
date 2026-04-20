"""
core/api.py — StockSense AI
==============================
The bridge layer between the Streamlit dashboard and the ML backend.

WHY THIS FILE EXISTS:
  Without this layer, app.py would import directly from models/train.py,
  features/technical.py etc. That creates tight coupling — if you rename
  a function in train.py, every dashboard component breaks.

  api.py provides 4 clean, stable functions the dashboard calls.
  The internals can change freely as long as these 4 signatures stay the same.

THE 4 CONTRACTS (agreed with frontend teammate on Day 1 planning):
  get_stock_chart_data(ticker, period)  → chart data + MA lines
  get_technical_signals(ticker)         → RSI, MACD, BB, Volume values
  get_prediction(ticker)               → direction, confidence, SHAP, explanation
  get_sentiment(ticker)                → headlines + scores + average

CACHING:
  All functions use @st.cache_data(ttl=3600) — results are cached for 1 hour.
  This means:
    • First call for AAPL: fetches data, trains nothing, returns in ~3-5s
    • Every subsequent call within 1 hour: instant (served from cache)
    • After 1 hour: automatically refreshes
  Without caching, every Streamlit widget interaction re-runs everything.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.technical import build_features, get_feature_columns
from features.sentiment import get_sentiment_score
from models.predict     import predict_ticker, load_model, calculate_risk_level
from models.explain     import get_local_explanation


# ─────────────────────────────────────────────
# 1. CHART DATA
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_chart_data(ticker: str, period: str = "1y") -> dict:
    """
    Fetch OHLCV data + computed MA lines for the candlestick chart.

    Args:
        ticker : Stock symbol e.g. 'AAPL'
        period : yfinance period string: '3mo', '6mo', '1y', '2y', '3y'

    Returns:
        dict:
          ohlcv   : pd.DataFrame with Open/High/Low/Close/Volume, DatetimeIndex
          sma_20  : pd.Series of 20-day SMA values
          sma_50  : pd.Series of 50-day SMA values
          ticker  : str
          period  : str
          error   : str or None

    The dashboard uses this to draw:
      - Candlestick (OHLCV)
      - SMA_20 line overlay (short-term trend)
      - SMA_50 line overlay (long-term trend)
      - Volume bar chart (subplot)
    """
    try:
        raw = yf.Ticker(ticker).history(period=period)

        if raw.empty:
            return {"error": f"No data found for '{ticker}'. Check the symbol.", "ticker": ticker}

        raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

        # Build full feature set to get the MA values
        # We need at least 50 rows for SMA_50 — add a buffer period first
        raw_extended = yf.Ticker(ticker).history(period="2y")
        raw_extended = raw_extended[['Open', 'High', 'Low', 'Close', 'Volume']]
        raw_extended.index = (raw_extended.index.tz_localize(None)
                              if raw_extended.index.tz else raw_extended.index)

        featured = build_features(raw_extended)

        # Filter to requested period
        cutoff = raw.index[0]
        featured = featured[featured.index >= cutoff]

        return {
            "ohlcv":   raw,
            "sma_20":  featured['SMA_20'],
            "sma_50":  featured['SMA_50'],
            "ticker":  ticker,
            "period":  period,
            "error":   None,
        }

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


# ─────────────────────────────────────────────
# 2. TECHNICAL SIGNALS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_technical_signals(ticker: str) -> dict:
    """
    Compute the latest technical indicator values for the signals panel.

    Returns the most recent (today's) value for each indicator,
    plus 60 days of MACD histogram history for the bar chart.

    Returns:
        dict:
          rsi              : float (0-100)
          rsi_label        : "Overbought" | "Oversold" | "Neutral"
          macd             : float
          macd_signal      : float
          macd_histogram   : pd.Series (last 60 days, for bar chart)
          macd_bullish     : bool
          bb_upper         : float
          bb_lower         : float
          bb_middle        : float
          bb_position      : float (0=lower band, 1=upper band)
          bb_width         : float
          volume_ratio     : float
          atr_normalised   : float
          latest_close     : float
          latest_date      : str
          error            : str or None
    """
    try:
        raw = yf.Ticker(ticker).history(period="1y")
        if raw.empty:
            return {"error": f"No data for '{ticker}'"}

        raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
        raw.index = raw.index.tz_localize(None) if raw.index.tz else raw.index

        featured = build_features(raw)
        latest   = featured.iloc[-1]

        # RSI label
        rsi = latest['RSI']
        if rsi > 70:
            rsi_label = "Overbought"
        elif rsi < 30:
            rsi_label = "Oversold"
        else:
            rsi_label = "Neutral"

        return {
            "rsi":            round(float(rsi), 2),
            "rsi_label":      rsi_label,
            "macd":           round(float(latest['MACD']), 4),
            "macd_signal":    round(float(latest['MACD_signal']), 4),
            "macd_histogram": featured['MACD_histogram'].iloc[-60:],   # last 60 days
            "macd_bullish":   bool(latest['MACD_bullish']),
            "bb_upper":       round(float(latest['BB_upper']), 2),
            "bb_lower":       round(float(latest['BB_lower']), 2),
            "bb_middle":      round(float(latest['BB_middle']), 2),
            "bb_position":    round(float(latest['BB_position']), 4),
            "bb_width":       round(float(latest['BB_width']), 4),
            "volume_ratio":   round(float(latest['volume_ratio']), 2),
            "atr_normalised": round(float(latest['ATR_normalised']), 4),
            "latest_close":   round(float(latest['Close']), 2),
            "latest_date":    str(featured.index[-1].date()),
            "error":          None,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# 3. PREDICTION
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def get_prediction(ticker: str) -> dict:
    """
    Load the trained model and return a prediction for today.

    Wraps models/predict.py and models/explain.py.

    Returns:
        dict:
          direction        : "UP" | "DOWN"
          confidence       : float (0.5–1.0)
          risk_level       : "Low" | "Medium" | "High"
          explanation      : str (plain-English reason)
          shap_values      : dict {feature: shap_value} for waterfall chart
          top_positive     : list of (feature, value) tuples → pushed UP
          top_negative     : list of (feature, value) tuples → pushed DOWN
          model_accuracy   : float (from saved metrics)
          model_roc_auc    : float (from saved metrics)
          uses_sentiment   : bool
          error            : str or None
    """
    try:
        # Check if model exists
        model_found = False
        suffix = None
        for s in ["_sentiment", ""]:
            if os.path.exists(f"models/saved/{ticker}{s}_model.joblib"):
                model_found = True
                suffix = s
                break

        if not model_found:
            return {
                "error": (
                    f"No trained model found for {ticker}.\n"
                    f"Run python day2_pipeline.py first."
                )
            }

        # Get prediction
        pred = predict_ticker(ticker)

        # Get SHAP local explanation
        try:
            shap_result = get_local_explanation(ticker)
            shap_values  = shap_result['shap_values']
            top_positive = shap_result['top_positive']
            top_negative = shap_result['top_negative']
        except Exception:
            shap_values  = {}
            top_positive = []
            top_negative = []

        # Load saved metrics
        metrics_path = f"models/saved/{ticker}{suffix}_metrics.json"
        model_accuracy = None
        model_roc_auc  = None
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                m = json.load(f)
            model_accuracy = m.get('accuracy')
            model_roc_auc  = m.get('roc_auc')

        return {
            "direction":      pred['direction'],
            "confidence":     pred['confidence'],
            "risk_level":     pred['risk_level'],
            "explanation":    pred['explanation'],
            "shap_values":    shap_values,
            "top_positive":   top_positive,
            "top_negative":   top_negative,
            "model_accuracy": model_accuracy,
            "model_roc_auc":  model_roc_auc,
            "uses_sentiment": suffix == "_sentiment",
            "prediction_date": pred['prediction_date'],
            "error":          None,
        }

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# 4. SENTIMENT
# ─────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)  # 30 min cache (news changes faster)
def get_sentiment(ticker: str) -> dict:
    """
    Fetch and score news headlines for the ticker.

    Returns:
        dict:
          headlines      : list of {text, polarity, label} dicts
          average_score  : float (-1.0 to +1.0)
          overall_label  : "Positive" | "Negative" | "Neutral"
          positive_count : int
          negative_count : int
          neutral_count  : int
          headline_count : int
          error          : str or None
    """
    try:
        result = get_sentiment_score(ticker, max_headlines=8)
        result['error'] = None
        return result
    except Exception as e:
        return {
            "headlines":      [],
            "average_score":  0.0,
            "overall_label":  "Neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count":  0,
            "headline_count": 0,
            "error":          str(e),
        }