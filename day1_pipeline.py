"""
day1_pipeline.py — StockSense AI
==================================
The master runner for Day 1. Ties together fetch_data.py and technical.py.

Run this file to complete all of Day 1:
    python day1_pipeline.py

What it does:
    1. Fetches 3 years of OHLCV data for 5 stocks
    2. Engineers all technical features
    3. Creates the binary target variable
    4. Saves feature-rich CSVs to data/features/
    5. Prints a detailed summary report

Expected output:
    data/
      raw/
        AAPL_raw.csv
        TSLA_raw.csv
        GOOGL_raw.csv
        MSFT_raw.csv
        AMZN_raw.csv
      features/
        AAPL_features.csv
        TSLA_features.csv
        ... etc

Time to run: ~60–90 seconds (most of that is downloading from Yahoo Finance)
"""

import os
import sys
import pandas as pd

# ── Import our modules ───────────────────────────────────────────────────────
# Add project root to path so imports work regardless of where you run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetch_data import fetch_stock_data, load_raw_data
from features.technical import build_features, get_feature_columns


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Stocks to process — good mix of sectors for a portfolio demo
TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# Period of historical data to fetch
# 3y ≈ 750 trading days → solid dataset for ML training
DATA_PERIOD = "3y"

# If True, re-download even if raw CSV already exists
FORCE_REDOWNLOAD = False


# ─────────────────────────────────────────────
# SINGLE TICKER PIPELINE
# ─────────────────────────────────────────────

def run_pipeline_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Full Day 1 pipeline for a single ticker.

    Steps:
        1. Fetch or load raw OHLCV data
        2. Engineer all features
        3. Save feature DataFrame to disk
        4. Return the DataFrame

    Args:
        ticker : Stock symbol, e.g. 'AAPL'

    Returns:
        pd.DataFrame with all features and target variable
    """
    print(f"\n{'═'*55}")
    print(f"  Processing: {ticker}")
    print(f"{'═'*55}")

    # ── Step 1: Get raw data ─────────────────────────────────────────────────
    raw_path = f"data/raw/{ticker}_raw.csv"
    if not FORCE_REDOWNLOAD and os.path.exists(raw_path):
        print(f"\n[1/3] Loading cached raw data from {raw_path}...")
        df_raw = load_raw_data(ticker)
        print(f"  ✓ Loaded {len(df_raw)} rows from cache")
    else:
        print(f"\n[1/3] Downloading raw OHLCV data from Yahoo Finance...")
        df_raw = fetch_stock_data(ticker, period=DATA_PERIOD, save=True)

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    print(f"\n[2/3] Engineering features...")
    df_features = build_features(df_raw)

    # ── Step 3: Save features ─────────────────────────────────────────────────
    print(f"\n[3/3] Saving feature dataset...")
    os.makedirs("data/features", exist_ok=True)
    output_path = f"data/features/{ticker}_features.csv"
    df_features.to_csv(output_path)
    print(f"  ✓ Saved to {output_path}")

    return df_features


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def print_ticker_summary(ticker: str, df: pd.DataFrame) -> None:
    """
    Print a detailed summary of the processed data for one ticker.
    This acts as a mini health-check before moving to Day 2.
    """
    feature_cols = get_feature_columns(df)

    print(f"\n{'─'*55}")
    print(f"  REPORT: {ticker}")
    print(f"{'─'*55}")
    print(f"  Date range   : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Trading days : {len(df)}")
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  NaN remaining: {df.isnull().sum().sum()} (should be 0)")

    # Target balance — important for understanding class imbalance
    up_days   = df['target'].sum()
    down_days = len(df) - up_days
    pct_up    = up_days / len(df) * 100

    print(f"\n  Target balance:")
    print(f"    UP   (1) : {up_days} days ({pct_up:.1f}%)")
    print(f"    DOWN (0) : {down_days} days ({100-pct_up:.1f}%)")

    if abs(pct_up - 50) > 10:
        print(f"  ⚠ Note: Significant class imbalance — consider using class_weight='balanced' in XGBoost")
    else:
        print(f"  ✓ Target is roughly balanced — good for training")

    # Latest indicators snapshot (useful for debugging)
    latest = df.iloc[-1]
    print(f"\n  Latest indicators ({df.index[-1].date()}):")
    print(f"    Close     : ${latest['Close']:.2f}")
    print(f"    RSI       : {latest['RSI']:.1f}  ({'OVERBOUGHT' if latest['RSI'] > 70 else 'OVERSOLD' if latest['RSI'] < 30 else 'neutral'})")
    print(f"    MACD      : {latest['MACD']:.4f}  ({'bullish' if latest['MACD_bullish'] else 'bearish'})")
    print(f"    BB pos    : {latest['BB_position']:.2f}  (0=lower band, 1=upper band)")
    print(f"    Volume    : {latest['volume_ratio']:.2f}x average")


def print_final_summary(results: dict) -> None:
    """Print the overall Day 1 completion report."""
    print(f"\n\n{'═'*55}")
    print(f"  DAY 1 COMPLETE ✓")
    print(f"{'═'*55}")
    print(f"  Stocks processed: {len(results)}")
    print(f"  Tickers         : {list(results.keys())}")
    print()

    # Show column count for the first successful ticker
    if results:
        sample_df   = list(results.values())[0]
        feature_cols = get_feature_columns(sample_df)
        print(f"  Features per stock: {len(feature_cols)}")
        print(f"\n  Full feature list:")
        for i, col in enumerate(feature_cols, 1):
            print(f"    {i:02d}. {col}")

    print(f"\n  Files saved:")
    print(f"    Raw data    → data/raw/<TICKER>_raw.csv")
    print(f"    Features    → data/features/<TICKER>_features.csv")
    print(f"\n  Ready for Day 2:")
    print(f"    • XGBoost model training (models/train.py)")
    print(f"    • Sentiment pipeline     (features/sentiment.py)")
    print(f"    • SHAP explainability    (models/explain.py)")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*55}")
    print(f"  STOCKSENSE AI — DAY 1 PIPELINE")
    print(f"  Data Collection + Feature Engineering")
    print(f"{'═'*55}")
    print(f"  Tickers : {TICKERS}")
    print(f"  Period  : {DATA_PERIOD}")
    print(f"  Re-fetch: {FORCE_REDOWNLOAD}")

    results = {}
    failed  = []

    for ticker in TICKERS:
        try:
            df = run_pipeline_for_ticker(ticker)
            print_ticker_summary(ticker, df)
            results[ticker] = df
        except Exception as e:
            print(f"\n  ✗ ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(ticker)

    if failed:
        print(f"\n  ⚠ Failed tickers: {failed}")
        print(f"    These will be skipped on Day 2.")

    print_final_summary(results)
