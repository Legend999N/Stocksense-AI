"""
fetch_data.py — StockSense AI
==============================
Handles all data collection from Yahoo Finance via yfinance.

What this module does:
  1. Downloads OHLCV (Open, High, Low, Close, Volume) data for any stock ticker
  2. Cleans and validates the data (missing values, timezone stripping, etc.)
  3. Saves raw data to data/raw/<TICKER>_raw.csv for reproducibility
  4. Supports bulk fetching of multiple tickers at once

Why yfinance?
  - Free, no API key required
  - 2+ years of daily historical data available
  - Reliable for prototyping; can swap for Alpha Vantage/Polygon later
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime


# ─────────────────────────────────────────────
# CORE FETCH FUNCTION
# ─────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "3y", save: bool = True) -> pd.DataFrame:
    """
    Download OHLCV data for a single stock ticker.

    Args:
        ticker : Stock symbol, e.g. 'AAPL', 'TSLA'
        period : How far back to go. Options: '1y', '2y', '3y', '5y', 'max'
                 3y gives ~750 trading days — enough for solid model training.
        save   : If True, saves raw CSV to data/raw/<ticker>_raw.csv

    Returns:
        pd.DataFrame with columns [Open, High, Low, Close, Volume]
        indexed by Date (DatetimeIndex, timezone-naive)

    Raises:
        ValueError if the ticker is invalid or yfinance returns no data
    """
    print(f"  Fetching {ticker} ({period} of data)...")

    # Download from Yahoo Finance
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    # Validate we actually got data
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            f"Check the symbol is correct (e.g. 'AAPL', not 'Apple')."
        )

    # ── Keep only OHLCV columns ──────────────────────────────────────────────
    # yfinance also returns 'Dividends' and 'Stock Splits' — we don't need those
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # ── Fix the DatetimeIndex ────────────────────────────────────────────────
    # yfinance returns timezone-aware index (America/New_York).
    # We strip timezone info to keep things simple — all our analysis
    # is date-based, not time-based.
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = 'Date'

    # ── Handle missing values ────────────────────────────────────────────────
    # Stock markets have occasional data gaps (holidays, trading halts).
    # Strategy:
    #   1. Forward fill  → carry the last known price forward (most realistic)
    #   2. Backward fill → only for any NaNs at the very start of the series
    missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    missing_after = df.isnull().sum().sum()

    if missing_before > 0:
        print(f"    Filled {missing_before} missing values (ffill/bfill)")

    # ── Data quality checks ──────────────────────────────────────────────────
    # Make sure Close prices are positive (sanity check)
    if (df['Close'] <= 0).any():
        raise ValueError(f"Negative/zero close prices found for {ticker}. Data may be corrupt.")

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"    ✓ {len(df)} trading days | "
          f"{df.index[0].date()} → {df.index[-1].date()} | "
          f"Missing values remaining: {missing_after}")

    # ── Optionally save to disk ──────────────────────────────────────────────
    if save:
        os.makedirs("data/raw", exist_ok=True)
        path = f"data/raw/{ticker}_raw.csv"
        df.to_csv(path)
        print(f"    ✓ Saved to {path}")

    return df


# ─────────────────────────────────────────────
# BULK FETCH FUNCTION
# ─────────────────────────────────────────────

def fetch_multiple_stocks(tickers: list, period: str = "3y") -> dict:
    """
    Fetch OHLCV data for a list of tickers.

    Args:
        tickers : List of stock symbols, e.g. ['AAPL', 'TSLA', 'GOOGL']
        period  : Same as fetch_stock_data — applies to all tickers

    Returns:
        dict mapping ticker → DataFrame
        Tickers that fail (bad symbol, no data) are skipped with a warning.

    Example:
        stocks = fetch_multiple_stocks(['AAPL', 'TSLA'])
        aapl_df = stocks['AAPL']
    """
    stocks = {}
    failed = []

    print(f"\nFetching {len(tickers)} stocks...\n")

    for ticker in tickers:
        try:
            stocks[ticker] = fetch_stock_data(ticker, period=period, save=True)
        except Exception as e:
            print(f"  ✗ Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    print(f"\n{'─'*40}")
    print(f"Fetched successfully : {list(stocks.keys())}")
    if failed:
        print(f"Failed              : {failed}")

    return stocks


# ─────────────────────────────────────────────
# LOAD FROM DISK (use after first fetch)
# ─────────────────────────────────────────────

def load_raw_data(ticker: str) -> pd.DataFrame:
    """
    Load previously saved raw CSV from data/raw/<ticker>_raw.csv.
    Use this on Day 2+ to avoid re-downloading every run.

    Args:
        ticker : Stock symbol

    Returns:
        pd.DataFrame with DatetimeIndex
    """
    path = f"data/raw/{ticker}_raw.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached data found at {path}. "
            f"Run fetch_stock_data('{ticker}') first."
        )
    df = pd.read_csv(path, index_col='Date', parse_dates=True)
    return df


# ─────────────────────────────────────────────
# QUICK TEST (run this file directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING fetch_data.py")
    print("=" * 50)

    # Test single fetch
    print("\n[Test 1] Single stock fetch")
    df = fetch_stock_data("AAPL", period="1y", save=True)
    print(f"\nSample data (last 3 rows):\n{df.tail(3)}")
    print(f"\nData types:\n{df.dtypes}")

    # Test bulk fetch
    print("\n[Test 2] Bulk fetch")
    stocks = fetch_multiple_stocks(["MSFT", "GOOGL"], period="1y")
    for t, d in stocks.items():
        print(f"  {t}: {d.shape}")

    # Test load from disk
    print("\n[Test 3] Load from disk")
    df_loaded = load_raw_data("AAPL")
    print(f"  Loaded AAPL from disk: {df_loaded.shape}")

    print("\n✓ fetch_data.py is working correctly!")
