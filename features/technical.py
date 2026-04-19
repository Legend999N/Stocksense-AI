"""
technical.py — StockSense AI
==============================
All feature engineering lives here: technical indicators, lag features,
rolling statistics, and the target variable.

Why each indicator?
  • Moving Averages (SMA/EMA) → Trend direction and momentum
  • RSI                        → Overbought / oversold conditions
  • MACD                       → Momentum changes, crossover signals
  • Bollinger Bands            → Volatility and price extremes
  • Volume indicators          → Confirms price moves (high volume = conviction)
  • Lag features               → Gives the model "memory" of recent behaviour
  • Target variable            → What we're trying to predict: UP or DOWN tomorrow

We use the `ta` library (Technical Analysis library for Python) which wraps
all these calculations cleanly. Under the hood it uses pandas rolling windows —
exactly what you'd build manually in a data science course.

Install: pip install ta
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


# ─────────────────────────────────────────────
# MOVING AVERAGES
# ─────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).

    SMA: Equal weight to all days in the window.
         SMA_20 = average of last 20 closing prices.

    EMA: More weight to recent days. Reacts faster to price changes.
         Used as the basis for MACD.

    Signals generated:
      • price_above_SMA20   → 1 if price is above the 20-day SMA (bullish short-term)
      • price_above_SMA50   → 1 if price is above the 50-day SMA (bullish long-term)
      • SMA_crossover       → "Golden Cross" when SMA20 > SMA50 (strong bull signal)
                              "Death Cross"  when SMA20 < SMA50 (strong bear signal)
    """
    df = df.copy()

    # Simple Moving Averages
    for window in [10, 20, 50]:
        indicator = SMAIndicator(close=df['Close'], window=window)
        df[f'SMA_{window}'] = indicator.sma_indicator()

    # Exponential Moving Averages (12 and 26 days are MACD standard)
    for window in [12, 26]:
        indicator = EMAIndicator(close=df['Close'], window=window)
        df[f'EMA_{window}'] = indicator.ema_indicator()

    # Derived signal features
    df['price_above_SMA20'] = (df['Close'] > df['SMA_20']).astype(int)
    df['price_above_SMA50'] = (df['Close'] > df['SMA_50']).astype(int)
    df['SMA_crossover']     = (df['SMA_20'] > df['SMA_50']).astype(int)

    # Distance from SMA (normalised) — useful continuous feature
    df['dist_from_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['dist_from_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    return df


# ─────────────────────────────────────────────
# RSI — Relative Strength Index
# ─────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    RSI measures the speed and magnitude of price changes.
    Range: 0–100.

    Key thresholds:
      • RSI > 70 → Overbought  (potential reversal DOWN)
      • RSI < 30 → Oversold    (potential reversal UP)
      • RSI ~50  → Neutral trend

    Formula: RSI = 100 - [100 / (1 + RS)]
             RS  = Average Gain over window / Average Loss over window

    14-day is the standard; Wilder's smoothing is used internally by `ta`.
    """
    df = df.copy()

    rsi_indicator = RSIIndicator(close=df['Close'], window=window)
    df['RSI'] = rsi_indicator.rsi()

    # Binary overbought/oversold flags
    df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_oversold']   = (df['RSI'] < 30).astype(int)

    # Normalised RSI (0 to 1) — sometimes easier for models than raw 0–100
    df['RSI_normalised'] = df['RSI'] / 100.0

    return df


# ─────────────────────────────────────────────
# MACD — Moving Average Convergence Divergence
# ─────────────────────────────────────────────

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD is a trend-following momentum indicator.

    Components:
      • MACD line     = EMA(12) - EMA(26)    — the fast line
      • Signal line   = EMA(9) of MACD       — the slow line
      • Histogram     = MACD - Signal        — the crossover visualisation

    Signals:
      • MACD crosses above Signal → Bullish (buy signal)
      • MACD crosses below Signal → Bearish (sell signal)
      • Histogram increasing      → Momentum strengthening
      • Histogram decreasing      → Momentum weakening

    Standard parameters: fast=12, slow=26, signal=9
    """
    df = df.copy()

    macd_indicator = MACD(
        close=df['Close'],
        window_fast=12,
        window_slow=26,
        window_sign=9
    )

    df['MACD']           = macd_indicator.macd()
    df['MACD_signal']    = macd_indicator.macd_signal()
    df['MACD_histogram'] = macd_indicator.macd_diff()  # histogram = MACD - Signal

    # Binary: is MACD currently bullish (above signal)?
    df['MACD_bullish'] = (df['MACD'] > df['MACD_signal']).astype(int)

    # Histogram direction (increasing = momentum building)
    df['MACD_hist_increasing'] = (
        df['MACD_histogram'] > df['MACD_histogram'].shift(1)
    ).astype(int)

    return df


# ─────────────────────────────────────────────
# BOLLINGER BANDS
# ─────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Bollinger Bands consist of:
      • Middle Band = 20-day SMA
      • Upper Band  = SMA + 2 × standard deviation
      • Lower Band  = SMA - 2 × standard deviation

    When volatility is high, the bands widen.
    When volatility is low,  the bands narrow (squeeze = breakout coming).

    Key features:
      • BB_width    → How wide the bands are (normalised by middle band)
                      High width = high volatility
      • BB_position → Where is price within the bands? (0 = at lower, 1 = at upper)
                      >1.0 means price broke above the upper band (very overbought)
                      <0.0 means price broke below the lower band (very oversold)
    """
    df = df.copy()

    bb = BollingerBands(close=df['Close'], window=window, window_dev=2)

    df['BB_upper']    = bb.bollinger_hband()
    df['BB_lower']    = bb.bollinger_lband()
    df['BB_middle']   = bb.bollinger_mavg()
    df['BB_width']    = bb.bollinger_wband()   # (upper - lower) / middle
    df['BB_position'] = bb.bollinger_pband()   # (close - lower) / (upper - lower)

    # Binary signals
    df['BB_above_upper'] = (df['Close'] > df['BB_upper']).astype(int)  # Overbought
    df['BB_below_lower'] = (df['Close'] < df['BB_lower']).astype(int)  # Oversold

    return df


# ─────────────────────────────────────────────
# VOLUME INDICATORS
# ─────────────────────────────────────────────

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume confirms price moves. A breakout on HIGH volume is more reliable
    than one on low volume.

    Features:
      • OBV (On-Balance Volume)
            Cumulative indicator. Adds volume on up-days, subtracts on down-days.
            Rising OBV with rising price = confirmed uptrend.
            Rising OBV with falling price = potential reversal up (accumulation).

      • volume_ratio
            Today's volume / 20-day average volume.
            Ratio > 1.5 means unusually high volume (significant event).
            Ratio < 0.5 means unusually low volume (low conviction).

      • ATR (Average True Range)
            Measures daily price volatility independent of direction.
            High ATR = high risk / high opportunity.
    """
    df = df.copy()

    # On-Balance Volume
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()

    # Volume moving average and ratio
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['Volume_SMA20']

    # High volume flag
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)

    # ATR — normalised by close price
    atr = AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=14
    )
    df['ATR']            = atr.average_true_range()
    df['ATR_normalised'] = df['ATR'] / df['Close']  # % of price, comparable across stocks

    return df


# ─────────────────────────────────────────────
# LAG FEATURES + ROLLING STATS
# ─────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag features give the model "memory". Without them, the model only sees
    today's indicators, not how they've been trending.

    Lag 1 = yesterday's value
    Lag 2 = two days ago
    etc.

    We also add:
      • Daily return        = % change in closing price
      • Rolling std         = recent price volatility (risk proxy)
      • HL_spread           = (High - Low) / Close = intraday volatility
      • OC_spread           = (Close - Open) / Open = day's direction + magnitude
    """
    df = df.copy()

    # Lagged closing prices
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)

    # Daily percentage returns
    df['daily_return'] = df['Close'].pct_change()

    # Lagged returns (past performance)
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

    # Rolling statistics
    df['rolling_std_5']  = df['Close'].rolling(window=5).std()   # 1-week volatility
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()  # 1-month volatility
    df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()  # 1-week average price

    # Intraday ranges (normalised)
    df['HL_spread'] = (df['High'] - df['Low']) / df['Close']      # Intraday volatility
    df['OC_spread'] = (df['Close'] - df['Open']) / df['Open']     # Day direction

    # Lagged HL spread (was yesterday also volatile?)
    df['HL_spread_lag1'] = df['HL_spread'].shift(1)

    return df


# ─────────────────────────────────────────────
# TARGET VARIABLE
# ─────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary classification target variable.

    Target = 1 if tomorrow's Close > today's Close  (price goes UP)
    Target = 0 if tomorrow's Close ≤ today's Close  (price goes DOWN or flat)

    IMPORTANT: We also store next_close and next_day_return for reference,
    but these must NEVER be used as input features — only as targets.
    They represent future information the model wouldn't have at prediction time.

    The last row will have NaN target (we don't know tomorrow's price yet).
    It will be dropped in the pipeline below, which is correct.
    """
    df = df.copy()

    # Tomorrow's closing price (shift back by 1 — looking into the future)
    df['next_close']      = df['Close'].shift(-1)

    # Percentage return tomorrow
    df['next_day_return'] = (df['next_close'] - df['Close']) / df['Close']

    # Binary target: 1 = UP, 0 = DOWN/flat
    df['target'] = (df['next_close'] > df['Close']).astype(int)

    return df


# ─────────────────────────────────────────────
# MASTER PIPELINE FUNCTION
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    Args:
        df : Raw OHLCV DataFrame (from fetch_data.py)

    Returns:
        Feature-rich DataFrame with all indicators, lag features,
        and target variable. NaN rows are dropped.

    Column count after: ~40+ features + target
    Row count: original rows minus ~50 (due to 50-day SMA window being the
               longest rolling calculation we use)
    """
    print("  [1/7] Adding moving averages...")
    df = add_moving_averages(df)

    print("  [2/7] Adding RSI...")
    df = add_rsi(df)

    print("  [3/7] Adding MACD...")
    df = add_macd(df)

    print("  [4/7] Adding Bollinger Bands...")
    df = add_bollinger_bands(df)

    print("  [5/7] Adding volume indicators...")
    df = add_volume_indicators(df)

    print("  [6/7] Adding lag features...")
    df = add_lag_features(df)

    print("  [7/7] Adding target variable...")
    df = add_target(df)

    # ── Drop rows with NaN ───────────────────────────────────────────────────
    # NaNs appear because:
    #   • SMA_50 needs 50 rows before it can compute → first 50 rows are NaN
    #   • Lag features shift data by up to 5 days → first 5 rows are NaN
    #   • Target (next_close) shifts -1 → last row is NaN
    # Dropping these is correct. Do NOT fill them — fabricated values here
    # would introduce data leakage and bias the model.
    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)

    print(f"\n  Dropped {rows_dropped} rows with NaN (expected — due to rolling windows)")
    print(f"  Final dataset: {len(df)} rows × {len(df.columns)} columns")

    return df


# ─────────────────────────────────────────────
# FEATURE LIST HELPER
# ─────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return only the input feature columns (exclude OHLCV base + target columns).
    These are the columns that will be fed into the ML model on Day 2.
    """
    # Columns that are NOT model features
    exclude = {
        'Open', 'High', 'Low', 'Close', 'Volume',  # raw OHLCV
        'next_close', 'next_day_return', 'target'   # target/leakage columns
    }
    return [col for col in df.columns if col not in exclude]


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf

    print("=" * 50)
    print("TESTING technical.py")
    print("=" * 50)

    # Fetch raw data directly
    print("\nFetching raw AAPL data...")
    raw = yf.Ticker("AAPL").history(period="2y")
    raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
    raw.index = raw.index.tz_localize(None)

    # Run full pipeline
    print("\nRunning feature engineering pipeline...")
    featured = build_features(raw)

    # Summary
    feature_cols = get_feature_columns(featured)
    print(f"\n{'─'*50}")
    print(f"Feature DataFrame shape : {featured.shape}")
    print(f"Number of features      : {len(feature_cols)}")
    print(f"\nAll feature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:02d}. {col}")

    print(f"\nTarget distribution:")
    vc = featured['target'].value_counts()
    print(f"  UP   (1): {vc.get(1, 0)} ({vc.get(1, 0)/len(featured)*100:.1f}%)")
    print(f"  DOWN (0): {vc.get(0, 0)} ({vc.get(0, 0)/len(featured)*100:.1f}%)")

    print(f"\nSample row (last row):")
    print(featured.iloc[-1][feature_cols[:10]])

    print("\n✓ technical.py is working correctly!")
