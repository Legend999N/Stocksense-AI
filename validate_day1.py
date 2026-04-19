"""
validate_day1.py — StockSense AI
==================================
Run this after day1_pipeline.py to verify everything is in order
before moving to Day 2.

    python validate_day1.py

Checks performed:
  ✓ All feature CSVs exist
  ✓ No NaN values in feature datasets
  ✓ Target column exists and is binary (0/1)
  ✓ Feature count is as expected (>30 features)
  ✓ Date index is correct
  ✓ No data leakage (next_close not in feature columns)
  ✓ Train/test split will be valid (enough rows)
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features.technical import get_feature_columns

TICKERS     = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
MIN_ROWS    = 400   # Need at least 400 rows for meaningful training
MIN_FEATURES = 30   # We expect ~40 features

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

all_passed = True


def check(condition: bool, message: str, critical: bool = True) -> bool:
    global all_passed
    if condition:
        print(f"{PASS} {message}")
        return True
    else:
        print(f"{FAIL} {message}")
        if critical:
            all_passed = False
        return False


print("\n" + "=" * 55)
print("  DAY 1 VALIDATION")
print("=" * 55)

for ticker in TICKERS:
    print(f"\n── {ticker} ──────────────────────────────────────")

    # Check file exists
    path = f"data/features/{ticker}_features.csv"
    if not check(os.path.exists(path), f"Feature CSV exists at {path}"):
        print(f"  → Skipping {ticker} — run day1_pipeline.py first")
        continue

    # Load the CSV
    try:
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
    except Exception as e:
        check(False, f"Could not load CSV: {e}")
        continue

    # Row count
    check(len(df) >= MIN_ROWS,
          f"Enough rows: {len(df)} (minimum {MIN_ROWS})")

    # No NaN values
    nan_count = df.isnull().sum().sum()
    check(nan_count == 0,
          f"No NaN values ({nan_count} found)")

    # Target column exists and is binary
    check('target' in df.columns,
          "Target column exists")
    if 'target' in df.columns:
        unique_vals = set(df['target'].unique())
        check(unique_vals.issubset({0, 1}),
              f"Target is binary (values: {unique_vals})")

        # Class balance warning (not a failure, just informational)
        pct_up = df['target'].mean() * 100
        if abs(pct_up - 50) > 15:
            print(f"{WARN} Class imbalance: {pct_up:.1f}% UP days "
                  f"— use class_weight='balanced' in XGBoost")
        else:
            print(f"{PASS} Target balance: {pct_up:.1f}% UP, {100-pct_up:.1f}% DOWN")

    # Feature count
    feature_cols = get_feature_columns(df)
    check(len(feature_cols) >= MIN_FEATURES,
          f"Feature count: {len(feature_cols)} (minimum {MIN_FEATURES})")

    # No data leakage — next_close must NOT be in features
    check('next_close' not in feature_cols,
          "No data leakage (next_close not in features)")

    # Date index is valid
    check(isinstance(df.index, pd.DatetimeIndex),
          "Date index is DatetimeIndex")
    if isinstance(df.index, pd.DatetimeIndex):
        check(df.index.is_monotonic_increasing,
              "Dates are in chronological order")

    # Train/test split viability check (80/20)
    train_size = int(len(df) * 0.8)
    test_size  = len(df) - train_size
    check(train_size >= 300,
          f"Sufficient training rows: {train_size}")
    check(test_size >= 50,
          f"Sufficient test rows    : {test_size}")

    print(f"\n  Summary: {len(df)} rows | "
          f"{len(feature_cols)} features | "
          f"{df.index[0].date()} → {df.index[-1].date()}")


# ── Final verdict ────────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
if all_passed:
    print("  ✓ ALL CHECKS PASSED — Ready for Day 2!")
    print("\n  Day 2 tasks:")
    print("    1. Train XGBoost model (models/train.py)")
    print("    2. Evaluate with time-series split")
    print("    3. Add SHAP explainability")
    print("    4. Build sentiment pipeline (features/sentiment.py)")
else:
    print("  ✗ SOME CHECKS FAILED — Fix issues above before Day 2")
    print("    Re-run day1_pipeline.py after fixing any issues.")
print("=" * 55 + "\n")
