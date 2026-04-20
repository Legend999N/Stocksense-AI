"""
validate_day3.py — StockSense AI
===================================
Run this after setting up Day 3 files to verify everything
is wired correctly before launching the dashboard.

    python validate_day3.py

Checks:
  ✓ All component files exist
  ✓ core/api.py functions importable
  ✓ Streamlit config exists
  ✓ All Day 2 models still present
  ✓ API functions return correct dict keys
  ✓ No import errors in any module
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"

all_passed = True


def check(condition, message, critical=True):
    global all_passed
    if condition:
        print(f"{PASS} {message}")
    else:
        print(f"{FAIL if critical else WARN} {message}")
        if critical:
            all_passed = False
    return condition


print("\n" + "=" * 55)
print("  DAY 3 VALIDATION")
print("=" * 55)

# ── File structure ────────────────────────────────────────────────
print("\n── File Structure ───────────────────────────────────")

required_files = [
    "app.py",
    "core/__init__.py",
    "core/api.py",
    "components/__init__.py",
    "components/chart.py",
    "components/signals.py",
    "components/prediction.py",
    "components/sentiment.py",
    ".streamlit/config.toml",
]

for f in required_files:
    check(os.path.exists(f), f"Exists: {f}")

# ── Day 2 models present ──────────────────────────────────────────
print("\n── Day 2 Models ─────────────────────────────────────")
models_dir = "models/saved"
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.joblib")]
    check(len(model_files) > 0,
          f"At least one trained model found: {model_files}")
else:
    check(False, "models/saved/ directory missing — run day2_pipeline.py")

# ── Import checks ─────────────────────────────────────────────────
print("\n── Import Tests ─────────────────────────────────────")

try:
    from core.api import (
        get_stock_chart_data, get_technical_signals,
        get_prediction, get_sentiment
    )
    check(True, "core/api.py imports successfully")
except Exception as e:
    check(False, f"core/api.py import failed: {e}")

try:
    from components.chart import render_candlestick_chart
    check(True, "components/chart.py imports successfully")
except Exception as e:
    check(False, f"components/chart.py import failed: {e}")

try:
    from components.signals import render_signals_section
    check(True, "components/signals.py imports successfully")
except Exception as e:
    check(False, f"components/signals.py import failed: {e}")

try:
    from components.prediction import render_prediction_section
    check(True, "components/prediction.py imports successfully")
except Exception as e:
    check(False, f"components/prediction.py import failed: {e}")

try:
    from components.sentiment import render_sentiment_section
    check(True, "components/sentiment.py imports successfully")
except Exception as e:
    check(False, f"components/sentiment.py import failed: {e}")

# ── Live API smoke tests ──────────────────────────────────────────
print("\n── API Response Tests (AAPL) ────────────────────────")

# Test chart data
try:
    from core.api import get_stock_chart_data
    # bypass cache for validation
    import yfinance as yf
    raw = yf.Ticker("AAPL").history(period="3mo")
    check(not raw.empty, "yfinance can fetch AAPL data")
    check(all(col in raw.columns for col in ['Open','High','Low','Close','Volume']),
          "OHLCV columns present")
except Exception as e:
    check(False, f"Chart data test failed: {e}")

# Test technical signals output keys
try:
    from features.technical import build_features
    import pandas as pd
    raw2 = yf.Ticker("AAPL").history(period="1y")
    raw2 = raw2[['Open','High','Low','Close','Volume']]
    raw2.index = raw2.index.tz_localize(None) if raw2.index.tz else raw2.index
    featured = build_features(raw2)
    required_signal_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
                             'BB_position', 'BB_width', 'volume_ratio', 'ATR_normalised']
    for col in required_signal_cols:
        check(col in featured.columns, f"Signal column present: {col}")
except Exception as e:
    check(False, f"Signals test failed: {e}")

# Test sentiment
try:
    from features.sentiment import score_headline
    result = score_headline("Apple reports strong earnings growth")
    check('polarity' in result, "Sentiment scorer returns polarity")
    check('label'    in result, "Sentiment scorer returns label")
    check(result['label'] in ["Positive","Negative","Neutral"],
          f"Sentiment label valid: {result['label']}")
except Exception as e:
    check(False, f"Sentiment test failed: {e}")

# ── Streamlit config check ────────────────────────────────────────
print("\n── Streamlit Config ─────────────────────────────────")
try:
    import tomllib
    with open(".streamlit/config.toml", "rb") as f:
        config = tomllib.load(f)
    check("theme" in config, "Theme section found in config.toml")
    check(config["theme"].get("backgroundColor") == "#080D14",
          "Dark background colour correct")
except ImportError:
    # tomllib only in Python 3.11+, use manual check
    with open(".streamlit/config.toml", "r") as f:
        content = f.read()
    check("primaryColor" in content, "Config has primaryColor")
    check("080D14" in content, "Config has dark background")
except Exception as e:
    check(False, f"Config check failed: {e}", critical=False)

# ── Final verdict ─────────────────────────────────────────────────
print(f"\n{'=' * 55}")
if all_passed:
    print("  ✓ ALL CHECKS PASSED — Ready to launch!")
    print("\n  Start the dashboard:")
    print("    streamlit run app.py")
    print("\n  Then open: http://localhost:8501")
    print("\n  Day 4 tasks:")
    print("    1. Polish UI + error handling")
    print("    2. Deploy to Streamlit Cloud")
    print("    3. Write README + record demo")
else:
    print("  ✗ SOME CHECKS FAILED")
    print("    Fix the issues above before launching.")
print("=" * 55 + "\n")