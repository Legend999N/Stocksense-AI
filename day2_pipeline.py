"""
day2_pipeline.py — StockSense AI
===================================
Master runner for Day 2. Ties together:
  - features/sentiment.py  (TextBlob scoring)
  - models/train.py        (XGBoost training)
  - models/explain.py      (SHAP explainability)

Run this file to complete all of Day 2:
    python day2_pipeline.py

What it does:
  For each ticker that has a Day 1 feature CSV:
    1. Trains Model A (technical features only)
    2. Fetches sentiment score via TextBlob
    3. Trains Model B (technical + sentiment)
    4. Compares A vs B — saves the better one
    5. Computes SHAP global feature importance
    6. Computes SHAP local explanation for latest prediction
    7. Saves everything to models/saved/

Expected output after running:
  models/saved/
    AAPL_sentiment_model.joblib
    AAPL_sentiment_features.json
    AAPL_sentiment_metrics.json
    AAPL_shap_importance.json
    TSLA_sentiment_model.joblib
    ... etc

Time to run: ~2–4 minutes for 5 tickers
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.train   import train_ticker
from models.explain import run_explain_pipeline


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Only process tickers that successfully completed Day 1
# This auto-detects from data/features/
def get_available_tickers() -> list:
    """
    Find tickers that have a feature CSV from Day 1.
    Avoids manual config — just checks what's on disk.
    """
    features_dir = "data/features"
    if not os.path.exists(features_dir):
        return []

    tickers = []
    for filename in os.listdir(features_dir):
        if filename.endswith("_features.csv"):
            ticker = filename.replace("_features.csv", "")
            tickers.append(ticker)

    return sorted(tickers)


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def print_training_summary(all_results: dict) -> None:
    """
    Print a comparison table of all trained models.
    Useful for quickly identifying which stocks have the strongest signal.
    """
    print(f"\n\n{'═'*65}")
    print(f"  DAY 2 TRAINING SUMMARY")
    print(f"{'═'*65}")
    print(f"  {'Ticker':<8} {'Model A Acc':>12} {'Model B Acc':>12} {'ROC-AUC':>9} {'Sentiment':>12} {'Best':>6}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*9} {'─'*12} {'─'*6}")

    for ticker, result in all_results.items():
        if 'error' in result:
            print(f"  {ticker:<8} {'ERROR':>12} {'─':>12} {'─':>9} {'─':>12} {'─':>6}")
            continue

        acc_a     = result['model_a_metrics']['accuracy'] * 100
        acc_b     = result['model_b_metrics']['accuracy'] * 100
        auc       = result['model_b_metrics']['roc_auc']
        sentiment = result['sentiment_score']
        best      = "B" if result['acc_delta'] >= 0 else "A"

        print(f"  {ticker:<8} {acc_a:>11.1f}% {acc_b:>11.1f}% {auc:>9.4f} "
              f"{sentiment:>+11.3f}  {best:>6}")

    print(f"{'─'*65}")
    print(f"  Model A: technical features only")
    print(f"  Model B: technical + TextBlob sentiment score")
    print(f"\n  Models saved to: models/saved/")
    print(f"\n  Day 2 complete! Files ready for Day 3 dashboard:")
    print(f"    • Trained models        → models/saved/<TICKER>_model.joblib")
    print(f"    • Feature lists         → models/saved/<TICKER>_features.json")
    print(f"    • SHAP importance       → models/saved/<TICKER>_shap_importance.json")
    print(f"    • Evaluation metrics    → models/saved/<TICKER>_metrics.json")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*65}")
    print(f"  STOCKSENSE AI — DAY 2 PIPELINE")
    print(f"  Model Training + Sentiment + SHAP Explainability")
    print(f"{'═'*65}")

    # Auto-detect available tickers from Day 1 output
    tickers = get_available_tickers()

    if not tickers:
        print("\n  ✗ No feature CSVs found in data/features/")
        print("    Run day1_pipeline.py first!\n")
        sys.exit(1)

    print(f"\n  Found {len(tickers)} ticker(s) from Day 1: {tickers}")
    print(f"  Running full Day 2 pipeline for each...\n")

    all_results = {}

    for ticker in tickers:
        try:
            # ── Step 1: Train both models (A and B) ──────────────
            train_result = train_ticker(ticker)
            all_results[ticker] = train_result

            # ── Step 2: SHAP explainability ───────────────────────
            print(f"\n  Running SHAP for {ticker}...")
            try:
                run_explain_pipeline(ticker)
            except Exception as e:
                # SHAP failing doesn't break the project — predictions still work
                print(f"  ⚠ SHAP failed for {ticker}: {e}")
                print(f"    This is non-critical — dashboard will still work.")

        except Exception as e:
            print(f"\n  ✗ ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ticker] = {'error': str(e)}

    print_training_summary(all_results)