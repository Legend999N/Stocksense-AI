"""
validate_day2.py — StockSense AI
===================================
Run this after day2_pipeline.py to verify everything is
in order before starting Day 3 (dashboard).

    python validate_day2.py

Checks performed:
  ✓ Trained model files exist for each ticker
  ✓ Feature JSON files match model
  ✓ Metrics JSON is valid and shows >50% accuracy
  ✓ ROC-AUC is above random (>0.50)
  ✓ SHAP importance file exists
  ✓ Model can load and make a test prediction
  ✓ Sentiment pipeline returns valid score
"""

import os
import sys
import json
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        symbol = FAIL if critical else WARN
        print(f"{symbol} {message}")
        if critical:
            all_passed = False
        return False


def get_available_tickers() -> list:
    features_dir = "data/features"
    if not os.path.exists(features_dir):
        return []
    return [f.replace("_features.csv", "")
            for f in os.listdir(features_dir)
            if f.endswith("_features.csv")]


print("\n" + "=" * 55)
print("  DAY 2 VALIDATION")
print("=" * 55)

tickers = get_available_tickers()

if not tickers:
    print(f"{FAIL} No feature CSVs found — run day1_pipeline.py first")
    sys.exit(1)

print(f"\n  Checking {len(tickers)} ticker(s): {tickers}\n")

for ticker in tickers:
    print(f"── {ticker} ──────────────────────────────────────")

    # ── Find which model version exists ──────────────────────────
    model_found    = False
    model_suffix   = None
    model_path     = None
    features_path  = None
    metrics_path   = None

    for suffix in ["_sentiment", ""]:
        mp = f"models/saved/{ticker}{suffix}_model.joblib"
        fp = f"models/saved/{ticker}{suffix}_features.json"
        if os.path.exists(mp):
            model_suffix  = suffix
            model_path    = mp
            features_path = fp
            metrics_path  = f"models/saved/{ticker}{suffix}_metrics.json"
            model_found   = True
            break

    # Model file exists
    check(model_found, f"Model file exists ({model_path})")
    if not model_found:
        print(f"  → Skipping {ticker} — run day2_pipeline.py first\n")
        continue

    # Features JSON exists
    check(os.path.exists(features_path),
          f"Features JSON exists ({features_path})")

    # Metrics JSON exists and is valid
    metrics_ok = False
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            metrics_ok = True
            check(True, "Metrics JSON is valid")

            # Accuracy above 50%
            acc = metrics.get('accuracy', 0)
            check(acc > 0.50, f"Accuracy above 50%: {acc*100:.1f}%")

            # ROC-AUC above 0.50 (better than random)
            auc = metrics.get('roc_auc', 0)
            check(auc > 0.50, f"ROC-AUC above random: {auc:.4f}")

            # Warning if accuracy seems too high (possible leakage)
            if acc > 0.75:
                print(f"{WARN} Accuracy {acc*100:.1f}% is unusually high — "
                      f"double-check for data leakage")

        except Exception as e:
            check(False, f"Metrics JSON unreadable: {e}")
    else:
        check(False, f"Metrics JSON missing: {metrics_path}")

    # Feature count is reasonable
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            feature_cols = json.load(f)
        check(len(feature_cols) >= 30,
              f"Feature count: {len(feature_cols)} (minimum 30)")
        has_sentiment = 'sentiment_score' in feature_cols
        print(f"{PASS} Sentiment feature included: {has_sentiment}")

    # SHAP importance file exists (non-critical)
    shap_path = f"models/saved/{ticker}_shap_importance.json"
    shap_exists = os.path.exists(shap_path)
    check(shap_exists, f"SHAP importance file exists", critical=False)
    if not shap_exists:
        print(f"    → SHAP is optional; dashboard will skip the importance chart")

    # Model can actually load and predict (critical smoke test)
    try:
        model = joblib.load(model_path)
        check(True, "Model loads without error")

        # Try a dummy prediction with zeros (just tests model integrity)
        import numpy as np
        dummy_X = pd.DataFrame(
            np.zeros((1, len(feature_cols))),
            columns=feature_cols
        )
        pred  = model.predict(dummy_X)
        proba = model.predict_proba(dummy_X)
        check(pred[0] in [0, 1],     "Model predicts valid class (0 or 1)")
        check(0 <= proba[0][1] <= 1, "Probability output is valid (0–1)")

    except Exception as e:
        check(False, f"Model load/predict failed: {e}")

    print()

# ── Sentiment pipeline check ─────────────────────────────────────
print("── Sentiment Pipeline ──────────────────────────────")
try:
    from features.sentiment import score_headline, aggregate_sentiment

    # Test scoring a known positive headline
    result = score_headline("Apple reports record-breaking profits and strong growth")
    check(result['polarity'] > 0,
          f"Sentiment scorer works: '{result['label']}' ({result['polarity']:+.3f})")

    # Test aggregation
    scored = [
        {'polarity': 0.3,  'label': 'Positive', 'subjectivity': 0.5},
        {'polarity': -0.1, 'label': 'Negative', 'subjectivity': 0.3},
        {'polarity': 0.0,  'label': 'Neutral',  'subjectivity': 0.1},
    ]
    agg = aggregate_sentiment(scored)
    check(agg['average_score'] == round((0.3 - 0.1 + 0.0) / 3, 4),
          f"Sentiment aggregation correct: {agg['average_score']:+.4f}")
    check(agg['overall_label'] in ['Positive', 'Negative', 'Neutral'],
          f"Overall label is valid: {agg['overall_label']}")

except Exception as e:
    check(False, f"Sentiment pipeline error: {e}")

# ── Final verdict ─────────────────────────────────────────────────
print(f"\n{'=' * 55}")
if all_passed:
    print("  ✓ ALL CHECKS PASSED — Ready for Day 3!")
    print("\n  Day 3 tasks:")
    print("    1. Build Streamlit dashboard (app.py)")
    print("    2. Candlestick chart component")
    print("    3. RSI / MACD / BB signal cards")
    print("    4. Prediction panel with SHAP waterfall")
    print("    5. Sentiment news panel")
else:
    print("  ✗ SOME CHECKS FAILED — Fix issues before Day 3")
    print("    Re-run day2_pipeline.py after fixing.")
print("=" * 55 + "\n")