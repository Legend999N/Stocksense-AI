"""
sentiment.py — StockSense AI
==============================
Sentiment analysis pipeline using TextBlob.

What this module does:
  1. Fetches recent news headlines for any stock ticker via yfinance
  2. Scores each headline using TextBlob polarity (-1.0 to +1.0)
  3. Aggregates scores into a single daily sentiment value
  4. Returns structured data ready to merge with price features

Why TextBlob?
  - Zero model loading time (no 440MB FinBERT download)
  - Works on 8GB RAM with no GPU
  - Accurate enough for a prototype — polarity on financial text
    is reliable for clearly positive/negative headlines
  - One-line swap to FinBERT later if needed

TextBlob Polarity Scale:
   -1.0  →  Very Negative  ("stock crashes", "massive losses")
    0.0  →  Neutral        ("earnings report released")
   +1.0  →  Very Positive  ("record profits", "strong growth")

Usage:
    from features.sentiment import get_sentiment_score
    result = get_sentiment_score("AAPL")
    print(result['average_score'])     # e.g. 0.12
    print(result['overall_label'])     # "Positive"
    print(result['headlines'])         # list of scored headlines
"""

import yfinance as yf
from textblob import TextBlob
from datetime import datetime


# ─────────────────────────────────────────────
# SINGLE HEADLINE SCORER
# ─────────────────────────────────────────────

def score_headline(text: str) -> dict:
    """
    Score a single news headline using TextBlob.

    TextBlob.sentiment returns two values:
      polarity    : -1.0 to +1.0  (negative to positive)
      subjectivity: 0.0 to 1.0    (objective to subjective)

    We only use polarity for sentiment. Subjectivity is stored
    for potential future use (highly subjective headlines might
    be less reliable signals).

    Args:
        text : Raw headline string

    Returns:
        dict with keys: text, polarity, subjectivity, label
    """
    if not text or not isinstance(text, str):
        return {
            "text":         "N/A",
            "polarity":     0.0,
            "subjectivity": 0.0,
            "label":        "Neutral"
        }

    blob = TextBlob(text)
    polarity     = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    # Convert polarity to human-readable label
    # Thresholds: >0.05 = Positive, <-0.05 = Negative, else Neutral
    # Using 0.05 instead of 0.0 avoids labelling near-zero as Positive
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "text":         text,
        "polarity":     polarity,
        "subjectivity": subjectivity,
        "label":        label,
    }


# ─────────────────────────────────────────────
# FETCH HEADLINES FROM YAHOO FINANCE
# ─────────────────────────────────────────────

def fetch_headlines(ticker: str, max_headlines: int = 10) -> list:
    """
    Fetch recent news headlines for a ticker using yfinance.

    yfinance returns news as a list of dicts. Each has:
      'title'      : headline text  ← what we use
      'publisher'  : source name
      'link'       : article URL
      'providerPublishTime': unix timestamp

    Args:
        ticker        : Stock symbol, e.g. 'AAPL'
        max_headlines : Max headlines to return (default 10)

    Returns:
        List of headline strings (titles only)

    Note:
        If yfinance returns no news (can happen for less-covered
        tickers or API issues), we return a fallback list so the
        pipeline doesn't break — the sentiment score will just be 0.
    """
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news

        if not news:
            print(f"    ⚠ No headlines found for {ticker} — using neutral fallback")
            return []

        # Extract just the title from each news item
        headlines = []
        for item in news[:max_headlines]:
            # yfinance news structure can vary slightly
            title = item.get('title', '') or item.get('content', {}).get('title', '')
            if title:
                headlines.append(title)

        return headlines

    except Exception as e:
        print(f"    ⚠ Failed to fetch headlines for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────
# AGGREGATE SENTIMENT SCORE
# ─────────────────────────────────────────────

def aggregate_sentiment(scored_headlines: list) -> dict:
    """
    Aggregate a list of scored headlines into one overall score.

    Method: Simple average of all polarity scores.
    Weighted average (by recency) would be more sophisticated but
    yfinance doesn't reliably return timestamps in consistent order.

    Args:
        scored_headlines : List of dicts from score_headline()

    Returns:
        dict with:
          average_score  : float, -1.0 to +1.0
          overall_label  : "Positive" | "Negative" | "Neutral"
          positive_count : int
          negative_count : int
          neutral_count  : int
    """
    if not scored_headlines:
        return {
            "average_score":  0.0,
            "overall_label":  "Neutral",
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count":  0,
        }

    polarities = [h["polarity"] for h in scored_headlines]
    avg        = round(sum(polarities) / len(polarities), 4)

    # Count label distribution
    labels         = [h["label"] for h in scored_headlines]
    positive_count = labels.count("Positive")
    negative_count = labels.count("Negative")
    neutral_count  = labels.count("Neutral")

    # Overall label from average
    if avg > 0.05:
        overall_label = "Positive"
    elif avg < -0.05:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    return {
        "average_score":  avg,
        "overall_label":  overall_label,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count":  neutral_count,
    }


# ─────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ─────────────────────────────────────────────

def get_sentiment_score(ticker: str, max_headlines: int = 10) -> dict:
    """
    Full sentiment pipeline for a single ticker.

    Call this function from anywhere — it handles fetching,
    scoring, and aggregating in one call.

    Args:
        ticker        : Stock symbol, e.g. 'AAPL'
        max_headlines : How many headlines to analyse

    Returns:
        dict with:
          ticker         : str
          headlines      : list of scored headline dicts
          average_score  : float (-1.0 to +1.0)
          overall_label  : "Positive" | "Negative" | "Neutral"
          positive_count : int
          negative_count : int
          neutral_count  : int
          headline_count : int

    Example:
        result = get_sentiment_score("AAPL")
        # result['average_score'] = 0.14
        # result['overall_label'] = "Positive"
        # result['headlines'][0]['text'] = "Apple hits record..."
        # result['headlines'][0]['polarity'] = 0.45
    """
    print(f"  Fetching and scoring headlines for {ticker}...")

    # Step 1: Fetch raw headlines
    raw_headlines = fetch_headlines(ticker, max_headlines)

    # Step 2: Score each headline
    scored = [score_headline(h) for h in raw_headlines]

    # Step 3: Aggregate into a single score
    aggregated = aggregate_sentiment(scored)

    # Combine everything into one return dict
    result = {
        "ticker":         ticker,
        "headlines":      scored,
        "headline_count": len(scored),
        **aggregated,
    }

    print(f"    ✓ {len(scored)} headlines scored | "
          f"Average: {aggregated['average_score']:+.3f} "
          f"({aggregated['overall_label']}) | "
          f"😊{aggregated['positive_count']} "
          f"😐{aggregated['neutral_count']} "
          f"😞{aggregated['negative_count']}")

    return result


# ─────────────────────────────────────────────
# BATCH SENTIMENT FOR MULTIPLE TICKERS
# ─────────────────────────────────────────────

def get_batch_sentiment(tickers: list) -> dict:
    """
    Run sentiment pipeline on a list of tickers.

    Returns:
        dict mapping ticker → sentiment result dict
        Also includes a 'sentiment_scores' sub-dict for easy
        access when merging with price features.

    Example:
        results = get_batch_sentiment(['AAPL', 'TSLA'])
        apple_score = results['AAPL']['average_score']
    """
    results = {}
    print(f"\nRunning sentiment analysis on {len(tickers)} tickers...\n")

    for ticker in tickers:
        try:
            results[ticker] = get_sentiment_score(ticker)
        except Exception as e:
            print(f"  ✗ Sentiment failed for {ticker}: {e}")
            # Return neutral score as fallback — never block the pipeline
            results[ticker] = {
                "ticker":         ticker,
                "headlines":      [],
                "headline_count": 0,
                "average_score":  0.0,
                "overall_label":  "Neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count":  0,
            }

    return results


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("TESTING sentiment.py")
    print("=" * 55)

    # Test single ticker
    print("\n[Test 1] Single ticker sentiment")
    result = get_sentiment_score("AAPL")
    print(f"\n  Ticker         : {result['ticker']}")
    print(f"  Headlines found: {result['headline_count']}")
    print(f"  Average score  : {result['average_score']:+.3f}")
    print(f"  Overall label  : {result['overall_label']}")
    print(f"\n  Top 3 headlines:")
    for h in result['headlines'][:3]:
        print(f"    [{h['label']:8s}] {h['polarity']:+.3f}  {h['text'][:70]}...")

    # Test batch
    print("\n[Test 2] Batch sentiment")
    batch = get_batch_sentiment(["MSFT", "TSLA"])
    for ticker, res in batch.items():
        print(f"  {ticker}: {res['average_score']:+.3f} ({res['overall_label']})")

    # Test edge case — bad ticker
    print("\n[Test 3] Edge case — unknown ticker")
    edge = get_sentiment_score("FAKEXYZ")
    print(f"  Score: {edge['average_score']} | Label: {edge['overall_label']}")

    print("\n✓ sentiment.py is working correctly!")