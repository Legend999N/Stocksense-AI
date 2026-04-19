"""
NLP pipeline
Sentiment analysis pipeline for stock-related news and social media
"""

from transformers import pipeline


def analyze_sentiment(texts):
    """
    Analyze sentiment of given texts
    
    Args:
        texts (list): List of text strings to analyze
    
    Returns:
        list: List of sentiment predictions
    """
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = [sentiment_pipeline(text) for text in texts]
    return results


if __name__ == "__main__":
    pass
