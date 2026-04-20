"""
components/sentiment.py — StockSense AI
=========================================
News sentiment panel.

Renders:
  1. Overall sentiment score as a coloured progress bar (-1 to +1)
  2. Breakdown: positive / neutral / negative counts as metrics
  3. Scored headlines table — each with a pill badge label
"""

import streamlit as st


# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
TEAL  = "#00C9A7"
RED   = "#EF4444"
GREEN = "#22C55E"
GOLD  = "#F4B942"
MUTED = "#8BA3C1"
TEXT  = "#E2EBF5"


# ─────────────────────────────────────────────
# OVERALL SCORE BAR
# ─────────────────────────────────────────────

def render_sentiment_score(sentiment: dict) -> None:
    """
    Visual sentiment score bar.

    Maps -1.0 → +1.0 to a coloured progress bar:
      < -0.05  : Red   (Negative)
      -0.05 – 0.05 : Grey  (Neutral)
      > 0.05   : Teal  (Positive)
    """
    score = sentiment.get("average_score", 0.0)
    label = sentiment.get("overall_label", "Neutral")

    # Map score (-1 to +1) to progress (0 to 100)
    progress = int((score + 1.0) / 2.0 * 100)

    if label == "Positive":
        bar_color = GREEN
        emoji     = "😊"
    elif label == "Negative":
        bar_color = RED
        emoji     = "😞"
    else:
        bar_color = GOLD
        emoji     = "😐"

    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between;
                        align-items: center; margin-bottom: 6px;">
                <span style="font-family: monospace; font-size: 13px;
                             color: {MUTED}; letter-spacing: 2px;">
                    SENTIMENT SCORE
                </span>
                <span style="font-family: monospace; font-size: 18px;
                             font-weight: bold; color: {bar_color};">
                    {emoji} {score:+.3f} &nbsp; {label.upper()}
                </span>
            </div>
            <div style="background: #1A2744; border-radius: 4px;
                        height: 8px; overflow: hidden;">
                <div style="
                    background: {bar_color};
                    width: {progress}%;
                    height: 100%;
                    border-radius: 4px;
                    transition: width 0.5s ease;
                "></div>
            </div>
            <div style="display: flex; justify-content: space-between;
                        font-size: 10px; color: {MUTED}; margin-top: 3px;
                        font-family: monospace;">
                <span>← Very Negative (-1.0)</span>
                <span>Neutral (0)</span>
                <span>(+1.0) Very Positive →</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# SENTIMENT BREAKDOWN METRICS
# ─────────────────────────────────────────────

def render_sentiment_breakdown(sentiment: dict) -> None:
    """3 metric cards: positive / neutral / negative headline counts."""
    pos   = sentiment.get("positive_count", 0)
    neu   = sentiment.get("neutral_count", 0)
    neg   = sentiment.get("negative_count", 0)
    total = sentiment.get("headline_count", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("😊 Positive", pos,
                  help=f"{pos}/{total} headlines scored positive")
    with col2:
        st.metric("😐 Neutral", neu,
                  help=f"{neu}/{total} headlines scored neutral")
    with col3:
        st.metric("😞 Negative", neg,
                  help=f"{neg}/{total} headlines scored negative")


# ─────────────────────────────────────────────
# HEADLINES TABLE
# ─────────────────────────────────────────────

def render_headlines(sentiment: dict) -> None:
    """
    Render each scored headline with:
      - Coloured pill badge (Positive / Neutral / Negative)
      - Polarity score
      - Truncated headline text
    """
    headlines = sentiment.get("headlines", [])

    if not headlines:
        st.info("No recent headlines found for this ticker.")
        return

    label_colors = {
        "Positive": ("#22C55E", "rgba(34,197,94,0.15)"),
        "Negative": ("#EF4444", "rgba(239,68,68,0.15)"),
        "Neutral":  ("#F4B942", "rgba(244,185,66,0.15)"),
    }

    for h in headlines:
        label    = h.get("label", "Neutral")
        polarity = h.get("polarity", 0.0)
        text     = h.get("text", "")

        # Truncate long headlines
        display_text = text if len(text) <= 100 else text[:97] + "..."

        color, bg = label_colors.get(label, (GOLD, "rgba(244,185,66,0.15)"))

        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 10px;
                padding: 8px 0;
                border-bottom: 1px solid #1A2744;
            ">
                <div style="flex-shrink: 0; margin-top: 2px;">
                    <span style="
                        background: {bg};
                        border: 1px solid {color};
                        color: {color};
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 10px;
                        font-family: monospace;
                        font-weight: bold;
                        white-space: nowrap;
                    ">{label}</span>
                </div>
                <div style="flex: 1;">
                    <span style="font-size: 12.5px; color: {TEXT};
                                 font-family: sans-serif; line-height: 1.4;">
                        {display_text}
                    </span>
                </div>
                <div style="flex-shrink: 0; font-family: monospace;
                            font-size: 11px; color: {color}; margin-top: 2px;">
                    {polarity:+.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# MASTER RENDER FUNCTION
# ─────────────────────────────────────────────

def render_sentiment_section(sentiment: dict) -> None:
    """
    Render the full sentiment panel.

    Args:
        sentiment : dict from core/api.get_sentiment()
    """
    st.markdown("### News Sentiment")

    if sentiment.get("error"):
        st.warning(f"Sentiment unavailable: {sentiment['error']}")
        return

    if sentiment.get("headline_count", 0) == 0:
        st.info(
            "No recent news found for this ticker. "
            "Sentiment score defaulting to Neutral (0.0)."
        )
        return

    render_sentiment_score(sentiment)
    render_sentiment_breakdown(sentiment)

    st.markdown("**Recent Headlines**")
    render_headlines(sentiment)

    st.markdown(
        f"<div style='font-size:10px; color:#546E8A; margin-top:8px;'>"
        f"Sentiment scored using TextBlob polarity analysis · "
        f"{sentiment.get('headline_count', 0)} headlines analysed</div>",
        unsafe_allow_html=True,
    )