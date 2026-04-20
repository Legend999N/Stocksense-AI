"""
app.py — StockSense AI
========================
Main Streamlit dashboard entry point.

Run with:
    streamlit run app.py

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  SIDEBAR: ticker input, watchlist, model info       │
  ├────────────────────────┬────────────────────────────┤
  │                        │  PREDICTION PANEL          │
  │  CANDLESTICK CHART     │  ↑ UP  |  72% confidence   │
  │  (full width top)      │  Risk badge                 │
  │                        │  Explanation text           │
  ├────────────────────────┤  SHAP waterfall chart      │
  │  RSI | MACD | BB       │  Model metrics             │
  │  signal cards row      ├────────────────────────────┤
  ├────────────────────────┤  SENTIMENT PANEL           │
  │                        │  Score bar + headlines     │
  └────────────────────────┴────────────────────────────┘

The chart takes the full width on top.
Below it, signal cards (left col) sit beside prediction + sentiment (right col).
"""

import os
import sys
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────
# Makes sure all our modules (features/, models/, core/, components/)
# are importable regardless of where streamlit is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.api import (
    get_stock_chart_data,
    get_technical_signals,
    get_prediction,
    get_sentiment,
)
from components.chart      import render_candlestick_chart
from components.signals    import render_signals_section
from components.prediction import render_prediction_section
from components.sentiment  import render_sentiment_section


# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
# Injected into the Streamlit page to override default styles.
# Streamlit has limited styling options natively so we use this
# sparingly for things that really matter: fonts, card borders,
# metric styling, and the disclaimer banner.

st.markdown("""
<style>
/* ── Import a distinctive monospace font ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&display=swap');

/* ── Root variables ── */
:root {
    --teal:    #00C9A7;
    --sky:     #38BDF8;
    --gold:    #F4B942;
    --red:     #EF4444;
    --navy:    #080D14;
    --card:    #112240;
    --muted:   #8BA3C1;
    --text:    #E2EBF5;
}

/* ── Font override for the whole app ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Remove Streamlit header padding ── */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

/* ── Metric cards — add subtle border ── */
[data-testid="stMetric"] {
    background: rgba(17, 34, 64, 0.6);
    border: 1px solid rgba(0, 201, 167, 0.2);
    border-radius: 8px;
    padding: 12px 16px;
}

[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    color: var(--teal) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    letter-spacing: 1px;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: #0D1B2A !important;
    border-right: 1px solid rgba(0, 201, 167, 0.15);
}

/* ── Dividers ── */
hr {
    border-color: rgba(139, 163, 193, 0.2) !important;
    margin: 12px 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080D14; }
::-webkit-scrollbar-thumb { background: #1A2744; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00C9A7; }

/* ── Warning / info boxes ── */
.stAlert {
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Section headers ── */
h3 {
    color: var(--text) !important;
    font-size: 1rem !important;
    letter-spacing: 1px;
    margin-bottom: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar() -> tuple:
    """
    Render the sidebar and return the user's selections.

    Returns:
        (ticker, period) tuple
    """
    with st.sidebar:

        # ── Logo / Title ──────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align: center; padding: 8px 0 16px 0;">
                <div style="font-size: 28px; font-weight: 900;
                            color: #E2EBF5; letter-spacing: -1px;
                            font-family: 'JetBrains Mono', monospace;">
                    📈 StockSense
                </div>
                <div style="font-size: 20px; font-weight: 900;
                            color: #00C9A7; letter-spacing: -1px;
                            font-family: 'JetBrains Mono', monospace;">
                    AI
                </div>
                <div style="font-size: 10px; color: #546E8A;
                            letter-spacing: 2px; margin-top: 2px;">
                    BY SKILLOWLS
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Ticker input ──────────────────────────────────────────
        st.markdown(
            "<div style='font-size:11px; color:#8BA3C1; "
            "letter-spacing:2px; margin-bottom:4px;'>TICKER SYMBOL</div>",
            unsafe_allow_html=True,
        )
        ticker_input = st.text_input(
            label="ticker",
            value="AAPL",
            max_chars=10,
            label_visibility="collapsed",
            placeholder="e.g. AAPL, TSLA, GOOGL",
        ).strip().upper()

        # ── Watchlist quick-select ────────────────────────────────
        st.markdown(
            "<div style='font-size:11px; color:#8BA3C1; "
            "letter-spacing:2px; margin: 12px 0 6px 0;'>WATCHLIST</div>",
            unsafe_allow_html=True,
        )

        watchlist  = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
        cols       = st.columns(3)
        selected_quick = None
        for i, wt in enumerate(watchlist):
            with cols[i % 3]:
                if st.button(wt, use_container_width=True, key=f"wl_{wt}"):
                    selected_quick = wt

        # If a watchlist button was clicked, use that ticker
        ticker = selected_quick if selected_quick else ticker_input

        if not ticker:
            ticker = "AAPL"

        st.divider()

        # ── Chart period ──────────────────────────────────────────
        st.markdown(
            "<div style='font-size:11px; color:#8BA3C1; "
            "letter-spacing:2px; margin-bottom:4px;'>CHART PERIOD</div>",
            unsafe_allow_html=True,
        )
        period = st.select_slider(
            label="period",
            options=["3mo", "6mo", "1y", "2y", "3y"],
            value="1y",
            label_visibility="collapsed",
        )

        st.divider()

        # ── Analyse button ────────────────────────────────────────
        analyse = st.button(
            "🔍  ANALYSE",
            use_container_width=True,
            type="primary",
        )

        st.divider()

        # ── Info section ──────────────────────────────────────────
        st.markdown(
            """
            <div style="font-size: 10px; color: #546E8A; line-height: 1.7;
                        font-family: 'JetBrains Mono', monospace;">
                <b style="color:#8BA3C1;">How it works</b><br>
                1. Downloads 3yr OHLCV data<br>
                2. Computes 40+ technical indicators<br>
                3. Scores news sentiment (TextBlob)<br>
                4. XGBoost predicts UP / DOWN<br>
                5. SHAP explains the decision<br>
                <br>
                <b style="color:#EF4444;">⚠ Disclaimer</b><br>
                For educational use only.<br>
                Not financial advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return ticker, period, analyse


# ─────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────

def render_header(ticker: str, signals: dict) -> None:
    """Top header bar with ticker info and key stats."""
    close = signals.get("latest_close", 0.0)
    date  = signals.get("latest_date", "")
    err   = signals.get("error")

    if err:
        st.markdown(
            f"<h2 style='color:#E2EBF5; font-family:JetBrains Mono;'>"
            f"📈 {ticker}</h2>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: baseline;
            gap: 16px;
            margin-bottom: 8px;
        ">
            <span style="font-size: 26px; font-weight: 900;
                         color: #E2EBF5; font-family: 'JetBrains Mono', monospace;">
                📈 {ticker}
            </span>
            <span style="font-size: 22px; font-weight: 700;
                         color: #00C9A7; font-family: 'JetBrains Mono', monospace;">
                ${close:,.2f}
            </span>
            <span style="font-size: 12px; color: #8BA3C1;
                         font-family: 'JetBrains Mono', monospace;">
                as of {date}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# LOADING STATE
# ─────────────────────────────────────────────

def load_all_data(ticker: str, period: str) -> tuple:
    """
    Fetch all 4 data sources with a spinner.
    Returns all results even if some fail — each component
    handles its own error state gracefully.
    """
    with st.spinner(f"Fetching data for {ticker}..."):
        chart_data = get_stock_chart_data(ticker, period)

    with st.spinner("Computing technical indicators..."):
        signals    = get_technical_signals(ticker)

    with st.spinner("Running prediction model..."):
        prediction = get_prediction(ticker)

    with st.spinner("Analysing news sentiment..."):
        sentiment  = get_sentiment(ticker)

    return chart_data, signals, prediction, sentiment


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # ── Sidebar (returns selections) ──────────────────────────────
    ticker, period, analyse = render_sidebar()

    # ── Session state — persist last loaded ticker ─────────────────
    # Without this, Streamlit reruns on every widget interaction,
    # which would re-fetch data every time someone clicks anything.
    if "loaded_ticker" not in st.session_state:
        st.session_state.loaded_ticker = None
        st.session_state.loaded_period = None

    # Load data if: first run, new ticker, new period, or Analyse clicked
    should_load = (
        analyse
        or st.session_state.loaded_ticker != ticker
        or st.session_state.loaded_period != period
    )

    if should_load:
        st.session_state.loaded_ticker = ticker
        st.session_state.loaded_period = period

        (
            st.session_state.chart_data,
            st.session_state.signals,
            st.session_state.prediction,
            st.session_state.sentiment,
        ) = load_all_data(ticker, period)

    # ── Get data from session state ────────────────────────────────
    chart_data = st.session_state.get("chart_data", {"error": "No data loaded yet."})
    signals    = st.session_state.get("signals",    {"error": "No data loaded yet."})
    prediction = st.session_state.get("prediction", {"error": "No data loaded yet."})
    sentiment  = st.session_state.get("sentiment",  {"headline_count": 0})

    # ── Header ─────────────────────────────────────────────────────
    render_header(ticker, signals)

    # ── Candlestick chart (full width) ─────────────────────────────
    render_candlestick_chart(chart_data, ticker)

    st.divider()

    # ── Bottom section: left col (signals) + right col (prediction + sentiment) ──
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        render_signals_section(signals)

    with right_col:
        render_prediction_section(prediction)
        st.divider()
        render_sentiment_section(sentiment)

    # ── Footer ─────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 20px 0 8px 0;
            font-size: 10px;
            color: #546E8A;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 1px;
        ">
            StockSense AI · Built by SkillOwls ·
            For educational purposes only · Not financial advice
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()