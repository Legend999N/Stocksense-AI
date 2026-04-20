"""
components/signals.py — StockSense AI
========================================
Three technical signal cards rendered side-by-side:
  Card 1 — RSI Gauge (dial chart 0-100)
  Card 2 — MACD Histogram (bar chart, last 60 days)
  Card 3 — Bollinger Bands position (progress bar + mini chart)

Each card shows:
  - The indicator's current value
  - A visual (gauge / bar chart / progress)
  - A plain-English signal label (Overbought / Bullish etc.)
  - A coloured status indicator
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
TEAL    = "#00C9A7"
SKY     = "#38BDF8"
GOLD    = "#F4B942"
RED     = "#EF4444"
GREEN   = "#22C55E"
MUTED   = "#8BA3C1"
BG      = "rgba(0,0,0,0)"
GRID    = "#1A2744"
TEXT    = "#E2EBF5"


# ─────────────────────────────────────────────
# RSI GAUGE CARD
# ─────────────────────────────────────────────

def render_rsi_card(signals: dict) -> None:
    """
    RSI gauge dial + label.

    The gauge has three zones:
      0–30   : Oversold zone (green — potential reversal UP)
      30–70  : Neutral zone (grey)
      70–100 : Overbought zone (red — potential reversal DOWN)
    """
    rsi       = signals.get("rsi", 50.0)
    rsi_label = signals.get("rsi_label", "Neutral")

    # Colour based on zone
    if rsi_label == "Overbought":
        needle_color = RED
        status_text  = "⚠ Overbought — possible reversal DOWN"
    elif rsi_label == "Oversold":
        needle_color = GREEN
        status_text  = "⚡ Oversold — possible reversal UP"
    else:
        needle_color = TEAL
        status_text  = "◎ Neutral — no extreme signal"

    # Build gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi,
        number=dict(
            font=dict(size=32, color=TEXT, family="monospace"),
            suffix="",
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=MUTED,
                tickfont=dict(size=10, color=MUTED),
                tickvals=[0, 30, 50, 70, 100],
            ),
            bar=dict(color=needle_color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 30],   color="#0D2A1A"),   # dark green zone
                dict(range=[30, 70],  color="#112240"),   # neutral zone
                dict(range=[70, 100], color="#2A0D0D"),   # dark red zone
            ],
            threshold=dict(
                line=dict(color=needle_color, width=3),
                thickness=0.75,
                value=rsi,
            ),
        ),
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=5),
        paper_bgcolor=BG,
        font=dict(color=TEXT, family="monospace"),
    )

    st.markdown("**RSI — Relative Strength Index**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(status_text)


# ─────────────────────────────────────────────
# MACD HISTOGRAM CARD
# ─────────────────────────────────────────────

def render_macd_card(signals: dict) -> None:
    """
    MACD histogram bar chart (last 60 days).

    Positive bars (green) = MACD above signal = bullish momentum
    Negative bars (red)   = MACD below signal = bearish momentum

    The trend of the bars matters as much as their sign:
      Growing green bars = momentum building UP
      Shrinking green bars = momentum fading
    """
    macd_hist   = signals.get("macd_histogram", pd.Series(dtype=float))
    macd_val    = signals.get("macd", 0.0)
    macd_signal = signals.get("macd_signal", 0.0)
    bullish     = signals.get("macd_bullish", False)

    # Colour each bar
    bar_colors = [GREEN if v >= 0 else RED for v in macd_hist.values]

    fig = go.Figure(go.Bar(
        x=macd_hist.index,
        y=macd_hist.values,
        marker_color=bar_colors,
        marker_opacity=0.8,
        name="MACD Histogram",
    ))

    # Zero line
    fig.add_hline(
        y=0,
        line_color=MUTED,
        line_width=1,
        line_dash="solid",
    )

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=5),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace"),
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=MUTED)),
        yaxis=dict(
            gridcolor=GRID,
            gridwidth=0.5,
            tickfont=dict(size=9, color=MUTED),
            zeroline=False,
        ),
    )

    signal_label = "🟢 Bullish Crossover" if bullish else "🔴 Bearish Crossover"
    st.markdown("**MACD — Momentum**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"{signal_label}  |  MACD: `{macd_val:.4f}`  Signal: `{macd_signal:.4f}`"
    )


# ─────────────────────────────────────────────
# BOLLINGER BANDS CARD
# ─────────────────────────────────────────────

def render_bb_card(signals: dict) -> None:
    """
    Bollinger Bands position indicator.

    BB Position (0–1):
      0.0  = price at lower band (oversold territory)
      0.5  = price at middle band (neutral)
      1.0  = price at upper band (overbought territory)
      >1.0 = price broke above upper band (very extended)
      <0.0 = price broke below lower band (very extended)

    Also shows BB width as a volatility measure.
    """
    bb_pos   = signals.get("bb_position", 0.5)
    bb_width = signals.get("bb_width", 0.0)
    upper    = signals.get("bb_upper", 0.0)
    lower    = signals.get("bb_lower", 0.0)
    middle   = signals.get("bb_middle", 0.0)
    close    = signals.get("latest_close", 0.0)

    # Clamp for display (can be slightly outside 0-1)
    bb_pos_clamped = max(0.0, min(1.0, bb_pos))

    # Colour based on position
    if bb_pos > 0.85:
        pos_color = RED
        pos_label = "⚠ Near upper band — extended"
    elif bb_pos < 0.15:
        pos_color = GREEN
        pos_label = "⚡ Near lower band — potential bounce"
    else:
        pos_color = TEAL
        pos_label = "◎ Mid-range — neutral"

    # Volatility label
    if bb_width > 0.1:
        vol_label = "High volatility"
    elif bb_width > 0.05:
        vol_label = "Normal volatility"
    else:
        vol_label = "Low volatility (squeeze)"

    # Build a simple horizontal gauge using indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bb_pos_clamped * 100,  # convert to 0-100 for gauge
        number=dict(
            font=dict(size=28, color=TEXT, family="monospace"),
            suffix="%",
            valueformat=".0f",
        ),
        title=dict(text="Band Position", font=dict(size=11, color=MUTED)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["Lower", "25", "Mid", "75", "Upper"],
                tickfont=dict(size=9, color=MUTED),
            ),
            bar=dict(color=pos_color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 20],  color="#0D2A1A"),
                dict(range=[20, 80], color="#112240"),
                dict(range=[80, 100], color="#2A0D0D"),
            ],
        ),
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=5),
        paper_bgcolor=BG,
        font=dict(color=TEXT, family="monospace"),
    )

    st.markdown("**Bollinger Bands**")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"{pos_label}  |  Width: `{bb_width:.3f}` ({vol_label})"
    )


# ─────────────────────────────────────────────
# VOLUME METRIC
# ─────────────────────────────────────────────

def render_volume_metric(signals: dict) -> None:
    """Small volume ratio metric shown below signal cards."""
    vol_ratio = signals.get("volume_ratio", 1.0)
    atr_norm  = signals.get("atr_normalised", 0.0)

    col1, col2 = st.columns(2)
    with col1:
        delta_str = f"{(vol_ratio - 1) * 100:+.0f}% vs avg"
        st.metric(
            label="Volume Ratio",
            value=f"{vol_ratio:.2f}×",
            delta=delta_str,
            delta_color="normal",
        )
    with col2:
        st.metric(
            label="ATR (Daily Move %)",
            value=f"{atr_norm * 100:.2f}%",
            help="Average True Range as % of price. Higher = more volatile.",
        )


# ─────────────────────────────────────────────
# MASTER RENDER FUNCTION
# ─────────────────────────────────────────────

def render_signals_section(signals: dict) -> None:
    """
    Render all three signal cards in a horizontal row,
    plus the volume metric row below.

    Args:
        signals : dict from core/api.get_technical_signals()
    """
    if signals.get("error"):
        st.warning(f"Could not load signals: {signals['error']}")
        return

    st.markdown("### Technical Signals")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_rsi_card(signals)

    with col2:
        render_macd_card(signals)

    with col3:
        render_bb_card(signals)

    render_volume_metric(signals)