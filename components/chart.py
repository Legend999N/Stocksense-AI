"""
components/chart.py — StockSense AI
======================================
Candlestick chart with MA overlays and volume subplot.

Built with Plotly's make_subplots for a two-panel layout:
  Row 1 (75% height): Candlestick + SMA_20 + SMA_50 lines
  Row 2 (25% height): Volume bars (green/red matching candle colour)

Shared x-axis between both panels so zooming/panning syncs them.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ─────────────────────────────────────────────
# COLOUR CONSTANTS (match app theme)
# ─────────────────────────────────────────────
BULL_COLOR  = "#00C9A7"   # teal  — up candles
BEAR_COLOR  = "#EF4444"   # red   — down candles
SMA20_COLOR = "#38BDF8"   # sky blue
SMA50_COLOR = "#F4B942"   # gold
GRID_COLOR  = "#1A2744"
BG_COLOR    = "rgba(0,0,0,0)"   # transparent — uses Streamlit bg
TEXT_COLOR  = "#8BA3C1"


def render_candlestick_chart(chart_data: dict, ticker: str) -> None:
    """
    Render the interactive candlestick chart with MA overlays and volume.

    Args:
        chart_data : dict returned by core/api.get_stock_chart_data()
        ticker     : Stock symbol (for the chart title)
    """
    if chart_data.get("error"):
        st.error(f"Chart error: {chart_data['error']}")
        return

    ohlcv  = chart_data["ohlcv"]
    sma_20 = chart_data["sma_20"]
    sma_50 = chart_data["sma_50"]

    # ── Build figure with 2 rows ──────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # ── Row 1: Candlestick ────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['Open'],
            high=ohlcv['High'],
            low=ohlcv['Low'],
            close=ohlcv['Close'],
            name=ticker,
            increasing_line_color=BULL_COLOR,
            increasing_fillcolor=BULL_COLOR,
            decreasing_line_color=BEAR_COLOR,
            decreasing_fillcolor=BEAR_COLOR,
            line=dict(width=1),
            whiskerwidth=0.3,
        ),
        row=1, col=1,
    )

    # ── Row 1: SMA_20 overlay ─────────────────────────────────────
    # Align SMA series to ohlcv index (they might have different lengths
    # because build_features drops NaN rows for the rolling window)
    sma_20_aligned = sma_20.reindex(ohlcv.index)
    fig.add_trace(
        go.Scatter(
            x=sma_20_aligned.index,
            y=sma_20_aligned.values,
            name="SMA 20",
            line=dict(color=SMA20_COLOR, width=1.5, dash="solid"),
            opacity=0.85,
        ),
        row=1, col=1,
    )

    # ── Row 1: SMA_50 overlay ─────────────────────────────────────
    sma_50_aligned = sma_50.reindex(ohlcv.index)
    fig.add_trace(
        go.Scatter(
            x=sma_50_aligned.index,
            y=sma_50_aligned.values,
            name="SMA 50",
            line=dict(color=SMA50_COLOR, width=1.5, dash="dot"),
            opacity=0.85,
        ),
        row=1, col=1,
    )

    # ── Row 2: Volume bars ────────────────────────────────────────
    # Colour each volume bar to match its candle (green if up, red if down)
    vol_colors = [
        BULL_COLOR if ohlcv['Close'].iloc[i] >= ohlcv['Open'].iloc[i]
        else BEAR_COLOR
        for i in range(len(ohlcv))
    ]

    fig.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv['Volume'],
            name="Volume",
            marker_color=vol_colors,
            marker_opacity=0.5,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Layout styling ────────────────────────────────────────────
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="monospace"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        # Hide the range selector buttons (cleaner look)
        xaxis_rangeslider_visible=False,
    )

    # ── Price axis (row 1) ────────────────────────────────────────
    fig.update_yaxes(
        row=1, col=1,
        gridcolor=GRID_COLOR,
        gridwidth=0.5,
        tickfont=dict(size=11, color=TEXT_COLOR),
        tickprefix="$",
        showgrid=True,
        zeroline=False,
    )

    # ── Volume axis (row 2) ───────────────────────────────────────
    fig.update_yaxes(
        row=2, col=1,
        gridcolor=GRID_COLOR,
        gridwidth=0.5,
        tickfont=dict(size=10, color=TEXT_COLOR),
        showgrid=True,
        zeroline=False,
        tickformat=".2s",   # e.g. "125M" instead of "125000000"
        title_text="Volume",
        title_font=dict(size=10, color=TEXT_COLOR),
    )

    # ── X axis ───────────────────────────────────────────────────
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        gridwidth=0.5,
        tickfont=dict(size=11, color=TEXT_COLOR),
        showgrid=False,
        zeroline=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "displaylogo": False,
    })