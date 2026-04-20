"""
components/prediction.py — StockSense AI
==========================================
Prediction panel: the centrepiece of the dashboard.

Renders:
  1. Big UP/DOWN signal with confidence score
  2. Risk level badge (Low / Medium / High)
  3. SHAP waterfall bar chart — top features pushing UP and DOWN
  4. Plain-English explanation text
  5. Model quality metrics (accuracy, ROC-AUC)
"""

import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
TEAL  = "#00C9A7"
SKY   = "#38BDF8"
RED   = "#EF4444"
GREEN = "#22C55E"
GOLD  = "#F4B942"
MUTED = "#8BA3C1"
BG    = "rgba(0,0,0,0)"
TEXT  = "#E2EBF5"


# ─────────────────────────────────────────────
# DIRECTION SIGNAL CARD
# ─────────────────────────────────────────────

def render_direction_card(prediction: dict) -> None:
    """
    The big UP/DOWN prediction with confidence score and risk badge.

    Uses st.markdown with inline HTML for coloured styling that
    Streamlit's native components can't achieve.
    """
    direction  = prediction.get("direction", "Unknown")
    confidence = prediction.get("confidence", 0.5)
    risk_level = prediction.get("risk_level", "Medium")
    pred_date  = prediction.get("prediction_date", "")

    # Direction styling
    if direction == "UP":
        arrow     = "↑"
        dir_color = "#00C9A7"   # teal
        dir_bg    = "rgba(0, 201, 167, 0.12)"
    else:
        arrow     = "↓"
        dir_color = "#EF4444"   # red
        dir_bg    = "rgba(239, 68, 68, 0.12)"

    # Risk badge colour
    risk_colors = {"Low": "#22C55E", "Medium": "#F4B942", "High": "#EF4444"}
    risk_color  = risk_colors.get(risk_level, GOLD)

    conf_pct = confidence * 100

    st.markdown(
        f"""
        <div style="
            background: {dir_bg};
            border: 1px solid {dir_color};
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <div style="font-size: 14px; color: {MUTED}; letter-spacing: 3px;
                        font-family: monospace; margin-bottom: 4px;">
                TOMORROW'S PREDICTION
            </div>
            <div style="font-size: 72px; font-weight: 900; color: {dir_color};
                        line-height: 1; font-family: monospace;">
                {arrow} {direction}
            </div>
            <div style="font-size: 22px; color: {TEXT}; margin-top: 8px;
                        font-family: monospace;">
                {conf_pct:.0f}% confidence
            </div>
            <div style="margin-top: 12px;">
                <span style="
                    background: rgba({_hex_to_rgb(risk_color)}, 0.2);
                    border: 1px solid {risk_color};
                    color: {risk_color};
                    padding: 3px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-family: monospace;
                    letter-spacing: 2px;
                ">
                    {risk_level.upper()} RISK
                </span>
            </div>
            <div style="font-size: 11px; color: {MUTED}; margin-top: 10px;
                        font-family: monospace;">
                Based on data as of {pred_date}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R, G, B' string for rgba() CSS."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"


# ─────────────────────────────────────────────
# SHAP WATERFALL CHART
# ─────────────────────────────────────────────

def render_shap_chart(prediction: dict) -> None:
    """
    Horizontal bar chart showing top features pushing UP (green)
    and top features pushing DOWN (red).

    This is the "why" of the prediction — the key differentiator
    that makes StockSense AI explainable vs a black box.

    Layout:
      Feature name | ←← bar (if negative/DOWN) | bar (if positive/UP) →→
    """
    top_positive = prediction.get("top_positive", [])
    top_negative = prediction.get("top_negative", [])

    if not top_positive and not top_negative:
        st.caption("SHAP data not available — run explain.py first")
        return

    # Combine and sort by absolute SHAP value for the chart
    all_shap = top_positive[:5] + top_negative[:5]
    if not all_shap:
        return

    # Sort by value so bars flow naturally (negative left, positive right)
    all_shap_sorted = sorted(all_shap, key=lambda x: x[1])

    features = [item[0] for item in all_shap_sorted]
    values   = [item[1] for item in all_shap_sorted]
    colors   = [GREEN if v >= 0 else RED for v in values]

    # Shorten long feature names for display
    def shorten(name: str) -> str:
        replacements = {
            'normalised': 'norm',
            'histogram':  'hist',
            'sentiment_score': 'sentiment',
            'volume_ratio': 'vol_ratio',
            'rolling_std': 'roll_std',
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name[:28]   # cap at 28 chars

    display_names = [shorten(f) for f in features]

    fig = go.Figure(go.Bar(
        x=values,
        y=display_names,
        orientation='h',
        marker_color=colors,
        marker_opacity=0.85,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=TEXT, family="monospace"),
        cliponaxis=False,
    ))

    # Zero line
    fig.add_vline(x=0, line_color=MUTED, line_width=1)

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=40, t=10, b=5),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace", size=10),
        xaxis=dict(
            showgrid=True,
            gridcolor="#1A2744",
            zeroline=False,
            tickfont=dict(size=9, color=MUTED),
            title=dict(text="← DOWN    UP →", font=dict(size=10, color=MUTED)),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=10, color=TEXT),
        ),
    )

    st.markdown("**Why this prediction?** *(SHAP feature contributions)*")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "🟢 Green bars pushed prediction toward **UP** · "
        "🔴 Red bars pushed prediction toward **DOWN**"
    )


# ─────────────────────────────────────────────
# EXPLANATION TEXT
# ─────────────────────────────────────────────

def render_explanation(prediction: dict) -> None:
    """Plain-English explanation from generate_explanation() in predict.py."""
    explanation = prediction.get("explanation", "")
    if not explanation:
        return

    st.markdown(
        f"""
        <div style="
            background: rgba(17, 34, 64, 0.8);
            border-left: 3px solid #38BDF8;
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin: 8px 0;
            font-family: monospace;
            font-size: 13px;
            color: {TEXT};
            line-height: 1.6;
        ">
            💡 {explanation}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MODEL QUALITY METRICS
# ─────────────────────────────────────────────

def render_model_metrics(prediction: dict) -> None:
    """
    Small row showing model training accuracy + ROC-AUC.
    Reminds the user what the model's baseline performance is.
    """
    accuracy  = prediction.get("model_accuracy")
    roc_auc   = prediction.get("model_roc_auc")
    uses_sent = prediction.get("uses_sentiment", False)

    if accuracy is None:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Model Accuracy",
            f"{accuracy*100:.1f}%",
            help="Accuracy on held-out test set (last 20% of historical data)",
        )
    with col2:
        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}" if roc_auc else "N/A",
            help="0.5 = random, 1.0 = perfect. >0.55 is meaningful on financial data.",
        )
    with col3:
        sent_str = "✓ Included" if uses_sent else "✗ Not used"
        st.metric(
            "Sentiment",
            sent_str,
            help="Whether the TextBlob sentiment score improved this model.",
        )


# ─────────────────────────────────────────────
# MASTER RENDER FUNCTION
# ─────────────────────────────────────────────

def render_prediction_section(prediction: dict) -> None:
    """
    Render the full prediction panel.

    Args:
        prediction : dict from core/api.get_prediction()
    """
    if prediction.get("error"):
        st.warning(
            f"⚠ Prediction unavailable: {prediction['error']}",
        )
        return

    st.markdown("### Prediction")
    render_direction_card(prediction)
    render_explanation(prediction)
    render_shap_chart(prediction)

    st.markdown("---")
    st.markdown("**Model Performance** *(on held-out test data)*")
    render_model_metrics(prediction)

    # Disclaimer
    st.markdown(
        "<div style='font-size:10px; color:#546E8A; margin-top:8px;'>"
        "⚠ For educational purposes only. Not financial advice."
        "</div>",
        unsafe_allow_html=True,
    )