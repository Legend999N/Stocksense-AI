# StockSense AI 📈

> Intelligent Market Trend Analyzer & Forecasting System

Combines technical analysis (RSI, MACD, Bollinger Bands) with NLP
sentiment analysis to predict short-term stock price direction,
with SHAP-based explainability.


## Status
- [x] Day 1 — Data pipeline + Feature engineering
- [x] Day 2 — ML model + Sentiment pipeline
- [x] Day 3 — Streamlit dashboard
- [ ] Day 4 — Polish + Deployment

## Setup

```bash
git clone https://github.com/Legend999N/Stocksense-AI.git
cd stocksense-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python day1_pipeline.py
python day2-pipeline.py
python app.py
```

## Project Structure