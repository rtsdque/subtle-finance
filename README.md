# Subtle Finance — ML Valuation Engine

A three-methodology valuation engine for S&P 500 companies, combining comparable company analysis, machine learning, and discounted cash flow modeling into a single interactive dashboard.

**Live App:** https://subtlefinance.streamlit.app/ 
**Stack:** Python, XGBoost, scikit-learn, Prophet, Streamlit, Claude API, MLflow, yfinance, SEC EDGAR

## Overview

Subtle Finance answers a single question: *what is a company worth?*

It approaches that question three ways simultaneously:

| Methodology | Approach | Output |
|---|---|---|
| Comparable Company Analysis | Rules-based trading multiples | Side-by-side peer comparison |
| ML Valuation Model | XGBoost regression trained on 400+ companies | Fair value per share + upside/downside % |
| DCF Automation | CAGR-based cash flow forecast + sensitivity analysis | Intrinsic value range across WACC/TGR assumptions |

When the three methodologies agree, that's a signal. When they diverge, that's a more interesting signal — and that's where the Claude AI analyst layer earns its place.

## Features

- **Live data** — real-time prices and financials via yfinance and SEC EDGAR for all 503 S&P 500 companies
- **Comps table** — 25 metrics side by side including margins, multiples, returns, and financial health indicators
- **Altman Z-Score** — bankruptcy risk scoring with safe/grey/distress zone flags, sector-aware handling for financials and REITs
- **ML fair value** — XGBoost model trained on 30 financial features, achieving R² of 0.9176 on held-out test set
- **SHAP explainability** — feature importance showing which financial metrics drove each valuation
- **DCF sensitivity grid** — 7×5 matrix of intrinsic values across WACC (7–13%) and terminal growth rate (1–5%) assumptions
- **Similar company finder** — KNN-based peer discovery using 17-dimensional financial feature space
- **AI analyst chatbot** — Claude-powered discussion of both companies using live financial data as context

## ML Model

**Target:** Log-transformed Enterprise Value (reduces skewness from 8.42 to 0.91)

**Features (30):** Revenue, gross profit, EBITDA, net income, free cash flow, operating cash flow, total debt, total cash, shares outstanding, P/E, forward P/E, P/S, P/B, EV/Revenue, PEG ratio, beta, gross margin, EBITDA margin, net margin, FCF margin, ROE, ROA, debt/equity, current ratio, quick ratio, dividend yield, 52-week position, net debt, P/FCF, Altman Z-Score

**Results:**
- R² Score: 0.9176
- MAE: $19.0B
- MAPE: 14.4%

**Top drivers (SHAP):** Net income, operating cash flow, EBITDA, P/E ratio, forward P/E

## Data Sources

| Source | Data | Cost |
|---|---|---|
| yfinance | Real-time prices, financials, ratios | Free |
| SEC EDGAR API | 10-K/10-Q financial statements | Free |
| Simfin | Supplementary financials | Free tier |
| Wikipedia | S&P 500 constituent list | Free |

## Methodology Notes

**Altman Z-Score** is marked as not applicable for financial services and REIT companies, consistent with standard practice — the formula was designed for manufacturing companies and produces misleading results for banks and insurers.

**DCF limitations** — growth rates are derived from 4–5 years of historical revenue data. Companies trading at significant premiums to DCF intrinsic value (e.g. Apple, Microsoft) reflect market pricing of future growth and intangible value not captured in historical cash flows.

**ML model** is trained on current market prices, meaning it learns market consensus rather than producing a fully independent valuation. The model is most accurate for stable, asset-heavy businesses and less reliable for high-growth or structurally unusual companies.

## Setup

```bash
# clone the repo
git clone https://github.com/yourusername/subtle-finance
cd subtle-finance

# create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# install dependencies
pip install -r requirements.txt

# add API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# run data pipeline (takes ~20 minutes)
jupyter notebook notebooks/01_data_pipeline.ipynb

# launch app
cd app
streamlit run streamlit_app.py
```

## Requirements

See `requirements.txt`. Key dependencies:

- `xgboost` — gradient boosting valuation model
- `scikit-learn` — KNN similarity, preprocessing
- `streamlit` — dashboard framework
- `anthropic` — Claude AI analyst
- `yfinance` — market data
- `mlflow` — experiment tracking
- `shap` — model explainability
- `plotly` — interactive charts

*This application is for analytical and educational purposes only. Nothing here constitutes financial advice.*