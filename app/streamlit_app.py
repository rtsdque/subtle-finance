import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import anthropic
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(
    page_title="Subtle Finance",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data():
    master_df = pd.read_csv("../data/processed/master_features.csv")
    final_df = pd.read_csv("../data/processed/final_valuations.csv")
    knn_df = pd.read_csv("../data/processed/knn_features.csv")
    dcf_df = pd.read_csv("../data/processed/dcf_valuations.csv")
    return master_df, final_df, knn_df, dcf_df

@st.cache_resource
def load_models():
    with open("../data/processed/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("../data/processed/knn_model.pkl", "rb") as f:
        knn_model = pickle.load(f)
    with open("../data/processed/knn_scaler.pkl", "rb") as f:
        knn_scaler = pickle.load(f)
    with open("../data/processed/model_features.pkl", "rb") as f:
        model_features = pickle.load(f)
    return xgb_model, knn_model, knn_scaler, model_features

master_df, final_df, knn_df, dcf_df = load_data()
xgb_model, knn_model, knn_scaler, model_features = load_models()

st.title("📈 Subtle Finance")
st.caption("A three-methodology valuation engine for S&P 500 companies.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    ticker1 = st.text_input("Ticker 1", value="AAPL", max_chars=10).upper().strip()
with col2:
    ticker2 = st.text_input("Ticker 2", value="MSFT", max_chars=10).upper().strip()

analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze:
    if not ticker1 or not ticker2:
        st.error("Please enter both ticker symbols before analyzing.")
    
    else:
        st.session_state['ticker1'] = ticker1
        st.session_state['ticker2'] = ticker2
        st.session_state['analyzed'] = True
        st.session_state['messages'] = []

if st.session_state.get('analyzed'):
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    ticker1 = st.session_state['ticker1']
    ticker2 = st.session_state['ticker2']

    c1 = final_df[final_df['symbol'] == ticker1]
    c2 = final_df[final_df['symbol'] == ticker2]

    if c1.empty:
        st.error(f"Ticker '{ticker1}' not found in S&P 500 dataset.")
        st.stop()
    if c2.empty:
        st.error(f"Ticker '{ticker2}' not found in S&P 500 dataset.")
        st.stop()

    c1 = c1.iloc[0]
    c2 = c2.iloc[0]

    master_c1 = master_df[master_df['symbol'] == ticker1].iloc[0]
    master_c2 = master_df[master_df['symbol'] == ticker2].iloc[0]

    # --- header ---
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{c1['name']}")
        st.caption(f"{c1['sector']} · {ticker1}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${c1['current_price']:,.2f}")
        m2.metric("Market Cap", f"${c1['market_cap']/1e9:,.1f}B")
        m3.metric("Altman Z-Score", f"{c1['altman_z_score']:,.2f}" if pd.notna(c1['altman_z_score']) else "N/A")

    with col2:
        st.subheader(f"{c2['name']}")
        st.caption(f"{c2['sector']} · {ticker2}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${c2['current_price']:,.2f}")
        m2.metric("Market Cap", f"${c2['market_cap']/1e9:,.1f}B")
        m3.metric("Altman Z-Score", f"{c2['altman_z_score']:,.2f}" if pd.notna(c2['altman_z_score']) else "N/A")

    # --- valuation summary ---
    st.divider()
    st.subheader("Valuation Summary")

    col1, col2 = st.columns(2)

    def valuation_block(c, ticker):
        current = c['current_price']
        ml_val = c['ml_fair_value']
        dcf_val = c['dcf_base_case']
        ml_upside = c['ml_upside']
        dcf_upside = c['dcf_upside']

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${current:,.2f}")

        if pd.notna(ml_val):
            m2.metric("ML Fair Value", f"${ml_val:,.2f}",
                     delta=f"{ml_upside:+.1f}%",
                     delta_color="normal")
        else:
            m2.metric("ML Fair Value", "N/A")

        if pd.notna(dcf_val) and dcf_val > 0:
            m3.metric("DCF Base Case", f"${dcf_val:,.2f}",
                     delta=f"{dcf_upside:+.1f}%",
                     delta_color="normal")
        else:
            m3.metric("DCF Base Case", "N/A")

        if pd.notna(ml_val) and pd.notna(dcf_val) and dcf_val > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Current Price', 'ML Fair Value', 'DCF Base Case'],
                y=[current, ml_val, dcf_val],
                marker_color=['#636EFA', '#00CC96', '#EF553B'],
                text=[f'${current:,.0f}', f'${ml_val:,.0f}', f'${dcf_val:,.0f}'],
                textposition='outside'
            ))
            fig.update_layout(
                title=f"{ticker} — Valuation Comparison",
                yaxis_title="Price per Share ($)",
                showlegend=False,
                height=350,
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col1:
        valuation_block(c1, ticker1)
    with col2:
        valuation_block(c2, ticker2)

    # --- comps table ---
    st.divider()
    st.subheader("Comparable Company Analysis")

    def fmt(val, prefix='', suffix='', decimals=2, billions=False):
        if pd.isna(val):
            return 'N/A'
        if billions:
            return f"{prefix}{val/1e9:,.1f}B{suffix}"
        return f"{prefix}{val:,.{decimals}f}{suffix}"

    comps_data = {
        'Metric': [
            'Current Price', 'Market Cap', 'Enterprise Value', 'Revenue',
            'Gross Margin', 'EBITDA Margin', 'Net Margin',
            'P/E Ratio', 'Forward P/E', 'EV/EBITDA', 'EV/Revenue',
            'P/S Ratio', 'P/B Ratio', 'P/FCF',
            'ROE', 'ROA', 'Debt/Equity', 'Current Ratio',
            'Beta', '52W Position', 'Dividend Yield',
            'Altman Z-Score', 'Z-Score Zone',
            'Analyst Target', 'Recommendation'
        ],
        ticker1: [
            fmt(master_c1['current_price'], '$'),
            fmt(master_c1['market_cap'], '$', billions=True),
            fmt(master_c1['enterprise_value'], '$', billions=True),
            fmt(master_c1['revenue'], '$', billions=True),
            fmt(master_c1['gross_margin']*100 if pd.notna(master_c1['gross_margin']) else None, suffix='%'),
            fmt(master_c1['ebitda_margin']*100 if pd.notna(master_c1['ebitda_margin']) else None, suffix='%'),
            fmt(master_c1['net_margin']*100 if pd.notna(master_c1['net_margin']) else None, suffix='%'),
            fmt(master_c1['pe_ratio']),
            fmt(master_c1['forward_pe']),
            fmt(master_c1['ev_ebitda']),
            fmt(master_c1['ev_revenue']),
            fmt(master_c1['ps_ratio']),
            fmt(master_c1['pb_ratio']),
            fmt(master_c1['p_fcf']),
            fmt(master_c1['roe']*100 if pd.notna(master_c1['roe']) else None, suffix='%'),
            fmt(master_c1['roa']*100 if pd.notna(master_c1['roa']) else None, suffix='%'),
            fmt(master_c1['debt_to_equity']),
            fmt(master_c1['current_ratio']),
            fmt(master_c1['beta']),
            fmt(master_c1['fifty_two_week_position'], suffix='%'),
            fmt(master_c1['dividend_yield'], suffix='%'),
            fmt(master_c1['altman_z_score'], decimals=3),
            str(master_c1['z_score_zone']) if pd.notna(master_c1['z_score_zone']) else 'N/A',
            fmt(master_c1['analyst_target_price'], '$'),
            str(master_c1['recommendation']) if pd.notna(master_c1['recommendation']) else 'N/A',
        ],
        ticker2: [
            fmt(master_c2['current_price'], '$'),
            fmt(master_c2['market_cap'], '$', billions=True),
            fmt(master_c2['enterprise_value'], '$', billions=True),
            fmt(master_c2['revenue'], '$', billions=True),
            fmt(master_c2['gross_margin']*100 if pd.notna(master_c2['gross_margin']) else None, suffix='%'),
            fmt(master_c2['ebitda_margin']*100 if pd.notna(master_c2['ebitda_margin']) else None, suffix='%'),
            fmt(master_c2['net_margin']*100 if pd.notna(master_c2['net_margin']) else None, suffix='%'),
            fmt(master_c2['pe_ratio']),
            fmt(master_c2['forward_pe']),
            fmt(master_c2['ev_ebitda']),
            fmt(master_c2['ev_revenue']),
            fmt(master_c2['ps_ratio']),
            fmt(master_c2['pb_ratio']),
            fmt(master_c2['p_fcf']),
            fmt(master_c2['roe']*100 if pd.notna(master_c2['roe']) else None, suffix='%'),
            fmt(master_c2['roa']*100 if pd.notna(master_c2['roa']) else None, suffix='%'),
            fmt(master_c2['debt_to_equity']),
            fmt(master_c2['current_ratio']),
            fmt(master_c2['beta']),
            fmt(master_c2['fifty_two_week_position'], suffix='%'),
            fmt(master_c2['dividend_yield'], suffix='%'),
            fmt(master_c2['altman_z_score'], decimals=3),
            str(master_c2['z_score_zone']) if pd.notna(master_c2['z_score_zone']) else 'N/A',
            fmt(master_c2['analyst_target_price'], '$'),
            str(master_c2['recommendation']) if pd.notna(master_c2['recommendation']) else 'N/A',
        ]
    }

    comps_table = pd.DataFrame(comps_data)
    st.dataframe(comps_table, use_container_width=True, hide_index=True)

    # --- similar companies ---
    st.divider()
    st.subheader("Similar Companies")

    def show_similar(ticker, label):
        st.markdown(f"**{label} peers**")
        if ticker not in knn_df['symbol'].values:
            st.caption("Not available in similarity dataset.")
            return

        idx = knn_df[knn_df['symbol'] == ticker].index[0]
        ticker_sector = knn_df.loc[idx, 'sector']
        sector_df = knn_df[knn_df['sector'] == ticker_sector].copy()

        if len(sector_df) < 2:
            st.caption("Not enough sector peers.")
            return

        knn_features = [col for col in knn_df.columns if col not in ['symbol', 'name', 'sector']]
        sector_scaled = knn_scaler.transform(sector_df[knn_features].fillna(0))
        sector_model = NearestNeighbors(n_neighbors=min(6, len(sector_df)), metric='euclidean')
        sector_model.fit(sector_scaled)

        ticker_idx = sector_df.index.get_loc(idx)
        distances, indices = sector_model.kneighbors([sector_scaled[ticker_idx]])

        similar = sector_df.iloc[indices[0][1:6]][['symbol', 'name', 'sector']].copy()
        similar['market_cap'] = similar['symbol'].map(master_df.set_index('symbol')['market_cap'])
        similar['pe_ratio'] = similar['symbol'].map(master_df.set_index('symbol')['pe_ratio'])
        similar['market_cap'] = similar['market_cap'].apply(lambda x: f"${x/1e9:,.1f}B" if pd.notna(x) else 'N/A')
        similar['pe_ratio'] = similar['pe_ratio'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else 'N/A')
        similar.columns = ['Symbol', 'Company', 'Sector', 'Market Cap', 'P/E']
        st.dataframe(similar, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        show_similar(ticker1, ticker1)
    with col2:
        show_similar(ticker2, ticker2)

    # --- dcf sensitivity ---
    st.divider()
    st.subheader("DCF Sensitivity Analysis")
    st.caption("Intrinsic value per share across WACC and terminal growth rate assumptions.")

    def show_dcf_sensitivity(ticker, label):
        st.markdown(f"**{label}**")
        try:
            income_stmt = pd.read_csv(f"../data/raw/{ticker}/income_statement.csv", index_col=0)
            cash_flow_stmt = pd.read_csv(f"../data/raw/{ticker}/cash_flow.csv", index_col=0)

            def get_val(df, names):
                for name in names:
                    if name in df.index:
                        vals = df.loc[name].dropna()
                        if len(vals) > 0:
                            return vals.astype(float)
                return None

            revenue = get_val(income_stmt, ['Total Revenue'])
            fcf = get_val(cash_flow_stmt, ['Free Cash Flow'])

            if revenue is None or fcf is None or len(revenue) < 2:
                st.caption("Insufficient data for DCF.")
                return

            revenue = revenue.sort_index()
            fcf = fcf.sort_index()

            years = len(revenue)
            cagr = (revenue.iloc[-1] / revenue.iloc[0]) ** (1/(years-1)) - 1
            cagr = max(min(cagr, 0.30), -0.10)
            avg_fcf_margin = max(min((fcf / revenue).mean(), 0.40), 0.01)
            latest_revenue = revenue.iloc[-1]

            wacc_range = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
            tgr_range = [0.01, 0.02, 0.03, 0.04, 0.05]

            company_row = master_df[master_df['symbol'] == ticker].iloc[0]
            shares = company_row['shares_outstanding']
            net_debt = company_row['net_debt'] if pd.notna(company_row['net_debt']) else 0

            projected_fcf = [latest_revenue * (1 + cagr) ** i * avg_fcf_margin for i in range(1, 6)]

            sensitivity = {}
            for wacc in wacc_range:
                row = {}
                for tgr in tgr_range:
                    if wacc <= tgr:
                        row[f"TGR {tgr*100:.0f}%"] = 'N/A'
                        continue
                    pv_fcfs = sum([f / (1+wacc)**i for i, f in enumerate(projected_fcf, 1)])
                    tv = projected_fcf[-1] * (1+tgr) / (wacc - tgr)
                    pv_tv = tv / (1+wacc)**5
                    ev = pv_fcfs + pv_tv
                    equity = ev - net_debt
                    per_share = equity / shares if shares else None
                    row[f"TGR {tgr*100:.0f}%"] = f"${per_share:,.0f}" if per_share and per_share > 0 else 'N/A'
                sensitivity[f"WACC {wacc*100:.0f}%"] = row

            sensitivity_df = pd.DataFrame(sensitivity).T
            company_row2 = master_df[master_df['symbol'] == ticker].iloc[0]
            current_price = company_row2['current_price']
            st.caption(f"Current price: ${current_price:,.2f} | Revenue CAGR: {cagr*100:.1f}% | FCF Margin: {avg_fcf_margin*100:.1f}%")
            st.dataframe(sensitivity_df, use_container_width=True)

        except Exception as e:
            st.caption(f"DCF sensitivity unavailable: {e}")

    col1, col2 = st.columns(2)
    with col1:
        show_dcf_sensitivity(ticker1, ticker1)
    with col2:
        show_dcf_sensitivity(ticker2, ticker2)

    # --- ai analyst ---
    st.divider()
    st.subheader("AI Analyst")
    st.caption("Ask anything about these two companies. Not financial advice.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    context = f"""
    You are a financial analyst assistant discussing {ticker1} ({c1['name']}) and {ticker2} ({c2['name']}).

    Here is the data for both companies:

    {ticker1} — {c1['name']}:
    - Current Price: ${c1['current_price']:,.2f}
    - Market Cap: ${c1['market_cap']/1e9:,.1f}B
    - ML Fair Value: ${c1['ml_fair_value']:,.2f} ({c1['ml_upside']:+.1f}%)
    - DCF Base Case: ${c1['dcf_base_case']:,.2f} ({c1['dcf_upside']:+.1f}%)
    - P/E Ratio: {master_c1['pe_ratio']:.2f}
    - EV/EBITDA: {master_c1['ev_ebitda']:.2f}
    - Gross Margin: {master_c1['gross_margin']*100:.1f}%
    - Net Margin: {master_c1['net_margin']*100:.1f}%
    - Altman Z-Score: {c1['altman_z_score']:.3f} ({c1['z_score_zone']})
    - Analyst Recommendation: {master_c1['recommendation']}

    {ticker2} — {c2['name']}:
    - Current Price: ${c2['current_price']:,.2f}
    - Market Cap: ${c2['market_cap']/1e9:,.1f}B
    - ML Fair Value: ${c2['ml_fair_value']:,.2f} ({c2['ml_upside']:+.1f}%)
    - DCF Base Case: ${c2['dcf_base_case']:,.2f} ({c2['dcf_upside']:+.1f}%)
    - P/E Ratio: {master_c2['pe_ratio']:.2f}
    - EV/EBITDA: {master_c2['ev_ebitda']:.2f}
    - Gross Margin: {master_c2['gross_margin']*100:.1f}%
    - Net Margin: {master_c2['net_margin']*100:.1f}%
    - Altman Z-Score: {c2['altman_z_score']:.3f} ({c2['z_score_zone']})
    - Analyst Recommendation: {master_c2['recommendation']}

    Discuss the companies using this data. Be analytical and insightful.
    Never give direct investment advice or tell the user to buy or sell.
    Always note that this is not financial advice.
    """

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about these companies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            client = anthropic.Anthropic()
            with client.messages.stream(
                model="claude-sonnet-4-5",
                max_tokens=1000,
                system=context,
                messages=[{"role": m["role"], "content": m["content"]}
                         for m in st.session_state.messages]
            ) as stream:
                response = st.write_stream(stream.text_stream)

        st.session_state.messages.append({"role": "assistant", "content": response})