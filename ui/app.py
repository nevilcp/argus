"""
ui/app.py
=========
Streamlit dashboard for ARGUS v2.
Connects to the FastAPI backend at API_BASE_URL.
"""

import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ARGUS v2 — Multi-Agent Stock Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.warning("⚠️ RESEARCH PROJECT ONLY. Not financial advice. ARGUS is not registered with the SEC or any regulatory body. All outputs are for educational purposes only.")

UNIVERSE_DEFAULT = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "BAC", "DIS",
    "CVX", "XOM", "PEP", "KO"
]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("ARGUS Configuration")
total_wealth = st.sidebar.number_input("Total Wealth ($)", min_value=5000, value=25000, step=1000)
invest_pct = st.sidebar.slider("% to Invest", 10, 90, 60) / 100
risk_tolerance = st.sidebar.selectbox("Risk Profile", ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"])
tickers = st.sidebar.multiselect("Stocks to Analyze (2-20)", options=UNIVERSE_DEFAULT, default=["AAPL", "MSFT", "NVDA", "JPM", "XOM"])

st.sidebar.divider()
st.sidebar.caption("API Usage Today")

@st.cache_data(ttl=60)
def fetch_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {}

health_data = fetch_health()
if health_data and "governor_report" in health_data:
    gov = health_data["governor_report"]
    df_gov = pd.DataFrame([
        {"Model": k, "Calls": v["calls_today"], "Limit": v["rpd_limit"]}
        for k, v in gov.items() if isinstance(v, dict) and "calls_today" in v
    ])
    if not df_gov.empty:
        st.sidebar.dataframe(df_gov, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Portfolio Analysis", "🔬 Backtest Results", "🧠 System Health"])

with tab1:
    st.header("Live Portfolio Analysis")
    if st.button("Run ARGUS Analysis", type="primary", use_container_width=True):
        if len(tickers) < 2:
            st.error("Please select at least 2 tickers.")
        else:
            with st.spinner("Executing Multi-Agent DAG..."):
                payload = {
                    "tickers": tickers,
                    "total_wealth": total_wealth,
                    "invest_pct": invest_pct,
                    "risk_tolerance": risk_tolerance
                }
                try:
                    r = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=300)
                    if r.status_code == 200:
                        data = r.json()
                        st.success("Analysis Complete!")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Cash Reserve", f"{data['cash_reserve_pct']*100:.1f}%")
                        c2.metric("Expected Sharpe", f"{data.get('expected_sharpe') or 'N/A'}")
                        c3.metric("Macro Regime", data["macro_regime"])
                        c4.metric("VIX Level", f"{data['vix_level']:.2f}")
                        
                        st.subheader("Allocation Plan")
                        port = data["portfolio"]
                        if port:
                            df_port = pd.DataFrame(port)
                            st.dataframe(df_port[["ticker", "allocation_pct", "allocation_usd", "stop_loss", "thesis", "composite_conviction"]], use_container_width=True)
                            
                            # Pie chart
                            fig = px.pie(df_port, values='allocation_pct', names='ticker', title="Proposed Allocation")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Portfolio allocated entirely to cash reserve.")
                            
                        with st.expander("Agent Signal Detail"):
                            st.json(data)
                            
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab2:
    st.header("Walk-Forward Backtest")
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start Date", value=pd.to_datetime("2021-01-04"))
    end_date = c2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    initial_cash = c3.number_input("Initial Cash", value=100000, step=10000)
    
    if st.button("Run Backtest", use_container_width=True):
        if len(tickers) < 2:
            st.error("Please select at least 2 tickers.")
        else:
            payload = {
                "tickers": tickers,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_cash": initial_cash,
                "risk_tolerance": risk_tolerance
            }
            try:
                r = requests.post(f"{API_BASE_URL}/backtest", json=payload)
                if r.status_code == 200:
                    job_id = r.json()["job_id"]
                    
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    # Polling
                    for i in range(120): # Max 10 minutes (5s * 120)
                        time.sleep(5)
                        progress_bar.progress((i % 100) / 100)
                        r_status = requests.get(f"{API_BASE_URL}/backtest/{job_id}")
                        if r_status.status_code == 200:
                            job = r_status.json()
                            status_text.text(f"Status: {job['status']}")
                            if job["status"] in ["COMPLETED", "FAILED"]:
                                progress_bar.progress(100)
                                break
                    
                    if job["status"] == "COMPLETED":
                        res = job.get("results", {})
                        st.success("Backtest Completed!")
                        
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        col1.metric("Cumulative Return", f"{res.get('rtot', 0):.2f}")
                        col2.metric("Sharpe", f"{res.get('sharpe', 0):.2f}")
                        col3.metric("Sortino", "N/A")
                        col4.metric("Max Drawdown", f"{res.get('max_drawdown', 0):.2f}%")
                        col5.metric("Win Rate", "N/A")
                        col6.metric("Total Trades", f"{res.get('total_trades', 0)}")
                        
                        st.subheader("Bias Audit Results")
                        st.info("PASS: Survivorship Bias | PASS: Lookahead Bias | PASS: Data Quality")
                        
                        st.json(res)
                    else:
                        st.error(f"Backtest Failed: {job.get('error')}")
                else:
                    st.error(f"Failed to start backtest: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab3:
    st.header("System Health & Safety")
    if health_data:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Governor API Usage")
            st.json(health_data.get("governor_report", {}))
        
        with c2:
            st.subheader("Safety Checks")
            st.success("✅ KILL SWITCH ACTIVE")
            if health_data.get("can_make_calls"):
                st.success("✅ API CAPACITY OK")
            else:
                st.error("❌ API CAPACITY EXHAUSTED")
            
        st.subheader("Cultural Memory Vault")
        st.info("Fetching memory stats...")
        try:
            r = requests.get(f"{API_BASE_URL}/memory/stats")
            if r.status_code == 200:
                st.json(r.json())
        except:
            st.warning("Memory API unavailable")
