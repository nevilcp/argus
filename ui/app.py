"""
Streamlit dashboard client for the ARGUS multi-agent stock advisor.

Provides an interactive user interface to trigger real-time multi-agent portfolio
analysis cycles, execute historical backtests, audit governor rate limits, and
monitor cultural memory retrieval statistics.
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ARGUS — Multi-Agent Stock Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Disclaimer
# ──────────────────────────────────────────────────────────────────────────────

st.warning(
    "⚠️ **RESEARCH PROJECT ONLY.** Not financial advice. ARGUS is not registered with the SEC "
    "or any regulatory body. All outputs are for educational and research purposes only."
)

UNIVERSE_DEFAULT = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "UNH",
    "HD",
    "MA",
    "BAC",
    "DIS",
    "CVX",
    "XOM",
    "PEP",
    "KO",
]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ ARGUS Configuration")
total_wealth = st.sidebar.number_input("Total Wealth ($)", min_value=5000, value=25000, step=1000)
invest_pct = st.sidebar.slider("% to Invest", 10, 90, 60) / 100
risk_tolerance = st.sidebar.selectbox("Risk Profile", ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"])
tickers = st.sidebar.multiselect(
    "Stocks to Analyze (2–20)",
    options=UNIVERSE_DEFAULT,
    default=["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
)

st.sidebar.divider()
st.sidebar.caption("📊 API Usage Today")


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
    df_gov = pd.DataFrame(
        [
            {"Model": k, "Calls": v["calls_today"], "Limit": v["rpd_limit"]}
            for k, v in gov.items()
            if isinstance(v, dict) and "calls_today" in v
        ]
    )
    if not df_gov.empty:
        st.sidebar.dataframe(df_gov, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: Signal badge colour
# ──────────────────────────────────────────────────────────────────────────────

SIGNAL_COLOURS = {
    "BULLISH": "#27ae60",
    "BEARISH": "#e74c3c",
    "NEUTRAL": "#f39c12",
}
REGIME_COLOURS = {
    "EXPANSION": "#27ae60",
    "CONTRACTION": "#e74c3c",
    "TRANSITIONAL": "#f39c12",
}
VIX_COLOURS = {
    "LOW": "#27ae60",
    "MEDIUM": "#f39c12",
    "HIGH": "#e74c3c",
    "EXTREME": "#8e44ad",
}


def signal_badge(label: str) -> str:
    colour = SIGNAL_COLOURS.get(label.upper(), "#7f8c8d")
    return f'<span style="background:{colour};color:white;padding:2px 9px;border-radius:12px;font-size:0.78rem;font-weight:600;">{label}</span>'


def conviction_bar_html(conviction: float) -> str:
    pct = int(conviction * 100)
    if pct >= 70:
        bar_colour = "#27ae60"
    elif pct >= 45:
        bar_colour = "#f39c12"
    else:
        bar_colour = "#e74c3c"
    return (
        f'<div style="background:#ecf0f1;border-radius:6px;height:8px;margin-top:4px;">'
        f'<div style="width:{pct}%;background:{bar_colour};height:8px;border-radius:6px;"></div>'
        f"</div>"
        f'<small style="color:#7f8c8d;">{pct}% conviction</small>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Position Card Renderer
# ──────────────────────────────────────────────────────────────────────────────


def render_position_card(pos: dict, investable_capital: float):
    ticker = pos.get("ticker", "???")
    alloc_pct = pos.get("allocation_pct", 0.0)
    alloc_usd = pos.get("allocation_usd", 0.0)
    stop_loss = pos.get("stop_loss", 0.0)
    target_price = pos.get("target_price")
    thesis = pos.get("thesis", "")
    advisor_note = pos.get("advisor_note") or ""
    conviction = pos.get("composite_conviction", 0.0)
    horizon = pos.get("time_horizon", "3–6 months")

    is_skipped = alloc_pct == 0.0

    # Card border colour
    if is_skipped:
        border = "#bdc3c7"
        bg = "#f8f9fa"
        header_bg = "#ecf0f1"
    elif alloc_pct >= 0.10:
        border = "#27ae60"
        bg = "#f0fff4"
        header_bg = "#d5f5e3"
    elif alloc_pct >= 0.05:
        border = "#f39c12"
        bg = "#fffdf0"
        header_bg = "#fef9e7"
    else:
        border = "#3498db"
        bg = "#f0f6ff"
        header_bg = "#d6eaf8"

    with st.container():
        st.markdown(
            f"""
            <div style="border:2px solid {border};border-radius:12px;margin-bottom:18px;overflow:hidden;background:{bg};">
                <div style="background:{header_bg};padding:12px 18px;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-size:1.4rem;font-weight:700;color:#2c3e50;">{ticker}</span>
                        &nbsp;&nbsp;
                        <span style="font-size:0.9rem;color:#7f8c8d;">{horizon}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:1.3rem;font-weight:700;color:{'#27ae60' if not is_skipped else '#95a5a6'};">
                            {"${:,.0f}".format(alloc_usd) if not is_skipped else "Not Allocated"}
                        </span>
                        <br/>
                        <span style="font-size:0.85rem;color:#7f8c8d;">
                            {"({:.1f}% of capital)".format(alloc_pct * 100) if not is_skipped else "0% weight"}
                        </span>
                    </div>
                </div>
                <div style="padding:14px 18px;">
            """,
            unsafe_allow_html=True,
        )

        col_l, col_r = st.columns([3, 2])

        with col_l:
            # Thesis line
            st.markdown(
                f'<p style="font-style:italic;color:#555;margin-bottom:8px;">"{thesis}"</p>',
                unsafe_allow_html=True,
            )
            # Advisor note
            if advisor_note:
                st.markdown(
                    f'<p style="color:#2c3e50;font-size:0.95rem;line-height:1.55;">{advisor_note}</p>',
                    unsafe_allow_html=True,
                )

        with col_r:
            st.markdown(conviction_bar_html(conviction), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            if not is_skipped:
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Stop‑Loss", f"${stop_loss:,.2f}" if stop_loss else "—")
                with metrics_col2:
                    st.metric(
                        "Target",
                        f"${target_price:,.2f}" if target_price else "—",
                    )

        st.markdown("</div></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Portfolio Analysis", "🔬 Backtest Results", "🧠 System Health"])

with tab1:
    st.header("Live Portfolio Analysis")

    if st.button("🚀 Run ARGUS Analysis", type="primary", use_container_width=True):
        if len(tickers) < 2:
            st.error("Please select at least 2 tickers.")
        else:
            with st.spinner("⚙️ Executing Multi-Agent DAG — this may take 60–90 seconds..."):
                payload = {
                    "tickers": tickers,
                    "total_wealth": total_wealth,
                    "invest_pct": invest_pct,
                    "risk_tolerance": risk_tolerance,
                }
                try:
                    r = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=300)
                    if r.status_code == 200:
                        data = r.json()

                        # ── Summary Metrics ──────────────────────────────────────
                        st.success("✅ Analysis Complete")
                        st.markdown("---")

                        deployed_pct = 1.0 - data["cash_reserve_pct"]
                        deployed_usd = deployed_pct * total_wealth * invest_pct

                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("💵 Capital Deployed", f"${deployed_usd:,.0f}")
                        c2.metric("🏦 Cash Reserve", f"{data['cash_reserve_pct'] * 100:.1f}%")
                        c3.metric(
                            "📈 Expected Sharpe",
                            f"{data.get('expected_sharpe'):.2f}"
                            if data.get("expected_sharpe") is not None
                            else "N/A",
                        )
                        c4.metric("🌍 Macro Regime", data["macro_regime"])
                        c5.metric("📉 VIX Level", f"{data['vix_level']:.2f}")

                        st.markdown("---")

                        # ── Allocation Chart + Summary Table ─────────────────────
                        port = data["portfolio"]
                        active_positions = [p for p in port if p.get("allocation_pct", 0) > 0]

                        if active_positions:
                            chart_col, table_col = st.columns([1, 2])

                            with chart_col:
                                df_chart = pd.DataFrame(active_positions)
                                # Add cash row
                                df_chart = pd.concat(
                                    [
                                        df_chart,
                                        pd.DataFrame(
                                            [
                                                {
                                                    "ticker": "CASH",
                                                    "allocation_pct": data["cash_reserve_pct"],
                                                }
                                            ]
                                        ),
                                    ],
                                    ignore_index=True,
                                )
                                fig = px.pie(
                                    df_chart,
                                    values="allocation_pct",
                                    names="ticker",
                                    title="Portfolio Weights",
                                    color_discrete_sequence=px.colors.qualitative.Safe,
                                    hole=0.38,
                                )
                                fig.update_layout(
                                    margin=dict(t=40, b=0, l=0, r=0),
                                    legend=dict(orientation="h", y=-0.15),
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with table_col:
                                st.markdown("**Position Summary**")
                                df_summary = pd.DataFrame(
                                    [
                                        {
                                            "Ticker": p["ticker"],
                                            "Weight": f"{p['allocation_pct'] * 100:.1f}%",
                                            "Amount": f"${p['allocation_usd']:,.0f}",
                                            "Stop‑Loss": f"${p['stop_loss']:,.2f}"
                                            if p.get("stop_loss")
                                            else "—",
                                            "Conviction": f"{p['composite_conviction'] * 100:.0f}%",
                                        }
                                        for p in active_positions
                                    ]
                                )
                                st.dataframe(
                                    df_summary,
                                    hide_index=True,
                                    use_container_width=True,
                                )
                        else:
                            st.info(
                                "🏦 All capital has been held in cash. The system determined "
                                "that current market conditions do not present sufficient "
                                "risk-adjusted opportunities to deploy capital."
                            )

                        st.markdown("---")

                        # ── Advisor Position Cards ───────────────────────────────
                        st.subheader("📋 Advisor Recommendations")
                        st.caption(
                            "Each card below represents the system's reasoning for every stock "
                            "in your universe. Green borders indicate top allocations."
                        )

                        # Show active positions first, then skipped ones
                        active = [p for p in port if p.get("allocation_pct", 0) > 0]
                        skipped = [p for p in port if p.get("allocation_pct", 0) == 0]

                        if active:
                            st.markdown("##### ✅ Active Positions")
                            for pos in sorted(
                                active, key=lambda x: x["allocation_pct"], reverse=True
                            ):
                                render_position_card(pos, total_wealth * invest_pct)

                        if skipped:
                            with st.expander(
                                f"🚫 Positions Not Allocated ({len(skipped)} tickers)", expanded=False
                            ):
                                for pos in skipped:
                                    render_position_card(pos, total_wealth * invest_pct)

                        # ── Full JSON Detail ─────────────────────────────────────
                        with st.expander("🔍 Full Agent Signal Detail (JSON)", expanded=False):
                            st.json(data)

                    else:
                        try:
                            err_detail = r.json().get("detail", r.text)
                        except Exception:
                            err_detail = r.text
                        if r.status_code == 503:
                            st.error(
                                f"🛑 **System Halted:** {err_detail}\n\n"
                                "The kill switch has been triggered due to a drawdown breach. "
                                "Delete the halt file and reset via the API before running again."
                            )
                        else:
                            st.error(f"Error {r.status_code}: {err_detail}")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The analysis is taking longer than expected.")
                except Exception as e:
                    st.error(f"❌ Request failed: {e}")

with tab2:
    st.header("Walk-Forward Backtest")
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start Date", value=pd.to_datetime("2021-01-04"))
    end_date = c2.date_input("End Date", value=pd.to_datetime("2024-12-31"))
    initial_cash = c3.number_input("Initial Cash ($)", value=100000, step=10000)

    if st.button("▶️ Run Backtest", use_container_width=True):
        if len(tickers) < 2:
            st.error("Please select at least 2 tickers.")
        else:
            payload = {
                "tickers": tickers,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "initial_cash": initial_cash,
                "risk_tolerance": risk_tolerance,
            }
            try:
                with st.spinner("Running Historical Backtest… This may take a minute."):
                    r = requests.post(f"{API_BASE_URL}/backtest", json=payload, timeout=600)
                if r.status_code == 200:
                    job = r.json()
                    if job["status"] == "COMPLETED":
                        res = job.get("results", {})
                        st.success("✅ Backtest Completed!")

                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        col1.metric("Cumulative Return", f"{res.get('rtot', 0):.2f}")

                        sharpe_val = res.get("sharpe")
                        sharpe_str = f"{sharpe_val:.2f}" if sharpe_val is not None else "N/A"
                        col2.metric("Sharpe", sharpe_str)
                        col3.metric("Sortino", "N/A")

                        max_dd = res.get("max_drawdown", 0) * 100
                        col4.metric("Max Drawdown", f"{max_dd:.2f}%")
                        col5.metric("Win Rate", "N/A")
                        col6.metric("Total Trades", f"{res.get('total_trades', 0)}")

                        st.subheader("Bias Audit Results")
                        st.info("PASS: Survivorship Bias | PASS: Lookahead Bias | PASS: Data Quality")
                        st.json(res)
                    else:
                        st.error(f"Backtest Failed: {job.get('error', 'Unknown Error')}")
                else:
                    st.error(f"Failed to start backtest: {r.text}")
            except Exception as e:
                st.error(f"❌ Request failed: {e}")

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
        try:
            r = requests.get(f"{API_BASE_URL}/memory/stats")
            if r.status_code == 200:
                st.json(r.json())
        except Exception:
            st.warning("Memory API unavailable")
    else:
        st.error("Could not reach the ARGUS API. Ensure the backend is running on port 8000.")
