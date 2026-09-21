"""
dashboard/app.py — Streamlit Operator Monitoring Dashboard (Phase 12).

Read-only inspection interface for the AI Stock Trader system:
  1. Portfolio & Open Positions (cash reserve floor, mark-to-market valuations)
  2. Trade History & Orders (fees deducted, realized net PnL)
  3. Predictions & Market Regime (SPY 200-day MA filter, ML candidate ranking)
  4. Backtest vs SPY Benchmark (equity curves, drawdowns, capital-preservation metrics)
  5. Risk Engine & Audit Logs (validation skips, risk vetos, pipeline events)

Run locally:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so dashboard can import config and src
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from config.settings import settings
from dashboard.charts import (
    build_correlation_heatmap,
    build_daily_returns_histogram,
    build_feature_importance_bar_chart,
    build_feature_importance_history_chart,
    build_leaderboard_equity_chart,
    build_portfolio_vs_spy_chart,
    build_sector_allocation_donut,
    build_shap_summary_chart,
    build_win_loss_trade_chart,
)
from dashboard.live_ticker import (
    get_live_quotes_for_held_stocks,
    is_us_market_open,
)
from dashboard.data_loader import (
    get_closed_trade_history,
    get_daily_returns_histogram_data,
    get_equity_history_df,
    get_feature_importance_data,
    get_feature_importance_history_data,
    get_fundamental_screener_data,
    get_held_positions_correlation_data,
    get_leaderboard_equity_curves,
    get_leaderboard_summary_data,
    get_macro_environment_summary,
    get_market_regime_and_predictions,
    get_model_drift_summary,
    get_orders_df,
    get_portfolio_diversification_summary,
    get_portfolio_sectors_summary,
    get_portfolio_summary,
    get_portfolio_vs_spy_chart_data,
    get_recent_orders_df,
    get_recent_trades_df,
    get_sector_allocation_donut_data,
    get_sector_rotation_summary,
    get_system_events_df,
    get_win_loss_trades_chart_data,
    ask_chatbot_trigger,
    run_backtest_lab_trigger,
    run_backtest_trigger,
    run_daily_paper_cycle_trigger,
    run_fundamental_screen_trigger,
    run_tax_report_trigger,
)

# ── Page Configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Stock Trader | Operator Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark-themed terminal feel and animations
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2a2e39;
        margin-bottom: 12px;
    }
    .status-badge-green {
        color: #26a69a;
        font-weight: 600;
        background: rgba(38, 166, 154, 0.15);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid rgba(38, 166, 154, 0.3);
    }
    .status-badge-red {
        color: #ef5350;
        font-weight: 600;
        background: rgba(239, 83, 80, 0.15);
        padding: 4px 10px;
        border-radius: 4px;
        border: 1px solid rgba(239, 83, 80, 0.3);
    }

    /* Alert pulse keyframe animations */
    @keyframes pulseRed {
        0% { box-shadow: 0 0 0 0 rgba(239,83,80,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(239,83,80,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,83,80,0); }
    }

    @keyframes pulseOrange {
        0% { box-shadow: 0 0 0 0 rgba(255,152,0,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255,152,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,152,0,0); }
    }

    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(38,166,154,0.7); }
        70% { box-shadow: 0 0 0 10px rgba(38,166,154,0); }
        100% { box-shadow: 0 0 0 0 rgba(38,166,154,0); }
    }

    .pulse-red {
        animation: pulseRed 2s infinite;
    }

    .pulse-orange {
        animation: pulseOrange 2s infinite;
    }

    .pulse-green {
        animation: pulseGreen 3s infinite;
    }

    /* Tab switch smooth fade & slide animation */
    .stTabsContent {
        animation: fadeSlideIn 0.3s ease-out;
    }

    @keyframes fadeSlideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Make tab bar horizontally scrollable with thin red scrollbar */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        overflow-y: hidden !important;
        white-space: nowrap !important;
        scrollbar-width: thin;
        scrollbar-color: #ff4b4b #1e1e2e;
        padding-bottom: 4px;
        gap: 8px;
    }

    /* Scrollbar styling for Chrome/Webkit */
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        height: 4px;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
        background: #1e1e2e;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
        background-color: #ff4b4b;
        border-radius: 4px;
    }

    /* Keep tabs on single line */
    .stTabs [data-baseweb="tab"] {
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        padding: 10px 20px;
        border-radius: 6px 6px 0 0;
    }

    /* Accessibility: Respect user's motion preference */
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Initial Loading Animation (2 Seconds) ──────────────────────────────────────
if "dashboard_loaded" not in st.session_state:
    import time
    loading_placeholder = st.empty()
    steps = [
        "🔌 Connecting to database...",
        "📊 Loading portfolio data...",
        "🤖 Warming up AI model...",
        "📈 Fetching live prices...",
        "✅ Ready!",
    ]
    for i, step in enumerate(steps):
        pct = (i + 1) * 20
        loading_placeholder.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 260px; animation: fadeSlideIn 0.3s ease-out;">
            <h2 style="color: #ff4b4b; font-size: 26px; font-weight: 700; margin-bottom: 10px;">📈 AI Stock Trader</h2>
            <div style="font-size: 16px; color: #e0e0e0; margin: 10px 0; padding: 12px 24px; background: #1e222d; border-radius: 8px; border: 1px solid #2a2e39;">
                {step}
            </div>
            <div style="width: 220px; background: #2a2e39; height: 6px; border-radius: 3px; margin-top: 15px; overflow: hidden;">
                <div style="width: {pct}%; background: #ff4b4b; height: 100%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.4)
    loading_placeholder.empty()
    st.session_state["dashboard_loaded"] = True


# ── Sidebar: System Info & Trigger Controls ────────────────────────────────────

with st.sidebar:
    st.title("⚡ System Controls")
    st.caption("AI Stock Trader V1 — Paper Trading Engine")

    # Safety Notice
    st.info("🛡️ **V1 Security Guarantee**: Paper trading only. No real money or live broker connections exist.")

    st.markdown("---")
    st.subheader("System Configuration")
    st.markdown(f"**Locked Capital**: `${settings.initial_capital:.2f}`")
    st.markdown(f"**Max Positions**: `{settings.max_positions}`")
    st.markdown(f"**Cash Reserve Floor**: `{settings.cash_reserve * 100:.1f}%`")
    st.markdown(f"**Fee per Trade**: `{settings.simulated_cost_per_trade * 100:.2f}%`")
    st.markdown(f"**Stop-Loss**: `−{settings.stop_loss * 100:.1f}%`")
    st.markdown(f"**Take-Profit**: `+{settings.take_profit * 100:.1f}%`")
    st.markdown(f"**Buy Bar**: `{settings.buy_bar:.2f}` | **Signal Exit**: `{settings.signal_exit:.2f}`")

    st.markdown("---")
    st.subheader("Manual Triggers")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Run Paper Cycle", use_container_width=True, help="Executes run_daily_pipeline()"):
            with st.spinner("Executing daily paper trading pipeline..."):
                try:
                    res = run_daily_paper_cycle_trigger()
                    st.success(f"Cycle completed for {res['run_date']}! Fills: {res['fills_count']}, Orders: {res['orders_count']}")
                    st.session_state["latest_audit_md"] = res["audit_markdown"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")

    with col_btn2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    if st.button("📊 Run Backtest Replay", use_container_width=True, help="Runs historical backtest against SPY"):
        with st.spinner("Running 2021-2024 backtest vs SPY benchmark..."):
            try:
                bt_res = run_backtest_trigger()
                st.session_state["bt_results"] = bt_res
                st.success("Backtest simulation completed!")
            except Exception as exc:
                st.error(f"Backtest error: {exc}")

    st.markdown("---")
    st.caption("Active Model: `logistic_regression_baseline_v1`")
    st.caption(f"Database: `{settings.db_url}`")


# ── Main Dashboard Header ──────────────────────────────────────────────────────

st.title("📈 AI Stock Trader — Operator Monitoring Dashboard")
st.markdown("Automated algorithmic trading system running daily paper cycles with strict capital preservation rules.")

# Pull live data
port = get_portfolio_summary()
equity_df = get_equity_history_df()
trades_df = get_recent_trades_df(limit=50)
orders_df = get_recent_orders_df(limit=50)
regime_info, preds_df = get_market_regime_and_predictions()
events_df = get_system_events_df(limit=100)


# ── Top Metric Banner ──────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)
with m1:
    pnl_val = port.get('unrealized_pnl_total', 0.0)
    pulse_cls = "pulse-green" if pnl_val > 0 else ""
    if pulse_cls:
        st.markdown(f'<div class="{pulse_cls}" style="border-radius:8px; padding:2px;">', unsafe_allow_html=True)
    st.metric(
        "Total Portfolio Value",
        f"${port['total_equity']:.2f}",
        delta=f"${port['unrealized_pnl_total']:+.2f} Today's Change",
        help="This is all your money combined — both what's invested in stocks and what's sitting as cash.",
    )
    if pulse_cls:
        st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.metric(
        "Cash Balance",
        f"${port['cash']:.2f}",
        help="Money ready to be invested or kept safe as cash.",
    )
with m3:
    floor_diff = port['cash_reserve_pct'] - (settings.cash_reserve * 100.0)
    st.metric(
        "Cash Reserve",
        f"{port['cash_reserve_pct']:.1f}%",
        delta=f"{floor_diff:+.1f}% vs floor",
        delta_color="normal",
        help="Money kept safe and not invested. We always keep at least 15% in cash as an emergency cushion.",
    )
with m4:
    is_risk_on = regime_info.get("is_risk_on", True)
    regime_label = "✅ Good Time to Buy" if is_risk_on else "🛑 Market Crash Protection Active"
    pulse_cb = "pulse-red" if not is_risk_on else ""
    if pulse_cb:
        st.markdown(f'<div class="{pulse_cb}" style="border-radius:8px; padding:2px;">', unsafe_allow_html=True)
    st.metric(
        "Market Condition",
        regime_label,
        help="Tells whether market conditions are favorable for buying new stocks.",
    )
    if pulse_cb:
        st.markdown('</div>', unsafe_allow_html=True)


# ── Tabs Navigation ────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab_perf, tab_lead, tab_lab, tab_tax, tab_ai, tab5, tab6 = st.tabs([
    "💼 Portfolio & Positions",
    "📜 Trades & Orders",
    "🧠 Model & Regime",
    "📈 Backtest vs SPY",
    "📊 Performance",
    "🏆 Strategy Leaderboard",
    "🧪 Backtest Lab",
    "📄 Tax Report",
    "🤖 Ask AI",
    "🛡️ Risk & Audit Logs",
    "🏛️ Long-Term Screener",
])


# ── Tab 1: Portfolio & Open Positions ──────────────────────────────────────────

with tab1:
    st.subheader("Current Portfolio Allocation")
    
    col_alloc1, col_alloc2 = st.columns([1, 2])
    with col_alloc1:
        st.markdown(f"**As of**: `{port['run_date']}`")
        st.markdown(f"- **Uninvested Cash**: `${port['cash']:.4f}`")
        st.markdown(f"- **Invested Value**: `${port['invested_value']:.4f}`")
        st.markdown(f"- **Total Portfolio Value**: `${port['total_equity']:.4f}`")
        st.markdown(f"- **Total Slippage Incurred**: `${port.get('total_slippage_cost', 0.0):.4f}`")
        st.markdown(f"- **Open Positions**: `{port['open_positions_count']} / {port['max_positions']}` slots occupied")
        
        # Cash reserve visual indicator
        progress_val = min(1.0, max(0.0, port['cash_reserve_pct'] / 100.0))
        st.progress(progress_val, text=f"Cash Cushion: {port['cash_reserve_pct']:.1f}% (Required: ≥{settings.cash_reserve*100:.0f}%)")
        if port['cash_reserve_pct'] < (settings.cash_reserve * 100.0):
            st.warning("⚠️ Cash reserve is below the mandatory 15% safety buffer. New buys are vetoed.")
        else:
            st.success("✅ Cash buffer healthy. Capital preservation rule §1.4 satisfied.")

    with col_alloc2:
        # Equity Curve History
        st.markdown("**Simulated Equity Curve (Daily Snapshots)**")
        if not equity_df.empty and len(equity_df) > 1:
            chart_data = equity_df.set_index("date")[["total_value", "cash"]]
            st.line_chart(chart_data, color=["#2962ff", "#26a69a"])
        else:
            st.info("Single snapshot recorded. Run daily cycles to build the multi-day equity history curve.")

    # ── Section 6 Item 3: Live Prices Ticker (Currently Held Stocks Only) ─────
    st.markdown("---")
    is_open, market_status_msg = is_us_market_open()
    status_badge_html = (
        '<span class="status-badge-green">● LIVE</span>'
        if is_open
        else f'<span class="status-badge-red">● CLOSED</span> &nbsp; <small style="color:#888;">({market_status_msg})</small>'
    )
    st.markdown(f"### 📡 Live Prices &nbsp; {status_badge_html}", unsafe_allow_html=True)

    # Use st.empty() container pattern for smooth in-place price updates
    ticker_container = st.empty()
    with ticker_container.container():
        held_positions = port.get("positions", [])
        if held_positions:
            cached_quotes = st.session_state.get("live_quotes_cache", {})
            live_quotes = get_live_quotes_for_held_stocks(held_positions, session_cache=cached_quotes)

            # Store in session state to eliminate flicker
            st.session_state["live_quotes_cache"] = {q["ticker"]: q for q in live_quotes}
            from datetime import datetime
            from zoneinfo import ZoneInfo
            now_ts = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET")
            st.session_state["last_ticker_refresh_ts"] = now_ts

            if is_open:
                st.caption(
                    f"🕒 **Last Refreshed**: `{now_ts}` &nbsp;·&nbsp; "
                    f"⏱️ **Next Refresh in**: `60s` &nbsp;·&nbsp; "
                    f"🟢 Regular Market Session Active (9:30 AM – 4:00 PM ET Mon–Fri)"
                )
            else:
                st.caption(
                    f"🕒 **Last Refreshed**: `{now_ts}` &nbsp;·&nbsp; "
                    f"🔒 **Market Closed** (Regular hours: 9:30 AM – 4:00 PM ET Mon–Fri). Showing last closing price."
                )

            # Responsive metric cards for each currently held stock
            num_cols = min(4, max(1, len(live_quotes)))
            card_cols = st.columns(num_cols)
            for idx, q in enumerate(live_quotes):
                c_target = card_cols[idx % num_cols]
                with c_target:
                    chg_d = q["change_dollar"]
                    chg_p = q["change_pct"]
                    unrealized = q["unrealized_pnl"]
                    unrealized_pct = q["unrealized_pnl_pct"]
                    stale_badge = " ⚠️ STALE" if q.get("is_stale") else ""

                    delta_label = f"{chg_d:+.2f} ({chg_p:+.2f}%) vs prev close"

                    c_target.metric(
                        label=f"{q['ticker']}{stale_badge}",
                        value=f"${q['current_price']:.2f}",
                        delta=delta_label,
                        delta_color="normal" if q["is_up"] else "inverse",
                        help=(
                            f"Entry Fill Price: ${q['entry_price']:.2f}\n"
                            f"Shares Held: {q['shares']:.4f}\n"
                            f"Unrealized P&L: ${unrealized:+.2f} ({unrealized_pct:+.2f}%)\n"
                            f"Yesterday Close: ${q['prev_close']:.2f}"
                        ),
                    )
                    pnl_badge_class = "status-badge-green" if unrealized >= 0 else "status-badge-red"
                    st.markdown(
                        f"<small>Position P&L: <span class='{pnl_badge_class}'>${unrealized:+.2f} ({unrealized_pct:+.2f}%)</span></small>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("💡 **No active stock positions held.** Live price updates will activate automatically when positions are bought.")

    st.markdown("---")
    st.subheader(f"Open Positions ({port['open_positions_count']})")
    if port["positions"]:
        pos_df = pd.DataFrame(port["positions"])
        # Build ordered display columns (include Section 5 intelligence if available)
        base_rename = {
            "ticker": "Ticker",
            "shares": "Shares Held",
            "entry_price": "Entry Price ($)",
            "current_price": "Current Price ($)",
            "market_value": "Market Value ($)",
            "unrealized_pnl": "Unrealized PnL ($)",
            "unrealized_pnl_pct": "Return (%)",
        }
        disp_cols = list(base_rename.keys())
        if "sentiment" in pos_df.columns:
            base_rename["sentiment"] = "Sentiment"
            disp_cols.append("sentiment")
        if "earnings" in pos_df.columns:
            base_rename["earnings"] = "Earnings Alert"
            disp_cols.append("earnings")
        if "size_tier" in pos_df.columns:
            base_rename["size_tier"] = "Size Tier"
            disp_cols.append("size_tier")
        pos_df = pos_df[disp_cols].rename(columns=base_rename)
        fmt = {
            "Shares Held": "{:.4f}",
            "Entry Price ($)": "${:.2f}",
            "Current Price ($)": "${:.2f}",
            "Market Value ($)": "${:.2f}",
            "Unrealized PnL ($)": "${:+.4f}",
            "Return (%)": "{:+.2f}%",
        }
        st.dataframe(
            pos_df.style.format(fmt),
            use_container_width=True,
        )
    else:
        st.info("💡 **No active stock positions held.** Cash is a valid position (100% dry powder awaiting qualified conviction setups.)")

    # ── V2.2 Wave 1: Portfolio Intelligence (Sector Analysis & Diversification) ──
    st.markdown("---")
    st.subheader("🧩 Portfolio Intelligence & Sector Diversification")
    st.caption("Cross-sector concentration tracking and explainable HHI diversification scoring.")

    # Sector Breakdown & Diversification Score Cards
    col_sec1, col_sec2 = st.columns([1, 1])

    sec_summary = get_portfolio_sectors_summary()
    div_summary = get_portfolio_diversification_summary()

    with col_sec1:
        st.markdown("#### Sector Exposure & Concentration")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Active Sectors", len(sec_summary["holdings_sectors"]))
        with sc2:
            st.metric("Top Sector Exposure", f"{sec_summary['top_sector_pct']:.1f}%", help=f"Top sector: {sec_summary['top_sector'] or 'None'}")
        with sc3:
            st.metric("Cash Reserve", f"{sec_summary['cash_reserve_pct']:.1f}%")

        if sec_summary["concentration_warning"]:
            st.warning(f"⚠️ **Sector Concentration Alert**: Top sector '{sec_summary['top_sector']}' exceeds {sec_summary['concentration_limit_pct']}% of total portfolio equity.")

        if sec_summary["holdings_sectors"]:
            sec_df = pd.DataFrame(sec_summary["holdings_sectors"])[[
                "sector", "position_count", "market_value", "weight_portfolio_pct", "weight_invested_pct"
            ]].rename(columns={
                "sector": "Sector",
                "position_count": "Positions",
                "market_value": "Market Value ($)",
                "weight_portfolio_pct": "Portfolio Weight (%)",
                "weight_invested_pct": "Invested Weight (%)",
            })
            st.dataframe(
                sec_df.style.format({
                    "Market Value ($)": "${:.2f}",
                    "Portfolio Weight (%)": "{:.1f}%",
                    "Invested Weight (%)": "{:.1f}%",
                }),
                use_container_width=True,
            )
        else:
            st.caption("100% Cash Reserve — 0 equity exposure in any market sector.")

    with col_sec2:
        st.markdown("#### Rule-Based Diversification Score")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Diversification Score", f"{div_summary['score']:.1f} / 100", help="Explainable 0-100 score")
        with d2:
            st.metric("Sector HHI", f"{div_summary['sector_hhi']:.3f}", help="Herfindahl-Hirschman Index (0=Equal, 1.0=Concentrated)")
        with d3:
            st.metric("Effective Sectors", f"{div_summary['effective_sectors']:.1f}", help="N_eff = 1 / HHI_sector")

        st.markdown(f"**Rating**: `{div_summary['rating']}` — *{div_summary['rating_label']}*")
        st.caption(f"ℹ️ **Formula Breakdown**: {div_summary['formula_explanation']}")

        # 3-Pillar Breakdown
        pb = div_summary.get("pillar_breakdown", {})
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.caption(f"**Sector Breadth**: `{pb.get('sector_breadth_pts', 0):.1f} / 40 pts`")
        with pcol2:
            st.caption(f"**Sector HHI**: `{pb.get('sector_hhi_pts', 0):.1f} / 40 pts`")
        with pcol3:
            st.caption(f"**Position Balance**: `{pb.get('position_balance_pts', 0):.1f} / 20 pts`")



# ── Tab 2: Trades & Orders ────────────────────────────────────────────────────

with tab2:
    st.subheader("📜 Closed Trade History")
    st.caption("Full scrollable history of completed (bought & sold) pairs. Sorted by Date Sold (newest first).")

    closed_df, summary_metrics, export_df = get_closed_trade_history()

    if not closed_df.empty:
        # Summary Row / Metrics cards at top of table
        sm1, sm2, sm3, sm4, sm5, sm6, sm7 = st.columns(7)
        with sm1:
            st.metric("Total Trades", f"{summary_metrics['total_trades']}")
        with sm2:
            st.metric("Total P&L", f"${summary_metrics['total_pnl']:+,.2f}")
        with sm3:
            st.metric("Win Rate", f"{summary_metrics['win_rate']:.1f}%")
        with sm4:
            st.metric("Avg Gain (Wins)", f"${summary_metrics['avg_gain']:+,.2f}")
        with sm5:
            st.metric("Avg Loss (Losses)", f"${summary_metrics['avg_loss']:+,.2f}")
        with sm6:
            st.metric("Best Trade", f"${summary_metrics['best_trade']:+,.2f}")
        with sm7:
            st.metric("Worst Trade", f"${summary_metrics['worst_trade']:+,.2f}")

        # Format display rows with summary row at bottom
        display_rows = []
        for _, r in closed_df.iterrows():
            score_val = r.get("Model Score at Entry")
            score_str = f"{float(score_val):.2f}" if score_val is not None and str(score_val) != "nan" and score_val != "" else "—"
            display_rows.append({
                "Date Bought": str(r["Date Bought"]),
                "Date Sold": str(r["Date Sold"]),
                "Ticker": str(r["Ticker"]),
                "Entry Price": f"${float(r['Entry Price']):.2f}",
                "Exit Price": f"${float(r['Exit Price']):.2f}",
                "Shares": f"{float(r['Shares']):.4f}",
                "Profit / Loss ($)": f"${float(r['Profit / Loss ($)']):+.2f}",
                "Profit / Loss (%)": f"{float(r['Profit / Loss (%)']):+.2f}%",
                "Exit Reason": str(r["Exit Reason"]),
                "Model Score at Entry": score_str,
            })

        # Summary row at bottom of table
        summary_row_display = {
            "Date Bought": "SUMMARY",
            "Date Sold": f"{summary_metrics['total_trades']} trades",
            "Ticker": f"Win Rate: {summary_metrics['win_rate']:.1f}%",
            "Entry Price": f"Avg Win: ${summary_metrics['avg_gain']:+.2f}",
            "Exit Price": f"Avg Loss: ${summary_metrics['avg_loss']:+.2f}",
            "Shares": "",
            "Profit / Loss ($)": f"${summary_metrics['total_pnl']:+.2f}",
            "Profit / Loss (%)": "",
            "Exit Reason": f"Best: ${summary_metrics['best_trade']:+.2f} / Worst: ${summary_metrics['worst_trade']:+.2f}",
            "Model Score at Entry": "",
        }
        display_rows.append(summary_row_display)
        table_display_df = pd.DataFrame(display_rows)

        def style_trade_history_row(row):
            if row.get("Date Bought") == "SUMMARY":
                return ["font-weight: bold; background-color: #2a2e39; color: #ffffff"] * len(row)
            val = str(row.get("Profit / Loss ($)", ""))
            try:
                num = float(val.replace("$", "").replace(",", "").replace("+", ""))
            except Exception:
                num = 0.0
            if num > 0:
                return ["background-color: rgba(38, 166, 154, 0.20); color: #26a69a"] * len(row)
            elif num < 0:
                return ["background-color: rgba(239, 83, 80, 0.20); color: #ef5350"] * len(row)
            return [""] * len(row)

        styled_trade_table = table_display_df.style.apply(style_trade_history_row, axis=1)
        st.dataframe(
            styled_trade_table,
            use_container_width=True,
            height=360,
            hide_index=True,
        )

        # Download CSV Button
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Trade History as CSV",
            data=csv_bytes,
            file_name="trade_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No closed trades recorded yet. Completed trades (bought & sold pairs) will appear here.")

    # ── Open Positions Shown Separately Below the Table ───────────────────────
    st.markdown("---")
    st.subheader("💼 Active Open Positions")
    st.caption("Positions currently held in portfolio (bought and awaiting exit trigger).")
    if port["positions"]:
        open_pos_df = pd.DataFrame(port["positions"])
        base_open_cols = ["ticker", "shares", "entry_price", "current_price", "market_value", "unrealized_pnl", "unrealized_pnl_pct", "size_tier"]
        avail_cols = [c for c in base_open_cols if c in open_pos_df.columns]
        open_rename = {
            "ticker": "Ticker",
            "shares": "Shares Held",
            "entry_price": "Entry Price ($)",
            "current_price": "Current Price ($)",
            "market_value": "Market Value ($)",
            "unrealized_pnl": "Unrealized PnL ($)",
            "unrealized_pnl_pct": "Return (%)",
            "size_tier": "Size Tier",
        }
        st.dataframe(
            open_pos_df[avail_cols].rename(columns=open_rename).style.format({
                "Shares Held": "{:.4f}",
                "Entry Price ($)": "${:.2f}",
                "Current Price ($)": "${:.2f}",
                "Market Value ($)": "${:.2f}",
                "Unrealized PnL ($)": "${:+.4f}",
                "Return (%)": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No open positions. 100% held in cash reserve.")

    st.markdown("---")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Recent Executed Fills")
        if not trades_df.empty:
            st.dataframe(
                trades_df.style.format({
                    "shares": "{:.4f}",
                    "price": "${:.2f}",
                    "fee": "${:.4f}",
                    "net_pnl": "${:+.4f}",
                }),
                use_container_width=True,
                height=320,
            )
        else:
            st.info("No trades executed yet. Fills appear here when the paper broker executes orders.")

    with col_t2:
        st.subheader("Generated Order Decisions")
        if not orders_df.empty:
            st.dataframe(
                orders_df.style.format({
                    "shares": "{:.4f}",
                    "price": "${:.2f}",
                }),
                use_container_width=True,
                height=320,
            )
        else:
            st.info("No orders generated yet.")

    # Fee Accounting Note
    st.markdown("""
    > [!NOTE]
    > **Locked Fee Accounting Convention (§1.6)**:
    > - **BUY**: Cash decrease = `shares × fill_price × (1 + 0.002)` (all-inclusive outflow).
    > - **SELL**: Cash increase = `shares × fill_price × (1 − 0.002)` (net of fee).
    > - **Realized Net PnL**: `Net Proceeds − (Gross Cost + Entry Fee)`.
    """)


# ── Tab 3: Model & Regime ──────────────────────────────────────────────────────

with tab3:
    st.subheader("Active Model & Market Regime Filter")

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.markdown(f"**Model Type**: `{regime_info.get('active_model', 'logistic_regression_baseline_v1')}`")
        st.markdown(f"**Promoted Variant**: Baseline (Logistic Regression)")
        st.markdown(f"**Benchmark Ticker**: `{settings.benchmark}`")
        if regime_info["spy_price"] is not None:
            st.markdown(f"- **Current SPY Close**: `${regime_info['spy_price']:.2f}`")
            st.markdown(f"- **200-Day SMA**: `${regime_info['spy_200ma']:.2f}`")
        st.markdown(f"**Regime Status**: `{regime_info['status']}`")
        st.caption(regime_info["explanation"])

    with col_r2:
        st.markdown(r"""
        **Regime Filter Logic (§1.4)**:
        - When `SPY close >= 200-day Simple Moving Average`: Market is **RISK-ON**. Top-ranked candidates meeting the $P \ge 0.60$ conviction bar may be purchased.
        - When `SPY close < 200-day Simple Moving Average`: Market is **RISK-OFF**. All BUY orders are strictly vetoed by the Risk Engine. Existing positions are evaluated normally for stop-loss or take-profit.
        """)

    st.markdown("---")
    st.subheader("Universe Candidate Ranking & Inference")
    if not preds_df.empty:
        st.dataframe(
            preds_df.style.format({
                "probability": "{:.4f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("No model predictions available for the universe tickers.")
    # ── Section 5: Macro Environment Card ──────────────────────────────────────
    st.markdown("---")
    st.subheader("🌐 Macro Economic Environment")
    st.caption("Fed Funds, CPI YoY, Unemployment, and 10Y Treasury from FRED API. Weekly fetch cadence (Monday refresh).")

    macro_env = get_macro_environment_summary()
    macro_regime = macro_env.get("macro_regime", "NEUTRAL")
    _regime_color = (
        "status-badge-green" if macro_regime == "FAVORABLE"
        else "status-badge-red" if macro_regime == "RESTRICTIVE"
        else "status-badge-yellow"
    )
    _regime_effect = (
        "No position size adjustment." if macro_regime == "FAVORABLE"
        else "Position sizes reduced by 20%." if macro_regime == "NEUTRAL"
        else "Position sizes reduced by 40%. Buy bar raised by +3%."
    )
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(f"**Regime**: <span class='{_regime_color}'>{macro_regime}</span>", unsafe_allow_html=True)
        st.caption(_regime_effect)
    with mcol2:
        _ff = macro_env.get("fed_funds_rate")
        st.metric("Fed Funds Rate", f"{_ff:.2f}%" if _ff is not None else "N/A", help="Federal Funds Rate (FEDFUNDS)")
    with mcol3:
        _cpi = macro_env.get("cpi_yoy")
        st.metric("CPI YoY", f"{_cpi:.2f}%" if _cpi is not None else "N/A", help="CPI Year-over-Year inflation (CPIAUCSL)")
    with mcol4:
        _unemp = macro_env.get("unemployment_rate")
        st.metric("Unemployment", f"{_unemp:.1f}%" if _unemp is not None else "N/A", help="U.S. Unemployment Rate (UNRATE)")
    with mcol5:
        _t10 = macro_env.get("treasury_10y")
        st.metric("10Y Treasury", f"{_t10:.2f}%" if _t10 is not None else "N/A", help="10-Year Treasury Yield (DGS10)")
    _src_lbl = macro_env.get("fetched_date", "N/A")
    _is_cached = macro_env.get("is_cached", True)
    st.caption(f"{'🗃️ Cached' if _is_cached else '🔴 Live'} — Data as of `{_src_lbl}`. Add FRED_API_KEY to `.env` for weekly live refresh.")

    # ── Section 5: Sector Rotation Panel ───────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Sector Rotation Strength (20-Day Returns)")
    st.caption("Ranks all 8 sectors 1–8 by average 20-day return. Rank 1–2 → +4% score boost. Rank 6–8 → -4% score penalty.")
    _sector_rows = get_sector_rotation_summary()
    if _sector_rows:
        _sec_df = pd.DataFrame(_sector_rows)
        _sec_df["signal"] = _sec_df["rank"].apply(
            lambda r: "🟢 BOOSTED" if r <= 2 else ("🔴 PENALIZED" if r >= 6 else "⚪ NEUTRAL")
        )
        _sec_df["20d Return (%)"] = (_sec_df["avg_20d_return"] * 100).round(2)
        _display_sec = _sec_df[["rank", "sector", "20d Return (%)", "rotation_multiplier", "signal"]].rename(columns={
            "rank": "Rank", "sector": "Sector", "rotation_multiplier": "Score Modifier", "signal": "Status",
        })
        st.dataframe(
            _display_sec.style.format({"20d Return (%)": "{:+.2f}%", "Score Modifier": "{:+.4f}"}),
            use_container_width=True, height=320,
        )
        st.caption(f"Rankings last computed: `{_sector_rows[0].get('date', 'N/A')}`")
    else:
        st.info("No sector rotation data available yet. Run the pipeline to compute sector momentum rankings.")

    # ── V2.1 Wave 1: Model Drift Monitor ───────────────────────────────────────
    st.markdown("---")
    st.subheader("🛡️ Model Health & Stability Monitor")
    st.caption("Tracks live prediction distribution against historical validation tests. Informational only — zero automatic retraining.")

    drift_info = get_model_drift_summary()
    dc1, dc2, dc3, dc4 = st.columns(4)

    with dc1:
        status_val = drift_info.get("status", "STABLE")
        disp_status = "Good ✅" if status_val == "STABLE" else ("⚠️ AI Model Needs Updating" if status_val == "DRIFT_ALERT" else "🟡 Monitoring")
        status_color = "status-badge-green" if status_val == "STABLE" else ("status-badge-red pulse-orange" if status_val == "DRIFT_ALERT" else "status-badge-yellow")
        st.markdown(f"**Model Health**: <span class='{status_color}'>{disp_status}</span>", unsafe_allow_html=True)
    with dc2:
        pulse_drift = "pulse-orange" if status_val == "DRIFT_ALERT" else ""
        if pulse_drift:
            st.markdown(f'<div class="{pulse_drift}" style="border-radius:8px; padding:2px;">', unsafe_allow_html=True)
        st.metric("Model Health", f"{drift_info.get('psi', 0.0):.4f}", help="Measures if the AI predictions stay reliable and healthy over time (<0.10 Good, >=0.25 Model Needs Updating)")
        if pulse_drift:
            st.markdown('</div>', unsafe_allow_html=True)
    with dc3:
        st.metric("KS Test p-value", f"{drift_info.get('ks_pvalue', 1.0):.4f}", help="Statistical test comparing live predictions against historical testing distribution.")
    with dc4:
        st.metric("Live vs Base Mean", f"{drift_info.get('live_mean', 0.5):.3f} / {drift_info.get('baseline_mean', 0.5):.3f}", help="Average AI prediction probability today vs baseline training average.")

    st.info(f"📋 **System Diagnosis**: {drift_info.get('recommendation', 'Live predictions consistent with baseline.')}")
    st.caption(f"🔒 **Honest Disclaimer**: {drift_info.get('disclaimer', 'Informational drift monitoring only. No automatic retraining or model promotion.')}")

    # ── Section 9 Item 2: Feature Importance Dashboard ─────────────────────────
    st.markdown("---")
    st.subheader("📊 What Drives Decisions Dashboard")
    st.caption("Visual breakdown of which market indicators matter most to the AI model right now.")

    col_fi1, _ = st.columns([1, 3])
    with col_fi1:
        sel_model = st.selectbox(
            "Select Model Type",
            options=["primary", "baseline", "xgboost", "ensemble"],
            format_func=lambda x: {
                "primary": "Smart AI Model (HistGradientBoosting)",
                "baseline": "Simple AI Model (Logistic Regression)",
                "xgboost": "Advanced AI Model (XGBoost)",
                "ensemble": "Combined AI Model (VotingClassifier)",
            }.get(x, x),
            key="fi_model_selector",
        )

    fi_rows = get_feature_importance_data(model_type=sel_model)
    if fi_rows:
        # Top 5 Features summary cards at top ("Most Influential Indicators")
        top_5 = fi_rows[:5]
        st.markdown("#### 🌟 Most Influential Indicators (Top 5)")
        fcols = st.columns(min(5, len(top_5)))
        for idx, item in enumerate(top_5):
            with fcols[idx]:
                st.metric(
                    label=item["feature_name"],
                    value=f"{item['importance_score'] * 100:.1f}%",
                    help=f"Importance Score: {item['importance_score']:.4f} (What Drives Decisions)",
                )

        # Bar chart
        fi_chart = build_feature_importance_bar_chart(fi_rows)
        if fi_chart is not None:
            st.altair_chart(fi_chart, use_container_width=True)

        # Retrain history trend lines
        history_rows = get_feature_importance_history_data(model_type=sel_model, top_n=5, last_n=10)
        if history_rows:
            st.markdown("#### 📈 Historical Feature Importance Trend (Last 10 Retrains)")
            hist_chart = build_feature_importance_history_chart(history_rows)
            if hist_chart is not None:
                st.altair_chart(hist_chart, use_container_width=True)
    else:
        st.info(f"No feature importance data recorded for `{sel_model}` model yet. Retrain the model to generate scores.")

    # SHAP Global Analysis section
    with st.expander("🔍 Why AI Made This Choice (SHAP Values Analysis)", expanded=False):
        st.caption("Shows feature impact direction: positive (pushes toward BUY) vs negative (pushes toward SELL).")
        try:
            import shap
            if st.button("Compute SHAP Values Now", key="btn_compute_shap"):
                with st.spinner("Computing SHAP values..."):
                    from src.ml.evaluate import load_active_model
                    from src.ml.importance import compute_global_shap
                    from src.data.dataset import load_dataset
                    active_mod = load_active_model()
                    X_df, _, _, _ = load_dataset()
                    shap_res = compute_global_shap(active_mod, X_df, sample_size=100)
                    shap_chart = build_shap_summary_chart(shap_res)
                    if shap_chart:
                        st.altair_chart(shap_chart, use_container_width=True)
                    else:
                        st.dataframe(pd.DataFrame(shap_res))
        except ImportError:
            st.warning("`shap` library is not installed. Run `pip install shap` to enable SHAP directional analysis.")


# ── Tab 4: Backtest vs SPY Benchmark ──────────────────────────────────────────

with tab4:
    st.subheader("Historical Backtest vs SPY Benchmark (2021–2024)")
    st.caption("Realistic simulation with 0.2% per-trade transaction costs and Next-Open morning fill lag rule.")

    # Check if user clicked Run Backtest button
    bt_data = st.session_state.get("bt_results")
    if bt_data:
        # Display Metrics Grid
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.metric("Strategy Return", f"{bt_data['cagr']:+.2f}% CAGR", delta=f"{bt_data['cagr'] - bt_data['benchmark_cagr']:+.2f}% vs SPY", help="Compounded annual return generated by the AI strategy.")
        with b2:
            st.metric("Market Comparison (SPY)", f"{bt_data['benchmark_cagr']:+.2f}% CAGR", help="Compounded annual return of buying and holding the SPY index.")
        with b3:
            st.metric("Biggest Loss Period", f"{bt_data['max_drawdown']:.2f}%", delta=f"{abs(bt_data['benchmark_max_drawdown']) - abs(bt_data['max_drawdown']):+.2f}% better", delta_color="normal", help="The worst losing streak your portfolio had. -8% means at its worst point you were down $800 from $10,000.")
        with b4:
            st.metric("SPY Biggest Loss Period", f"{bt_data['benchmark_max_drawdown']:.2f}%", help="The worst losing streak of the S&P 500 benchmark during the same period.")
        with b5:
            st.metric("Completed Trades", f"{bt_data['total_trades']}", help=f"Win Rate: {bt_data['win_rate']:.1f}% | Profit Factor: {bt_data['profit_factor']:.2f}")





# ── Tab 4: Backtest vs SPY Benchmark ──────────────────────────────────────────

with tab4:
    st.subheader("Historical Backtest vs SPY Benchmark (2021–2024)")
    st.caption("Realistic simulation with 0.2% per-trade transaction costs and Next-Open morning fill lag rule.")

    # Check if user clicked Run Backtest button
    bt_data = st.session_state.get("bt_results")
    if bt_data:
        # Display Metrics Grid
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            st.metric("Strategy Return", f"{bt_data['cagr']:+.2f}% CAGR", delta=f"{bt_data['cagr'] - bt_data['benchmark_cagr']:+.2f}% vs SPY")
        with b2:
            st.metric("SPY Benchmark", f"{bt_data['benchmark_cagr']:+.2f}% CAGR")
        with b3:
            st.metric("Strategy Max DD", f"{bt_data['max_drawdown']:.2f}%", delta=f"{abs(bt_data['benchmark_max_drawdown']) - abs(bt_data['max_drawdown']):+.2f}% better", delta_color="normal")
        with b4:
            st.metric("SPY Max DD", f"{bt_data['benchmark_max_drawdown']:.2f}%")
        with b5:
            st.metric("Completed Trades", f"{bt_data['total_trades']}", help=f"Win Rate: {bt_data['win_rate']:.1f}% | Profit Factor: {bt_data['profit_factor']:.2f}")

        # Chart
        st.markdown(f"### Comparative Equity Curve (${settings.initial_capital:,.2f} Starting Capital)")
        if "equity_curve" in bt_data and not bt_data["equity_curve"].empty:
            eq_curve = bt_data["equity_curve"].copy()
            if "date" in eq_curve.columns:
                eq_curve = eq_curve.set_index("date")
            cols_to_plot = [c for c in ["strategy_equity", "benchmark_equity", "cash"] if c in eq_curve.columns]
            st.line_chart(eq_curve[cols_to_plot])
    else:
        # Show pre-computed Phase 10 validation stats
        st.info("Click **[📊 Run Backtest Replay]** in the sidebar to run the live backtest engine. Below are the verified Phase 10 backtest results:")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("#### Baseline Model (Promoted)")
            st.markdown("- **CAGR**: `+3.02%` (Total Net: `+9.81%`)")
            st.markdown("- **Max Drawdown**: `-16.60%` (Protected vs SPY)")
            st.markdown("- **Win Rate**: `53.8%` (14 wins / 12 losses)")
            st.markdown("- **Profit Factor**: `1.26`")
            st.markdown("- **Completed Trades**: `26 trades`")
        with col_p2:
            st.markdown("#### SPY Benchmark (Buy & Hold)")
            st.markdown("- **CAGR**: `+8.55%` (Total Net: `+29.43%`)")
            st.markdown("- **Max Drawdown**: `-25.36%` (2022 Bear Market)")
            st.markdown("- **Sharpe Ratio**: `0.58`")
        with col_p3:
            st.markdown("#### Core Values Rationale")
            st.markdown("""
            **Safety Over Raw Return**:
            Baseline is retained because it cuts drawdowns substantially during market downturns (e.g. 2022 bear market) and faithfully honors the §1.1 capital preservation mandate.
            """)


# ── Performance Tab: Visual Performance Charts ──────────────────────────────

with tab_perf:
    st.subheader("📊 Visual Performance Charts")
    st.caption("Institutional-grade visual tracking of portfolio returns, benchmark comparison, trade outcomes, and sector exposure.")

    # 1. Portfolio vs SPY Line Chart
    st.markdown("### 1. Portfolio vs SPY Benchmark")
    st.caption("Historical equity trajectory compared to SPY buy-and-hold benchmark. Periods of underperformance highlighted in red.")
    spy_comp_df = get_portfolio_vs_spy_chart_data()
    spy_chart = build_portfolio_vs_spy_chart(spy_comp_df)
    if spy_chart is not None:
        st.altair_chart(spy_chart, use_container_width=True)
    else:
        st.info("No portfolio snapshot data available yet to render benchmark comparison.")

    st.markdown("---")

    col_p1, col_p2 = st.columns(2)

    # 2. Daily Returns Histogram
    with col_p1:
        st.markdown("### 2. Daily Returns Distribution")
        st.caption("Frequency distribution of daily percentage returns. Green = positive days, Red = negative days.")
        returns_hist_df = get_daily_returns_histogram_data()
        hist_chart = build_daily_returns_histogram(returns_hist_df)
        if hist_chart is not None:
            st.altair_chart(hist_chart, use_container_width=True)
        else:
            st.info("Requires at least 2 daily snapshots to compute return frequency distribution.")

    # 3. Win/Loss Trade Bar Chart
    with col_p2:
        st.markdown("### 3. Closed Trades Win/Loss Breakdown")
        st.caption("Realized profit/loss per closed trade in chronological order. Green = profit, Red = loss.")
        win_loss_df = get_win_loss_trades_chart_data()
        trade_chart = build_win_loss_trade_chart(win_loss_df)
        if trade_chart is not None:
            st.altair_chart(trade_chart, use_container_width=True)
        else:
            st.info("No closed trades recorded yet. Realized trade profits/losses will appear here once positions are sold.")

    st.markdown("---")

    # 4. Sector Allocation Donut Chart
    st.markdown("### 4. Sector & Cash Allocation")
    st.caption("Current breakdown of invested capital across economic sectors alongside cash reserve cushion.")
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        donut_df = get_sector_allocation_donut_data()
        donut_chart = build_sector_allocation_donut(donut_df)
        if donut_chart is not None:
            st.altair_chart(donut_chart, use_container_width=True)
        else:
            st.info("No portfolio holdings data available.")
    with col_d2:
        st.markdown("#### Allocation Summary")
        if not donut_df.empty:
            st.dataframe(
                donut_df.rename(columns={"sector": "Component", "value": "Value ($)", "pct": "Weight (%)"}).style.format({
                    "Value ($)": "${:,.2f}",
                    "Weight (%)": "{:.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # 5. Position Correlation Heatmap
    st.markdown("### 5. Position Correlation Heatmap")
    st.caption("Pairwise correlation between currently held positions based on 60-day rolling returns (Green = low correlation, Red = high correlation).")
    corr_matrix_df = get_held_positions_correlation_data()
    corr_chart = build_correlation_heatmap(corr_matrix_df)
    if corr_chart is not None:
        st.altair_chart(corr_chart, use_container_width=True)
    else:
        st.info("Requires at least 2 held positions with price history to calculate pairwise return correlation matrix.")


# ── Tab: Paper Trading Leaderboard (§10 Item 1) ────────────────────────────────

with tab_lead:
    st.subheader("🏆 Parallel Strategy Leaderboard (§10 Item 1)")
    st.caption("Tracks multiple strategy variants in parallel on paper money against live market data.")

    rankings, winning = get_leaderboard_summary_data()

    if winning:
        st.markdown(f"### 🏆 Current Leader: **{winning['name']}** ({winning['return_pct']:+.2f}%)")

    if rankings:
        # Leaderboard table
        lead_df = pd.DataFrame(rankings)
        lead_df["Rank"] = lead_df.apply(lambda r: f"{r['trophy']} #{r['rank']}" if r['is_winning'] else f"#{r['rank']}", axis=1)
        lead_df["Strategy"] = lead_df["name"]
        lead_df["Start"] = lead_df["starting_capital"].apply(lambda v: f"${v:,.2f}")
        lead_df["Value"] = lead_df["portfolio_value"].apply(lambda v: f"${v:,.2f}")
        lead_df["Return"] = lead_df["return_pct"].apply(lambda v: f"{v:+.2f}%")
        lead_df["Win Rate"] = lead_df["win_rate_pct"].apply(lambda v: f"{v:.1f}%")
        lead_df["Trades"] = lead_df["trades_count"]

        display_cols = ["Rank", "Strategy", "Start", "Value", "Return", "Win Rate", "Trades"]
        st.dataframe(lead_df[display_cols], use_container_width=True)

        # Multi-line equity chart
        st.markdown("---")
        st.markdown("#### 📈 Strategy Variants Portfolio Value Over Time")
        curves_df = get_leaderboard_equity_curves()
        if not curves_df.empty:
            l_chart = build_leaderboard_equity_chart(curves_df)
            if l_chart:
                st.altair_chart(l_chart, use_container_width=True)
            else:
                st.line_chart(curves_df.pivot(index="date", columns="Strategy", values="portfolio_value"))
        else:
            st.info("No daily snapshot history recorded yet for strategy variants.")
    else:
        st.info("No strategy variants recorded yet. Run the daily trading pipeline to start tracking.")


# ── Tab: Backtest Lab ──────────────────────────────────────────────────────────

with tab_lab:
    st.subheader("🧪 Interactive Strategy Backtest Lab")
    st.markdown(
        "Test custom risk rules, thresholds, and intelligence toggles on historical market data "
        "and instantly evaluate performance against the SPY benchmark."
    )

    col_controls, col_results = st.columns([1, 2])

    with col_controls:
        st.markdown("### ⚙️ Strategy Controls")

        # 1. Date Range
        st.markdown("#### 1. Date Range")
        import datetime
        min_date = datetime.date(2008, 1, 1)
        default_start = datetime.date(2023, 1, 1)
        default_end = datetime.date.today()

        lab_start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=default_end)
        lab_end_date = st.date_input("End Date", value=default_end, min_value=min_date, max_value=default_end)

        start_str = lab_start_date.strftime("%Y-%m-%d")
        end_str = lab_end_date.strftime("%Y-%m-%d")

        # Warning for date range < 6 months (180 days)
        date_delta = (lab_end_date - lab_start_date).days
        if date_delta < 180:
            st.warning("⚠️ Warning: Selected date range has less than 6 months of data. Results may not be statistically significant.")

        # 2. Strategy Settings
        st.markdown("#### 2. Strategy Settings")
        lab_buy_bar = st.slider("Buy threshold", min_value=0.55, max_value=0.75, value=0.60, step=0.01)
        lab_exit_bar = st.slider("Exit threshold", min_value=0.35, max_value=0.55, value=0.45, step=0.01)
        lab_stop_loss = st.slider("Stop loss %", min_value=3, max_value=15, value=8, step=1)
        lab_take_profit = st.slider("Take profit %", min_value=5, max_value=30, value=15, step=1)
        lab_max_positions = st.slider("Max positions", min_value=1, max_value=5, value=3, step=1)
        lab_position_sizing = st.selectbox("Position sizing", options=["Fixed", "Confidence-Based"], index=0)

        # 3. Intelligence Toggles
        st.markdown("#### 3. Intelligence Toggles")
        lab_use_sentiment = st.toggle("Use sentiment analysis", value=True)
        lab_use_earnings = st.toggle("Use earnings blackout", value=True)
        lab_use_sector = st.toggle("Use sector rotation", value=True)
        lab_use_macro = st.toggle("Use macro regime", value=True)
        lab_use_correlation = st.toggle("Use correlation filter", value=True)
        lab_use_trailing = st.toggle("Use trailing stop", value=True)

        st.markdown("---")
        run_lab_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

        if run_lab_btn:
            with st.spinner("Running backtest simulation..."):
                res = run_backtest_lab_trigger(
                    start_date=start_str,
                    end_date=end_str,
                    buy_threshold=lab_buy_bar,
                    exit_threshold=lab_exit_bar,
                    stop_loss_pct=lab_stop_loss / 100.0,
                    take_profit_pct=lab_take_profit / 100.0,
                    max_positions=lab_max_positions,
                    position_sizing=lab_position_sizing,
                    use_sentiment=lab_use_sentiment,
                    use_earnings_blackout=lab_use_earnings,
                    use_sector_rotation=lab_use_sector,
                    use_macro_regime=lab_use_macro,
                    use_correlation_filter=lab_use_correlation,
                    use_trailing_stop=lab_use_trailing,
                )
                st.session_state["backtest_lab_result"] = res

    with col_results:
        st.markdown("### 📊 Backtest Results")
        lab_res = st.session_state.get("backtest_lab_result")

        if lab_res is not None:
            # 1. Summary Metrics
            st.markdown("#### 1. Summary Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                ret_val = lab_res.get("total_return_pct", 0.0)
                st.metric("Total Return", f"{ret_val:+.2f}%", help="Compounded total gain or loss from all completed trades.")
                st.metric("Total Trades", lab_res.get("total_trades", 0), help="Number of simulated trades executed during backtest.")
            with m_col2:
                alpha_val = lab_res.get("alpha_pct", 0.0)
                st.metric("Extra Return vs Market", f"{alpha_val:+.2f}%", help="Additional return earned compared to buying and holding the S&P 500.")
                st.metric("Days in Cash", f"{lab_res.get('days_in_cash_pct', 0.0):.1f}%", help="Percentage of trading days portfolio was held safely in 100% cash.")
            with m_col3:
                dd_val = lab_res.get("max_drawdown_pct", 0.0)
                st.metric("Biggest Loss Period", f"{dd_val:.2f}%", help="The worst losing streak your portfolio had. -8% means at its worst point you were down $800 from $10,000.")
                st.metric("Risk-Adjusted Return", f"{lab_res.get('sharpe_ratio', 0.0):.2f}", help="How much return your portfolio generated relative to the amount of risk taken.")
            with m_col4:
                wr_val = lab_res.get("win_rate_pct", 0.0)
                st.metric("Win Rate", f"{wr_val:.1f}%", help="Percentage of closed trades that resulted in a net profit.")
                st.metric("Best / Worst", f"{lab_res.get('best_month', '—')} / {lab_res.get('worst_month', '—')}", help="Best and worst performing calendar months.")

            # 2. Equity Curve Chart
            st.markdown("---")
            st.markdown("#### 2. Equity Curve vs SPY Benchmark")
            eq_df = lab_res.get("equity_curve", pd.DataFrame())
            if not eq_df.empty:
                chart_data = eq_df.set_index("date")[["Strategy", "SPY Benchmark"]]
                st.line_chart(chart_data, color=["#2962ff", "#ff9800"])
            else:
                st.info("No equity curve data generated.")

            # 3. Monthly Returns Heatmap
            st.markdown("---")
            st.markdown("#### 3. Monthly Returns Heatmap")
            m_df = lab_res.get("monthly_returns", pd.DataFrame())
            if not m_df.empty:
                def style_ret_cell(val):
                    if not isinstance(val, str) or val in ("—", "N/A"):
                        return ""
                    try:
                        num = float(val.replace("%", "").replace("+", ""))
                        if num > 0:
                            return "background-color: rgba(38, 166, 154, 0.25); color: #26a69a; font-weight: bold;"
                        elif num < 0:
                            return "background-color: rgba(239, 83, 80, 0.25); color: #ef5350; font-weight: bold;"
                    except Exception:
                        pass
                    return ""
                st.dataframe(m_df.style.map(style_ret_cell), use_container_width=True, height=220)
            else:
                st.info("No monthly returns data available.")

            # 4. Trade List
            st.markdown("---")
            st.markdown("#### 4. Completed Trade Log")
            t_df = lab_res.get("trades_df", pd.DataFrame())
            if not t_df.empty:
                st.dataframe(t_df, use_container_width=True, height=250)
            else:
                st.info("No trades executed during this backtest period.")
        else:
            st.info("Adjust the strategy controls on the left and click **[▶ Run Backtest]** to generate results.")


# ── Tab: Tax Report ────────────────────────────────────────────────────────────

with tab_tax:
    st.subheader("📄 Annual Tax Report Generator")
    st.markdown("Generate and export downloadable annual tax reports for paper trading activity.")

    col_t1, col_t2 = st.columns([1, 2])

    with col_t1:
        st.markdown("### ⚙️ Report Controls")
        tax_year = st.selectbox("Tax Year", options=[2024, 2025, 2026, 2027], index=2)
        tax_country = st.selectbox("Country Jurisdiction", options=["India", "USA", "Other"], index=0)

        st.markdown("---")
        gen_tax_btn = st.button("📄 Generate Report", type="primary", use_container_width=True)

        if gen_tax_btn or "tax_report_result" not in st.session_state:
            with st.spinner("Generating tax report..."):
                rep_res = run_tax_report_trigger(year=tax_year, country=tax_country)
                st.session_state["tax_report_result"] = rep_res

        current_rep = st.session_state.get("tax_report_result")
        if current_rep:
            st.markdown("---")
            st.markdown("### 💾 Export Options")
            pdf_path = current_rep.get("pdf_path")
            csv_path = current_rep.get("csv_path")

            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Download as PDF",
                        data=f.read(),
                        file_name=f"tax_{tax_year}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            if csv_path and Path(csv_path).exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 Download as CSV",
                        data=f.read(),
                        file_name=f"tax_{tax_year}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    with col_t2:
        st.markdown("### 📋 Report Preview")
        rep = st.session_state.get("tax_report_result")
        if rep:
            # 1. Header
            h = rep.get("header", {})
            st.markdown(f"## {h.get('title', 'Annual Trade Report')}")
            st.markdown(f"**Year:** `{h.get('year')}` | **Generated:** `{h.get('generated_date')}` | **System:** `{h.get('system')}`")
            st.info(f"ℹ️ {h.get('note')}")

            # 2. Annual Summary
            st.markdown("---")
            st.markdown("#### 1. Annual Summary")
            s = rep.get("annual_summary", {})
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)
            with s_col1:
                st.metric("Total Trades", s.get("total_trades", 0))
                st.metric("Gross Gains", f"+${s.get('gross_gains', 0.0):,.2f}")
            with s_col2:
                st.metric("Winning Trades", f"{s.get('winning_trades_count', 0)} ({s.get('winning_trades_pct', 0.0)}%)")
                st.metric("Gross Losses", f"-${abs(s.get('gross_losses', 0.0)):,.2f}")
            with s_col3:
                st.metric("Losing Trades", f"{s.get('losing_trades_count', 0)} ({s.get('losing_trades_pct', 0.0)}%)")
                st.metric("Total Fees", f"-${abs(s.get('total_fees', 0.0)):,.2f}")
            with s_col4:
                net_p = s.get("net_pnl", 0.0)
                st.metric("Net P&L", f"{'+' if net_p > 0 else ''}${net_p:,.2f}")

            # 3. Capital Gains Breakdown
            st.markdown("---")
            st.markdown(f"#### 2. Capital Gains Breakdown ({tax_country})")
            tb = rep.get("tax_breakdown", {})
            for rule in tb.get("rules_description", []):
                st.markdown(f"- {rule}")
            st.markdown(f"**Estimated Paper Tax Owed:** `${tb.get('estimated_tax', 0.0):,.2f}`")

            # 4. Monthly Breakdown
            st.markdown("---")
            st.markdown("#### 3. Monthly Breakdown")
            mb = rep.get("monthly_breakdown", [])
            if mb:
                mb_df = pd.DataFrame(mb)
                mb_df["Net PnL ($)"] = mb_df["Net PnL ($)"].apply(lambda v: f"{'+' if v > 0 else ''}${v:,.2f}")
                st.dataframe(mb_df, use_container_width=True, height=200)

            # 5. Trade Summary Table
            st.markdown("---")
            st.markdown("#### 4. Trade Summary Table")
            tr = rep.get("trades", [])
            if tr:
                tr_df = pd.DataFrame(tr).rename(columns={
                    "entry_date": "Date Bought",
                    "exit_date": "Date Sold",
                    "ticker": "Ticker",
                    "shares": "Quantity",
                    "buy_price": "Buy Price",
                    "sell_price": "Sell Price",
                    "gross_pnl": "Gross P&L ($)",
                    "fees": "Fees ($)",
                    "net_pnl": "Net P&L ($)",
                    "holding_days": "Holding Days",
                    "term": "Term",
                })
                st.dataframe(tr_df, use_container_width=True, height=250)
            else:
                st.info("No closed trades recorded for this year.")

            # 6. Disclaimer
            st.markdown("---")
            st.warning(f"⚖️ **Disclaimer:** {rep.get('disclaimer')}")


# ── Tab: Ask AI Chatbot ────────────────────────────────────────────────────────

with tab_ai:
    col_ai_hdr, col_ai_clr = st.columns([4, 1])
    with col_ai_hdr:
        st.subheader("🤖 AI Portfolio Assistant")
        st.markdown("Ask questions about portfolio holdings, recent trades, risk rules, or system decisions in plain English.")
    with col_ai_clr:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    # Initialize chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Display suggested prompt buttons
    st.markdown("**Suggested Questions:**")
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    selected_prompt = None

    if col_s1.button("📊 Portfolio summary", use_container_width=True):
        selected_prompt = "Can you give me a full summary of my portfolio right now?"
    if col_s2.button("📈 How am I doing?", use_container_width=True):
        selected_prompt = "How am I doing today and what is my overall performance?"
    if col_s3.button("⚠️ Any alerts?", use_container_width=True):
        selected_prompt = "Are there any active risk alerts or circuit breakers?"
    if col_s4.button("🏆 Best trade so far", use_container_width=True):
        selected_prompt = "What is my best performing trade so far?"
    if col_s5.button("❓ What happened today?", use_container_width=True):
        selected_prompt = "What happened today in the market and in my portfolio?"

    # Display existing chat history
    st.markdown("---")
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your portfolio...")
    prompt_to_process = selected_prompt or user_input

    if prompt_to_process:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = ask_chatbot_trigger(
                    user_message=prompt_to_process,
                    chat_history=st.session_state["chat_messages"],
                )
                st.markdown(response_text)
                st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})


# ── Tab 5: Risk Engine & Audit Logs ───────────────────────────────────────────

with tab5:
    st.subheader("System Event Log & Audit Trail")

    # Filter selector
    log_filter = st.selectbox("Filter by Level", ["ALL", "INFO", "WARNING", "ERROR"])
    filtered_events = events_df
    if log_filter != "ALL":
        filtered_events = events_df[events_df["level"] == log_filter]

    if not filtered_events.empty:
        st.dataframe(
            filtered_events,
            use_container_width=True,
            height=400,
        )
    else:
        st.info(f"No events found matching level '{log_filter}'.")

    # Audit Report Markdown viewer if available
    audit_md = st.session_state.get("latest_audit_md")
    if audit_md:
        st.markdown("---")
        st.subheader("Latest Daily Pipeline Audit Report")
        st.markdown(audit_md)


# ── Tab 6: Long-Term Fundamentals Screener ─────────────────────────────────────

with tab6:
    st.subheader("🏛️ Long-Term Fundamentals-Based Stock Screener (3–5+ Year Horizon)")
    st.caption("Rule-based, 5-pillar fundamental analysis evaluating valuation, profitability, solvency, cash flow, and capital stewardship.")

    st.info(
        "ℹ️ **ARCHITECTURAL ISOLATION GUARANTEE**: "
        "This long-term fundamental screener is **strictly informational** and completely decoupled from the automated 5-day paper trading pipeline. "
        "High scores here do **NOT** generate orders, bypass the 200-day regime filter, or enter the paper broker."
    )

    col_scr1, col_scr2 = st.columns([1, 1])
    with col_scr1:
        if st.button("🔍 Run Full Fundamental Screen (~45 Curated Tickers)", use_container_width=True):
            with st.spinner("Screening curated universe via yfinance..."):
                scr_df, scr_cards = run_fundamental_screen_trigger(full_universe=True, force=False)
                st.session_state["screener_df"] = scr_df
                st.session_state["screener_cards"] = scr_cards
                st.success(f"Screened {len(scr_df)} companies successfully!")

    with col_scr2:
        if st.button("⚡ Quick Screen (Top 15 Tickers)", use_container_width=True):
            with st.spinner("Running quick screen..."):
                scr_df, scr_cards = run_fundamental_screen_trigger(full_universe=False, force=False)
                st.session_state["screener_df"] = scr_df
                st.session_state["screener_cards"] = scr_cards
                st.success(f"Screened {len(scr_df)} companies!")

    # Load screener data
    if "screener_df" not in st.session_state:
        scr_df, scr_cards = get_fundamental_screener_data()
        st.session_state["screener_df"] = scr_df
        st.session_state["screener_cards"] = scr_cards
    else:
        scr_df = st.session_state["screener_df"]
        scr_cards = st.session_state.get("screener_cards", [])

    if not scr_df.empty:
        # Metrics cards
        t1_count = sum(1 for t in scr_df["tier"] if "Tier 1" in str(t))
        t2_count = sum(1 for t in scr_df["tier"] if "Tier 2" in str(t))
        t3_count = sum(1 for t in scr_df["tier"] if "Tier 3" in str(t))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Companies", len(scr_df))
        with c2:
            st.metric("Tier 1 (High Quality)", f"{t1_count}", delta=f"{t1_count/len(scr_df)*100:.0f}% of pool")
        with c3:
            st.metric("Tier 2 (Moderate)", f"{t2_count}")
        with c4:
            st.metric("Tier 3 (Elevated Risk)", f"{t3_count}")

        # Sector filter
        all_sectors = ["All Sectors"] + sorted(list(scr_df["sector"].unique()))
        selected_sector = st.selectbox("Filter by Sector", all_sectors, key="screener_sector_filter")
        display_df = scr_df if selected_sector == "All Sectors" else scr_df[scr_df["sector"] == selected_sector]

        st.markdown("#### 5-Pillar Scorecard Master Table")
        st.caption("5 pillars scored out of 20 pts each (Max 100). All pillar columns shown by default for full checklist transparency.")

        table_cols = [
            "ticker", "company", "sector", "score", "tier",
            "val_20", "prof_20", "solv_20", "cash_20", "grow_20",
            "pe", "roe", "de", "fcf", "div_yield"
        ]
        col_rename = {
            "ticker": "Ticker",
            "company": "Company",
            "sector": "Sector",
            "score": "Score (100)",
            "tier": "Quality Tier",
            "val_20": "Valuation (20)",
            "prof_20": "Profitability (20)",
            "solv_20": "Solvency (20)",
            "cash_20": "Cash Flow (20)",
            "grow_20": "Growth/Div (20)",
            "pe": "P/E",
            "roe": "ROE",
            "de": "D/E",
            "fcf": "FCF ($B)",
            "div_yield": "Div Yield",
        }
        st.dataframe(
            display_df[table_cols].rename(columns=col_rename),
            use_container_width=True,
            height=350,
        )

        # Expandable Deep-Dive Card
        st.markdown("---")
        st.markdown("#### Company Deep-Dive Scorecard")
        ticker_choices = list(display_df["ticker"])
        if ticker_choices:
            chosen_ticker = st.selectbox("Select Ticker for Detailed Breakdown", ticker_choices)
            chosen_card = next((c for c in scr_cards if c.ticker == chosen_ticker), None)
            if chosen_card:
                col_cd1, col_cd2 = st.columns([1, 2])
                with col_cd1:
                    st.markdown(f"### {chosen_card.ticker} — {chosen_card.company_name}")
                    st.markdown(f"**Sector**: `{chosen_card.sector}`")
                    st.markdown(f"**Composite Score**: `{chosen_card.composite_score} / 100`")
                    st.markdown(f"**Quality Tier**: `{chosen_card.tier}`")
                    st.markdown(f"**Data Status**: `{chosen_card.status}`")
                    if chosen_card.missing_fields:
                        st.warning(f"Missing Fields: {', '.join(chosen_card.missing_fields)}")
                with col_cd2:
                    st.markdown("**5-Pillar Evaluation Details**:")
                    for p_name, p_score in chosen_card.pillar_scores.items():
                        icon = "✅" if p_score.passed else ("⚠️" if p_score.score > 0 else "❌")
                        st.markdown(f"- {icon} **{p_name}** (`{p_score.score} / {p_score.max_score} pts`): {p_score.reason}")
    else:
        st.info("No fundamental data available yet. Click **[⚡ Quick Screen]** above to run.")

