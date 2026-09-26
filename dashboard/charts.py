"""
dashboard/charts.py — Altair visualization builders for Streamlit Performance tab (Section 6 Item 1).

Renders four core institutional-grade visual performance charts:
1. Portfolio vs SPY Benchmark Line Chart (with red underperformance shading)
2. Daily Returns Frequency Histogram (green positive, red negative, vertical zero line)
3. Win/Loss Closed Trades Bar Chart (green profit, red loss, dollar height)
4. Sector Allocation Donut Chart (color-coded sector slices + cash reserve)
5. Candlestick Chart with Trade Markers (Plotly)
"""

from __future__ import annotations

from typing import Optional
import altair as alt
import pandas as pd
import plotly.graph_objects as go


def build_portfolio_vs_spy_chart(df: pd.DataFrame) -> Optional[alt.LayerChart]:
    """
    Builds the Portfolio vs SPY Benchmark line chart.

    - X axis: Date
    - Y axis: Dollar Value ($)
    - Lines: "My Portfolio" (solid blue) and "SPY Benchmark" (dashed amber)
    - Shaded Area: Highlighted in red (#ef5350) for any periods where Portfolio is below SPY
    """
    if df.empty:
        return None

    # Melt data for dual-line encoding
    plot_df = df.copy()
    if "date" in plot_df.columns:
        plot_df["date"] = pd.to_datetime(plot_df["date"])

    # Base chart
    base = alt.Chart(plot_df).encode(
        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%Y-%m-%d", labelAngle=-45))
    )

    # Line 1: My Portfolio
    line_portfolio = base.mark_line(color="#2962ff", strokeWidth=2.5).encode(
        y=alt.Y("My Portfolio:Q", title="Dollar Value ($)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("My Portfolio:Q", format="$,.2f", title="My Portfolio"),
            alt.Tooltip("SPY Benchmark:Q", format="$,.2f", title="SPY Benchmark"),
        ],
    )

    # Line 2: SPY Benchmark
    line_spy = base.mark_line(color="#ff9800", strokeWidth=2.0, strokeDash=[5, 4]).encode(
        y=alt.Y("SPY Benchmark:Q", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("My Portfolio:Q", format="$,.2f", title="My Portfolio"),
            alt.Tooltip("SPY Benchmark:Q", format="$,.2f", title="SPY Benchmark"),
        ],
    )

    layers = [line_portfolio, line_spy]

    # Red shaded area highlighting any periods where portfolio is below SPY
    under_df = plot_df[plot_df["underperforming"]].copy()
    if not under_df.empty:
        shade_under = alt.Chart(under_df).mark_area(
            opacity=0.35,
            color="#ef5350",
        ).encode(
            x=alt.X("date:T"),
            y=alt.Y("My Portfolio:Q"),
            y2=alt.Y2("SPY Benchmark:Q"),
        )
        points_under = alt.Chart(under_df).mark_circle(
            size=60,
            color="#ef5350",
        ).encode(
            x=alt.X("date:T"),
            y=alt.Y("My Portfolio:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date (Underperforming)", format="%Y-%m-%d"),
                alt.Tooltip("My Portfolio:Q", format="$,.2f", title="My Portfolio"),
                alt.Tooltip("SPY Benchmark:Q", format="$,.2f", title="SPY Benchmark"),
                alt.Tooltip("deficit:Q", format="$,.2f", title="Deficit vs SPY"),
            ],
        )
        layers.extend([shade_under, points_under])

    chart = alt.layer(*layers).properties(
        title="Portfolio vs SPY Benchmark Value ($)",
        height=380,
        usermeta={
            "embedOptions": {
                "actions": False,
                "renderer": "svg",
                "theme": "dark"
            }
        }
    ).interactive()

    return chart


def build_daily_returns_histogram(df: pd.DataFrame) -> Optional[alt.LayerChart]:
    """
    Builds the Daily Returns frequency histogram.

    - Frequency of daily % returns
    - Green bars: positive return days
    - Red bars: negative return days
    - Vertical dashed white line at 0.0%
    """
    if df.empty:
        return None

    bars = alt.Chart(df).mark_bar(size=26).encode(
        x=alt.X("bin_center:Q", title="Daily Return (%)", axis=alt.Axis(format="+.2f")),
        y=alt.Y("count:Q", title="Frequency (Days)", axis=alt.Axis(tickMinStep=1)),
        color=alt.Color(
            "sign:N",
            scale=alt.Scale(
                domain=["Positive", "Negative"],
                range=["#26a69a", "#ef5350"],
            ),
            legend=alt.Legend(title="Return Type"),
        ),
        tooltip=[
            alt.Tooltip("bin_label:N", title="Return Range"),
            alt.Tooltip("count:Q", title="Frequency (Days)"),
            alt.Tooltip("sign:N", title="Type"),
        ],
    )

    zero_line = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(
        color="#ffffff",
        strokeDash=[4, 4],
        strokeWidth=2,
    ).encode(x="zero:Q")

    chart = alt.layer(bars, zero_line).properties(
        title="Distribution of Daily Returns (%)",
        height=320,
        usermeta={
            "embedOptions": {
                "actions": False,
                "renderer": "svg",
                "theme": "dark"
            }
        }
    )

    return chart


def build_win_loss_trade_chart(df: pd.DataFrame) -> Optional[alt.LayerChart]:
    """
    Builds the Win/Loss Closed Trades bar chart.

    - Each bar = one closed trade
    - Green bar = profitable trade (net PnL >= 0)
    - Red bar = losing trade (net PnL < 0)
    - Bar height = profit/loss amount ($)
    - Horizontal baseline at $0
    """
    if df.empty:
        return None

    bars = alt.Chart(df).mark_bar(width=22).encode(
        x=alt.X("trade_label:N", sort=None, title="Closed Trade (Order of Execution)"),
        y=alt.Y("net_pnl:Q", title="Realized Profit / Loss ($)"),
        color=alt.Color(
            "sign:N",
            scale=alt.Scale(
                domain=["Win", "Loss"],
                range=["#26a69a", "#ef5350"],
            ),
            legend=alt.Legend(title="Outcome"),
        ),
        tooltip=[
            alt.Tooltip("trade_label:N", title="Trade"),
            alt.Tooltip("date:N", title="Date"),
            alt.Tooltip("ticker:N", title="Ticker"),
            alt.Tooltip("net_pnl:Q", format="$,.2f", title="Net Realized PnL"),
            alt.Tooltip("sign:N", title="Outcome"),
        ],
    )

    baseline = alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(
        color="#ffffff",
        strokeDash=[3, 3],
        strokeWidth=1.5,
    ).encode(y="zero:Q")

    chart = alt.layer(bars, baseline).properties(
        title="Realized PnL per Closed Trade ($)",
        height=320,
        usermeta={
            "embedOptions": {
                "actions": False,
                "renderer": "svg",
                "theme": "dark"
            }
        }
    )

    return chart


def build_sector_allocation_donut(df: pd.DataFrame) -> Optional[alt.Chart]:
    """
    Builds the Sector Allocation Donut Chart.

    - Current portfolio split by sector
    - Cash shown as its own slice
    - Color coded by sector
    """
    if df.empty:
        return None

    # Custom color palette ensuring Cash stands out nicely
    chart = alt.Chart(df).mark_arc(innerRadius=65, outerRadius=125).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color(
            "sector:N",
            scale=alt.Scale(scheme="category10"),
            legend=alt.Legend(title="Portfolio Component"),
        ),
        tooltip=[
            alt.Tooltip("sector:N", title="Sector"),
            alt.Tooltip("value:Q", format="$,.2f", title="Allocation ($)"),
            alt.Tooltip("pct:Q", format=".1f", title="Weight (%)"),
        ],
    ).properties(
        title="Current Portfolio Sector & Cash Allocation",
        height=340,
        usermeta={
            "embedOptions": {
                "actions": False,
                "renderer": "svg",
                "theme": "dark"
            }
        }
    )

    return chart


def build_correlation_heatmap(df: pd.DataFrame) -> Optional[alt.LayerChart]:
    """
    Builds correlation heatmap for held positions.
    - Color scale: green = low correlation, red = high correlation.
    """
    if df.empty:
        return None

    corr_df = df.copy()
    if corr_df.index.name != "ticker" and "ticker" not in corr_df.columns:
        corr_df = corr_df.reset_index().rename(columns={"index": "ticker_a"})

    tickers = [c for c in corr_df.columns if c != "ticker_a"]
    if not tickers:
        return None

    records = []
    for idx, row in corr_df.iterrows():
        t_a = str(row.get("ticker_a", idx))
        for t_b in tickers:
            records.append({
                "ticker_a": t_a,
                "ticker_b": str(t_b),
                "correlation": float(row[t_b]),
            })

    tidy_df = pd.DataFrame(records)
    if tidy_df.empty:
        return None

    base = alt.Chart(tidy_df).encode(
        x=alt.X("ticker_a:O", title="Ticker A"),
        y=alt.Y("ticker_b:O", title="Ticker B"),
    )

    rects = base.mark_rect().encode(
        color=alt.Color(
            "correlation:Q",
            scale=alt.Scale(domain=[-1.0, 0.0, 0.70, 1.0], range=["#2e7d32", "#81c784", "#ffb74d", "#e53935"]),
            title="Correlation (r)",
        ),
        tooltip=[
            alt.Tooltip("ticker_a:O", title="Ticker A"),
            alt.Tooltip("ticker_b:O", title="Ticker B"),
            alt.Tooltip("correlation:Q", format=".2f", title="Correlation"),
        ],
    )

    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.value("white"),
    )

    return (rects + text).properties(
        title="Held Positions 60-Day Return Correlation Matrix",
        width=400,
        height=350,
    )


# ── Feature Importance Charts (Section 9 Item 2) ──────────────────────────────

def build_feature_importance_chart(rows: list) -> Optional[alt.Chart]:
    """
    Build a horizontal bar chart of feature importances sorted highest → lowest.
    Each bar labelled with score and percentage.

    Parameters
    ----------
    rows : list of {feature_name, importance_score}

    Returns
    -------
    altair.Chart or None if rows is empty.
    """
    return build_feature_importance_bar_chart(rows)


def build_feature_importance_bar_chart(rows: list) -> Optional[alt.Chart]:
    if not rows:
        return None

    df = pd.DataFrame(rows)[["feature_name", "importance_score"]].copy()
    df["importance_score"] = df["importance_score"].astype(float)
    df["pct_label"] = df["importance_score"].apply(lambda x: f"{x:.3f} ({x*100:.1f}%)")
    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    bars = (
        alt.Chart(df)
        .mark_bar(color="#4C9BE8", cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("importance_score:Q", title="Importance Score", axis=alt.Axis(format=".3f")),
            y=alt.Y("feature_name:N", sort="-x", title="Feature"),
            tooltip=[
                alt.Tooltip("feature_name:N", title="Feature"),
                alt.Tooltip("importance_score:Q", title="Score", format=".4f"),
                alt.Tooltip("pct_label:N", title="Share"),
            ],
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=4, fontSize=11, color="#E0E0E0")
        .encode(
            x=alt.X("importance_score:Q"),
            y=alt.Y("feature_name:N", sort="-x"),
            text=alt.Text("pct_label:N"),
        )
    )

    return (bars + labels).properties(
        title=alt.TitleParams(
            "Feature Importance (Normalised — sums to 100%)",
            fontSize=14,
            color="#FFFFFF",
        ),
        height=max(200, len(df) * 28),
        width=600,
    ).configure_axis(
        labelColor="#CCCCCC",
        titleColor="#AAAAAA",
        gridColor="#2A2A2A",
    ).configure_view(
        strokeWidth=0,
        fill="#1A1A2E",
    )


def build_shap_summary_chart(shap_rows: list) -> Optional[alt.Chart]:
    """
    Build a horizontal bar chart of global mean-absolute SHAP values, colour-coded
    by direction (UP=green, DOWN=red).

    Parameters
    ----------
    shap_rows : list of {feature_name, mean_abs_shap, mean_signed_shap, direction}
    """
    if not shap_rows:
        return None

    df = pd.DataFrame(shap_rows)
    df["mean_abs_shap"] = df["mean_abs_shap"].astype(float)
    df["color"] = df["direction"].apply(lambda d: "#4CAF50" if d == "UP" else "#EF5350")
    df["direction_label"] = df.apply(
        lambda r: f"High value → score {'UP' if r['direction']=='UP' else 'DOWN'} ({r['mean_signed_shap']:+.4f})",
        axis=1,
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP| Value"),
            y=alt.Y("feature_name:N", sort="-x", title="Feature"),
            color=alt.Color("direction:N", scale=alt.Scale(
                domain=["UP", "DOWN"], range=["#4CAF50", "#EF5350"]
            ), legend=alt.Legend(title="Direction")),
            tooltip=[
                alt.Tooltip("feature_name:N", title="Feature"),
                alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".5f"),
                alt.Tooltip("direction_label:N", title="Direction"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                "SHAP Global Feature Impact (Green=Positive, Red=Negative)",
                fontSize=14,
                color="#FFFFFF",
            ),
            height=max(200, len(df) * 28),
            width=600,
        )
    )
    return chart


def build_feature_importance_history_chart(rows: list) -> Optional[alt.Chart]:
    """
    Build a multi-line chart of top-5 feature importance across last N retrains.

    Parameters
    ----------
    rows : list of {date, feature_name, importance_score}
    """
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["importance_score"] = df["importance_score"].astype(float)
    df["date"] = pd.to_datetime(df["date"])

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Retrain Date"),
            y=alt.Y("importance_score:Q", title="Importance Score", axis=alt.Axis(format=".3f")),
            color=alt.Color("feature_name:N", title="Feature"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("feature_name:N", title="Feature"),
                alt.Tooltip("importance_score:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                "Top-5 Feature Importance Over Last Retrains",
                fontSize=14,
                color="#FFFFFF",
            ),
            height=300,
            width=700,
        )
    )
    return chart


# ── Section 10 Item 1: Leaderboard Chart Builder ──────────────────────────────

def build_leaderboard_equity_chart(df: pd.DataFrame) -> Optional[alt.Chart]:
    """
    Build a multi-line Altair chart plotting portfolio value over time for all strategy variants on the same chart.

    Parameters
    ----------
    df : pd.DataFrame with columns ['date', 'Strategy', 'portfolio_value']
    """
    if df.empty:
        return None

    df_clean = df.copy()
    df_clean["portfolio_value"] = df_clean["portfolio_value"].astype(float)
    df_clean["date"] = pd.to_datetime(df_clean["date"])

    chart = (
        alt.Chart(df_clean)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("portfolio_value:Q", title="Portfolio Value ($)", axis=alt.Axis(format="$,.2f")),
            color=alt.Color("Strategy:N", title="Strategy Variant", scale=alt.Scale(
                domain=["Conservative", "Balanced", "Aggressive"],
                range=["#4C9BE8", "#00E676", "#FF5252"],
            )),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("Strategy:N", title="Strategy"),
                alt.Tooltip("portfolio_value:Q", title="Value", format="$,.2f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                "Strategy Variant Portfolio Values Over Time",
                fontSize=14,
                color="#FFFFFF",
            ),
            height=350,
            width=700,
        )
        .configure_axis(
            labelColor="#CCCCCC",
            titleColor="#AAAAAA",
            gridColor="#2A2A2A",
        )
        .configure_view(
            strokeWidth=0,
            fill="#1A1A2E",
        )
    )
    return chart


# ── Candlestick Chart with Trade Markers (Plotly) ─────────────────────────────────

def build_candlestick_chart(
    ohlcv_df: pd.DataFrame,
    ticker: str,
    trades_df: Optional[pd.DataFrame] = None,
) -> Optional[go.Figure]:
    """
    Build an interactive Plotly candlestick chart with buy/sell markers.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        DataFrame with columns: date, open, high, low, close, volume.
        Should contain the last 30 trading days of data.
    ticker : str
        Ticker symbol for the chart title.
    trades_df : pd.DataFrame, optional
        DataFrame with columns: date, ticker, action (BUY/SELL), price.
        Used to mark entry/exit points on the chart.

    Returns
    -------
    plotly.graph_objects.Figure or None if ohlcv_df is empty.
    """
    if ohlcv_df.empty:
        return None

    df = ohlcv_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30).reset_index(drop=True)

    fig = go.Figure()

    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
    ))

    # Volume bars as secondary y-axis
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["volume"],
        name="Volume",
        marker_color="rgba(100, 100, 255, 0.3)",
        yaxis="y2",
        opacity=0.4,
    ))

    # Buy/Sell markers from trades
    if trades_df is not None and not trades_df.empty:
        ticker_trades = trades_df[trades_df["ticker"].str.upper() == ticker.upper()].copy()
        if not ticker_trades.empty:
            ticker_trades["date"] = pd.to_datetime(ticker_trades["date"])

            buys = ticker_trades[ticker_trades["action"].str.upper() == "BUY"]
            sells = ticker_trades[ticker_trades["action"].str.upper() == "SELL"]

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["date"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#26a69a",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="<b>BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ))

            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["date"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#ef5350",
                        line=dict(width=1, color="white"),
                    ),
                    hovertemplate="<b>SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ))

    # Layout
    fig.update_layout(
        title=f"{ticker} — Last 30 Trading Days (Candlestick + Volume)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=True,
        ),
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    return fig
