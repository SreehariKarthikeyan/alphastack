"""
ui.py
-----
Nylaris – Streamlit dashboard for checking stock signal details.

Run with:
    streamlit run nylaris/ui.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Nylaris Signal Engine",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar – settings
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "UNH", "XOM", "PG", "HD",
]

tickers_input = st.sidebar.text_area(
    "Tickers (one per line or comma-separated)",
    value="\n".join(DEFAULT_TICKERS),
    height=220,
)
tickers = [t.strip().upper() for t in tickers_input.replace(",", "\n").splitlines() if t.strip()]

mode = st.sidebar.selectbox(
    "Scoring mode",
    options=["phase1", "full"],
    index=0,
    help="phase1 uses only trend + momentum; full adds fundamentals & sentiment",
)

force_refresh = st.sidebar.checkbox(
    "Force refresh cached data",
    value=False,
    help="Re-download market data from Yahoo Finance",
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("📈 Nylaris Signal Engine")
st.caption("Phase-1 stock scoring dashboard — trend · momentum · composite rank")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_pipeline(tickers_key: tuple, mode: str, refresh: bool) -> pd.DataFrame:
    """Run the full Nylaris pipeline and return the composite-scored DataFrame."""
    from nylaris.data.market_data import fetch_ohlcv
    from nylaris.signals.trend import compute_trend_score
    from nylaris.signals.momentum import compute_momentum_score
    from nylaris.signals.sentiment import compute_sentiment_score
    from nylaris.data.news_data import fetch_news_sentiment
    from nylaris.scoring.composite import compute_composite_score

    ticker_list = list(tickers_key)
    df = fetch_ohlcv(tickers=ticker_list, force_refresh=refresh)
    if df.empty:
        return df

    df = compute_trend_score(df)
    df = compute_momentum_score(df)
    news_df = fetch_news_sentiment(tickers=ticker_list)
    df = compute_sentiment_score(df, news_df)

    if mode == "full":
        from nylaris.data.fundamentals import load_fundamentals
        from nylaris.signals.fundamentals import (
            compute_fundamental_score,
            align_fundamentals_to_market,
        )
        fund_raw = load_fundamentals(tickers=ticker_list)
        if not fund_raw.empty:
            fund_scored = compute_fundamental_score(fund_raw)
            df = align_fundamentals_to_market(df, fund_scored)

    df = compute_composite_score(df, mode=mode)
    return df


def _build_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    from nylaris.scoring.composite import build_snapshot
    return build_snapshot(df)


# State management
if "scored_df" not in st.session_state:
    st.session_state.scored_df = None
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

if run_button:
    with st.spinner("Running pipeline… this may take a minute on first run."):
        try:
            df = _run_pipeline(tuple(tickers), mode, force_refresh)
            if df.empty:
                st.error("No data returned. Check tickers and network access.")
            else:
                st.session_state.scored_df = df
                st.session_state.snapshot = _build_snapshot(df)
                st.success(f"Pipeline complete — {df['ticker'].nunique()} tickers processed.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")

# ---------------------------------------------------------------------------
# Snapshot table
# ---------------------------------------------------------------------------

if st.session_state.snapshot is not None:
    snap: pd.DataFrame = st.session_state.snapshot

    st.subheader("🏆 Ranked Snapshot")
    st.caption("Latest-date scores for each ticker, sorted by composite score (highest first)")

    score_cols = [c for c in ["composite_score", "trend_score", "momentum_score",
                               "sentiment_score", "fundamental_score"] if c in snap.columns]

    display = snap[["ticker"] + score_cols + ["volatility_regime"]].copy()

    for col in score_cols:
        display[col] = display[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------------------
    # Individual ticker detail
    # ---------------------------------------------------------------------------

    st.subheader("🔍 Ticker Detail")

    ticker_list = snap["ticker"].tolist()
    selected = st.selectbox("Select a ticker", options=ticker_list)

    if selected and st.session_state.scored_df is not None:
        full_df: pd.DataFrame = st.session_state.scored_df
        ticker_df = full_df[full_df["ticker"] == selected].sort_values("date")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**{selected} — Price History**")
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="#1f77b4"),
                ))
                if "sma_50" in ticker_df.columns:
                    fig.add_trace(go.Scatter(
                        x=ticker_df["date"],
                        y=ticker_df["sma_50"],
                        mode="lines",
                        name="SMA 50",
                        line=dict(color="#ff7f0e", dash="dash"),
                    ))
                if "sma_200" in ticker_df.columns:
                    fig.add_trace(go.Scatter(
                        x=ticker_df["date"],
                        y=ticker_df["sma_200"],
                        mode="lines",
                        name="SMA 200",
                        line=dict(color="#2ca02c", dash="dot"),
                    ))
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    height=350,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(ticker_df.set_index("date")["close"])

        with col2:
            st.markdown(f"**{selected} — Latest Scores**")
            latest = ticker_df.iloc[-1]

            score_fields = {
                "Composite Score": "composite_score",
                "Trend Score": "trend_score",
                "Momentum Score": "momentum_score",
                "Sentiment Score": "sentiment_score",
                "Fundamental Score": "fundamental_score",
                "Volatility Regime": "volatility_regime",
            }
            rows = []
            for label, col in score_fields.items():
                if col in latest.index and pd.notna(latest[col]):
                    val = latest[col]
                    rows.append({"Signal": label, "Value": f"{val:.3f}" if isinstance(val, float) else val})

            if rows:
                st.table(pd.DataFrame(rows).set_index("Signal"))

            st.markdown("**Latest OHLCV**")
            ohlcv_fields = {"Date": "date", "Open": "open", "High": "high",
                            "Low": "low", "Close": "close", "Volume": "volume"}
            ohlcv_rows = []
            for label, col in ohlcv_fields.items():
                if col in latest.index:
                    val = latest[col]
                    if col == "date":
                        val = str(val)[:10]
                    elif col == "volume":
                        val = f"{int(val):,}"
                    else:
                        val = f"{val:.2f}"
                    ohlcv_rows.append({"Field": label, "Value": val})
            if ohlcv_rows:
                st.table(pd.DataFrame(ohlcv_rows).set_index("Field"))

        # Composite score over time chart
        if "composite_score" in ticker_df.columns:
            st.markdown(f"**{selected} — Composite Score Over Time**")
            try:
                import plotly.graph_objects as go

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=ticker_df["date"],
                    y=ticker_df["composite_score"],
                    mode="lines",
                    name="Composite Score",
                    line=dict(color="#9467bd"),
                    fill="tozeroy",
                    fillcolor="rgba(148,103,189,0.15)",
                ))
                fig2.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.line_chart(ticker_df.set_index("date")["composite_score"])

else:
    st.info(
        "👈 Configure your settings in the sidebar, then click **▶ Run Pipeline** to load data and compute scores."
    )
    st.markdown("""
### How it works

| Step | Description |
|------|-------------|
| **Ingest** | Downloads 5 years of daily OHLCV data from Yahoo Finance |
| **Trend signals** | Price vs 200-DMA, MA crossover (50/200), ATR volatility regime |
| **Momentum signals** | RSI-14, 3-month return, 6-month return |
| **Sentiment** | News sentiment scores (placeholder in phase 1) |
| **Composite score** | Weighted combination of all signal scores (0 = weakest, 1 = strongest) |
""")
