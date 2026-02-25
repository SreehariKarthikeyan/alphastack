"""
engine.py
---------
Rolling backtest engine.

Strategy
--------
* Universe: stocks ranked by composite_score each month.
* Selection: top-3 by composite score.
* Holding period: 3 months (≈63 trading days).
* Rebalance: monthly (≈21 trading days).

Metrics computed
----------------
* CAGR
* Maximum Drawdown
* Sharpe Ratio (annualised, risk-free rate = 0)
* Comparison vs SPY buy-and-hold benchmark
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate from a daily return series."""
    cumulative = (1 + returns).prod()
    n_years = len(returns) / 252
    if n_years <= 0 or cumulative <= 0:
        return float("nan")
    return cumulative ** (1 / n_years) - 1


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a daily return series."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return drawdown.min()


def _sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe Ratio."""
    excess = returns - risk_free_rate / 252
    std = excess.std()
    if std == 0:
        return float("nan")
    return (excess.mean() / std) * math.sqrt(252)


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    scored_df: pd.DataFrame,
    price_df: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    top_n: int = 3,
    rebalance_days: int = 21,
) -> dict:
    """
    Run a rolling monthly-rebalance backtest.

    Parameters
    ----------
    scored_df : DataFrame with columns [date, ticker, composite_score].
    price_df  : DataFrame with columns [date, ticker, close] (daily prices).
    spy_df    : Optional DataFrame with columns [date, close] for SPY
                benchmark.  If None, SPY is fetched from price_df if present,
                otherwise benchmark metrics are skipped.
    top_n     : Number of stocks to hold each period (default 3).
    rebalance_days : Approximate trading days between rebalances (default 21).

    Returns
    -------
    dict with keys:
      portfolio_returns, benchmark_returns,
      cagr, max_drawdown, sharpe,
      benchmark_cagr, benchmark_max_drawdown, benchmark_sharpe,
      holdings_log
    """
    scored_df = scored_df.copy()
    price_df = price_df.copy()

    scored_df["date"] = pd.to_datetime(scored_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])

    # Pivot prices: rows=date, cols=ticker
    pivot = price_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()

    # Daily returns per ticker
    daily_returns = pivot.pct_change()

    # Rebalance dates: every ~rebalance_days trading days
    all_dates = sorted(pivot.index)
    rebalance_dates = all_dates[::rebalance_days]

    portfolio_daily: List[float] = []
    portfolio_dates: List[pd.Timestamp] = []
    holdings_log: List[dict] = []

    current_holdings: List[str] = []
    next_rebalance_idx = 0

    for i, date in enumerate(all_dates[1:], start=1):
        prev_date = all_dates[i - 1]

        # Rebalance if we've reached the next rebalance date
        if next_rebalance_idx < len(rebalance_dates) and date >= rebalance_dates[next_rebalance_idx]:
            # Score as-of rebalance date
            scores_today = scored_df[scored_df["date"] <= date].sort_values("date")
            scores_today = scores_today.groupby("ticker").last().reset_index()
            scores_today = scores_today.sort_values("composite_score", ascending=False)

            # Only pick tickers with available price data on this date
            available = [t for t in scores_today["ticker"] if t in daily_returns.columns]
            scores_today = scores_today[scores_today["ticker"].isin(available)]
            current_holdings = scores_today["ticker"].head(top_n).tolist()

            holdings_log.append({"date": date, "holdings": list(current_holdings)})
            next_rebalance_idx += 1

        if not current_holdings:
            continue

        # Equal-weight portfolio return for this day
        day_rets = []
        for ticker in current_holdings:
            if ticker in daily_returns.columns:
                r = daily_returns.loc[date, ticker]
                if not np.isnan(r):
                    day_rets.append(r)

        if day_rets:
            portfolio_daily.append(np.mean(day_rets))
            portfolio_dates.append(date)

    portfolio_returns = pd.Series(portfolio_daily, index=portfolio_dates)

    # ---- Benchmark (SPY) ----
    if spy_df is not None:
        spy_df = spy_df.copy()
        spy_df["date"] = pd.to_datetime(spy_df["date"])
        spy_series = spy_df.set_index("date")["close"].sort_index()
        benchmark_returns = spy_series.pct_change().dropna()
    elif "SPY" in daily_returns.columns:
        benchmark_returns = daily_returns["SPY"].dropna()
    else:
        benchmark_returns = pd.Series(dtype=float)

    # Align benchmark to portfolio date range
    if not benchmark_returns.empty and not portfolio_returns.empty:
        benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).fillna(0)

    return {
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "cagr": _cagr(portfolio_returns),
        "max_drawdown": _max_drawdown(portfolio_returns),
        "sharpe": _sharpe(portfolio_returns),
        "benchmark_cagr": _cagr(benchmark_returns) if not benchmark_returns.empty else float("nan"),
        "benchmark_max_drawdown": _max_drawdown(benchmark_returns) if not benchmark_returns.empty else float("nan"),
        "benchmark_sharpe": _sharpe(benchmark_returns) if not benchmark_returns.empty else float("nan"),
        "holdings_log": holdings_log,
    }


def print_backtest_report(results: dict) -> None:
    """Pretty-print the backtest results to stdout."""
    print("\n" + "=" * 50)
    print("  NYLARIS BACKTEST REPORT")
    print("=" * 50)
    print(f"  Strategy CAGR        : {results['cagr']:.2%}")
    print(f"  Strategy Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Strategy Sharpe      : {results['sharpe']:.2f}")
    print("-" * 50)
    if not math.isnan(results.get("benchmark_cagr", float("nan"))):
        print(f"  SPY CAGR             : {results['benchmark_cagr']:.2%}")
        print(f"  SPY Max Drawdown     : {results['benchmark_max_drawdown']:.2%}")
        print(f"  SPY Sharpe           : {results['benchmark_sharpe']:.2f}")
    print("=" * 50)

    if results.get("holdings_log"):
        print("\nSample rebalance dates (first 5):")
        for entry in results["holdings_log"][:5]:
            tickers = ", ".join(entry["holdings"])
            print(f"  {entry['date'].date()}  →  {tickers}")
    print()
