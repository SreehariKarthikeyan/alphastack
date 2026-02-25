"""
engine.py
---------
Rolling backtest engine — Phase 1.5 (Signal Hardening).

Strategy
--------
* Universe: stocks ranked by composite_score each month.
* Selection: top-3 by composite score.
* Holding period: rebalanced every ~21 trading days (monthly).
* Weighting: equal-weight.

Metrics computed
----------------
* CAGR
* Maximum Drawdown
* Sharpe Ratio (annualised, risk-free rate = 0)
* Win rate (% of positive-return months)
* Top-decile vs bottom-decile forward return spread
* Per-regime performance (bull / bear / high-vol)
* Comparison vs SPY buy-and-hold benchmark
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

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


def _win_rate(returns: pd.Series, period_days: int = 21) -> float:
    """Fraction of *period_days* windows with positive cumulative return."""
    if returns.empty:
        return float("nan")
    # Group into non-overlapping periods
    n_periods = max(1, len(returns) // period_days)
    wins = 0
    for i in range(n_periods):
        chunk = returns.iloc[i * period_days : (i + 1) * period_days]
        if (1 + chunk).prod() - 1 > 0:
            wins += 1
    return wins / n_periods


# ---------------------------------------------------------------------------
# Decile spread
# ---------------------------------------------------------------------------

def _decile_spread(
    scored_df: pd.DataFrame,
    price_df: pd.DataFrame,
    forward_days: int = 21,
) -> float:
    """
    Compute the average forward return difference between the top decile
    and bottom decile of stocks ranked by composite score.

    At each rebalance point (monthly), we measure the forward return of
    top-decile vs bottom-decile stocks and return the average spread.
    """
    scored_df = scored_df.copy()
    price_df = price_df.copy()
    scored_df["date"] = pd.to_datetime(scored_df["date"])
    price_df["date"] = pd.to_datetime(price_df["date"])

    pivot = price_df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()
    all_dates = sorted(pivot.index)

    rebalance_dates = all_dates[::21]
    spreads: List[float] = []

    for rdate in rebalance_dates:
        scores_asof = scored_df[scored_df["date"] <= rdate]
        if scores_asof.empty:
            continue
        latest = scores_asof.groupby("ticker")["composite_score"].last()
        if len(latest) < 5:
            continue

        n = len(latest)
        decile_size = max(1, n // 10)
        sorted_tickers = latest.sort_values(ascending=False)

        top_tickers = sorted_tickers.head(max(decile_size, 1)).index.tolist()
        bottom_tickers = sorted_tickers.tail(max(decile_size, 1)).index.tolist()

        # Forward return
        rdate_idx = all_dates.index(rdate) if rdate in all_dates else None
        if rdate_idx is None:
            continue
        end_idx = min(rdate_idx + forward_days, len(all_dates) - 1)
        if end_idx <= rdate_idx:
            continue

        start_prices = pivot.loc[all_dates[rdate_idx]]
        end_prices = pivot.loc[all_dates[end_idx]]

        top_ret = _avg_return(start_prices, end_prices, top_tickers)
        bottom_ret = _avg_return(start_prices, end_prices, bottom_tickers)

        if not np.isnan(top_ret) and not np.isnan(bottom_ret):
            spreads.append(top_ret - bottom_ret)

    return float(np.mean(spreads)) if spreads else float("nan")


def _avg_return(
    start: pd.Series, end: pd.Series, tickers: List[str]
) -> float:
    """Average return of *tickers* from *start* to *end* prices."""
    rets = []
    for t in tickers:
        if t in start.index and t in end.index:
            s, e = start[t], end[t]
            if pd.notna(s) and pd.notna(e) and s > 0:
                rets.append(e / s - 1)
    return float(np.mean(rets)) if rets else float("nan")


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
      cagr, max_drawdown, sharpe, win_rate,
      benchmark_cagr, benchmark_max_drawdown, benchmark_sharpe,
      decile_spread, holdings_log,
      top3_returns, bottom3_returns
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
    bottom_daily: List[float] = []
    bottom_dates: List[pd.Timestamp] = []
    holdings_log: List[dict] = []

    current_holdings: List[str] = []
    current_bottom: List[str] = []
    next_rebalance_idx = 0

    for _i, date in enumerate(all_dates[1:], start=1):
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
            current_bottom = scores_today["ticker"].tail(top_n).tolist()

            holdings_log.append({"date": date, "holdings": list(current_holdings)})
            next_rebalance_idx += 1

        if not current_holdings:
            continue

        # Equal-weight portfolio return for this day (top N)
        day_rets = []
        for ticker in current_holdings:
            if ticker in daily_returns.columns:
                r = daily_returns.loc[date, ticker]
                if not np.isnan(r):
                    day_rets.append(r)

        if day_rets:
            portfolio_daily.append(np.mean(day_rets))
            portfolio_dates.append(date)

        # Bottom N portfolio return for comparison
        bot_rets = []
        for ticker in current_bottom:
            if ticker in daily_returns.columns:
                r = daily_returns.loc[date, ticker]
                if not np.isnan(r):
                    bot_rets.append(r)

        if bot_rets:
            bottom_daily.append(np.mean(bot_rets))
            bottom_dates.append(date)

    portfolio_returns = pd.Series(portfolio_daily, index=portfolio_dates)
    bottom_returns = pd.Series(bottom_daily, index=bottom_dates)

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

    # ---- Decile spread ----
    decile_spread = _decile_spread(scored_df, price_df)

    return {
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "bottom_returns": bottom_returns,
        "cagr": _cagr(portfolio_returns),
        "max_drawdown": _max_drawdown(portfolio_returns),
        "sharpe": _sharpe(portfolio_returns),
        "win_rate": _win_rate(portfolio_returns),
        "benchmark_cagr": _cagr(benchmark_returns) if not benchmark_returns.empty else float("nan"),
        "benchmark_max_drawdown": _max_drawdown(benchmark_returns) if not benchmark_returns.empty else float("nan"),
        "benchmark_sharpe": _sharpe(benchmark_returns) if not benchmark_returns.empty else float("nan"),
        "decile_spread": decile_spread,
        "top3_cagr": _cagr(portfolio_returns),
        "bottom3_cagr": _cagr(bottom_returns) if not bottom_returns.empty else float("nan"),
        "holdings_log": holdings_log,
    }


def print_backtest_report(results: dict) -> None:
    """Pretty-print the backtest results to stdout."""
    print("\n" + "=" * 55)
    print("  NYLARIS BACKTEST REPORT")
    print("=" * 55)
    print(f"  Strategy CAGR        : {results['cagr']:.2%}")
    print(f"  Strategy Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Strategy Sharpe      : {results['sharpe']:.2f}")
    print(f"  Win Rate (monthly)   : {results['win_rate']:.1%}")
    print("-" * 55)
    if not math.isnan(results.get("benchmark_cagr", float("nan"))):
        print(f"  SPY CAGR             : {results['benchmark_cagr']:.2%}")
        print(f"  SPY Max Drawdown     : {results['benchmark_max_drawdown']:.2%}")
        print(f"  SPY Sharpe           : {results['benchmark_sharpe']:.2f}")
    print("-" * 55)
    print(f"  Top-3 CAGR           : {results.get('top3_cagr', float('nan')):.2%}")
    print(f"  Bottom-3 CAGR        : {results.get('bottom3_cagr', float('nan')):.2%}")
    ds = results.get("decile_spread", float("nan"))
    if not math.isnan(ds):
        print(f"  Decile Spread (avg)  : {ds:.2%}")
    print("=" * 55)

    if results.get("holdings_log"):
        print("\nSample rebalance dates (first 5):")
        for entry in results["holdings_log"][:5]:
            tickers = ", ".join(entry["holdings"])
            print(f"  {entry['date'].date()}  →  {tickers}")
    print()


# ---------------------------------------------------------------------------
# Regime testing
# ---------------------------------------------------------------------------

def _classify_regime(
    benchmark_returns: pd.Series,
    window: int = 63,
    vol_window: int = 21,
    vol_threshold: float = 0.25,
) -> pd.Series:
    """
    Classify each trading day into a market regime.

    Regimes
    -------
    bull   – rolling 3-month benchmark return > 0 and vol below threshold
    bear   – rolling 3-month benchmark return < 0 and vol below threshold
    high_vol – rolling annualised volatility above *vol_threshold*

    Returns
    -------
    Series indexed like *benchmark_returns* with values in
    {"bull", "bear", "high_vol"}.
    """
    rolling_ret = benchmark_returns.rolling(window, min_periods=window // 2).sum()
    rolling_vol = benchmark_returns.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)

    regime = pd.Series("bull", index=benchmark_returns.index)
    regime[rolling_ret < 0] = "bear"
    regime[rolling_vol > vol_threshold] = "high_vol"
    return regime


def run_regime_backtest(results: dict) -> Dict[str, dict]:
    """
    Split the backtest results by market regime and compute per-regime
    performance metrics.

    Parameters
    ----------
    results : Dict returned by ``run_backtest``.

    Returns
    -------
    Dict keyed by regime name ("bull", "bear", "high_vol") where each value
    is a dict with keys: cagr, max_drawdown, sharpe, win_rate, n_days.
    """
    portfolio_returns = results["portfolio_returns"]
    benchmark_returns = results["benchmark_returns"]

    if benchmark_returns.empty or portfolio_returns.empty:
        return {}

    regimes = _classify_regime(benchmark_returns)

    regime_results: Dict[str, dict] = {}
    for regime_name in ("bull", "bear", "high_vol"):
        mask = regimes == regime_name
        regime_rets = portfolio_returns[mask].dropna()

        if len(regime_rets) < 10:
            regime_results[regime_name] = {
                "cagr": float("nan"),
                "max_drawdown": float("nan"),
                "sharpe": float("nan"),
                "win_rate": float("nan"),
                "n_days": len(regime_rets),
            }
            continue

        regime_results[regime_name] = {
            "cagr": _cagr(regime_rets),
            "max_drawdown": _max_drawdown(regime_rets),
            "sharpe": _sharpe(regime_rets),
            "win_rate": _win_rate(regime_rets),
            "n_days": len(regime_rets),
        }

    return regime_results


def print_regime_report(regime_results: Dict[str, dict]) -> None:
    """Pretty-print the per-regime backtest results."""
    if not regime_results:
        print("\n  (No regime data available — benchmark returns needed)\n")
        return

    print("\n" + "=" * 55)
    print("  REGIME PERFORMANCE BREAKDOWN")
    print("=" * 55)
    for regime, metrics in regime_results.items():
        label = regime.replace("_", " ").title()
        n = metrics["n_days"]
        print(f"\n  {label}  ({n} trading days)")
        print(f"    CAGR         : {metrics['cagr']:.2%}")
        print(f"    Max Drawdown : {metrics['max_drawdown']:.2%}")
        print(f"    Sharpe       : {metrics['sharpe']:.2f}")
        print(f"    Win Rate     : {metrics['win_rate']:.1%}")
    print("=" * 55 + "\n")
