"""
main.py
-------
Nylaris – Phase 1.5 Signal Engine  (CLI entry point)

Usage
-----
  # Full pipeline: ingest → signals → score → backtest → diagnostics
  python -m nylaris.main --run all

  # Ingest market data only
  python -m nylaris.main --run ingest

  # Compute signals & scores (requires cached market data)
  python -m nylaris.main --run signals

  # Run backtest (requires scored data)
  python -m nylaris.main --run backtest

  # Print today's ranked snapshot
  python -m nylaris.main --run snapshot

  # Run distribution diagnostics
  python -m nylaris.main --run diagnostics

Options
-------
  --tickers  AAPL MSFT ...   Override default universe
  --refresh                  Force refresh of cached data
  --mode     phase1|full     Scoring mode (default: phase1)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the repo root is on sys.path so `nylaris.*` imports work whether
# this file is run as `python nylaris/main.py`, `python -m nylaris.main`,
# or `streamlit run nylaris/main.py`.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nylaris",
        description="Nylaris Phase-1.5 Signal Engine",
    )
    p.add_argument(
        "--run",
        choices=["all", "ingest", "signals", "backtest", "snapshot", "diagnostics"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        default=None,
        help="Override the default stock universe",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh of cached market/fundamental data",
    )
    p.add_argument(
        "--mode",
        choices=["phase1", "full"],
        default="phase1",
        help="Composite scoring mode (default: phase1)",
    )
    return p


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_ingest(tickers: Optional[List[str]], refresh: bool) -> pd.DataFrame:
    from nylaris.data.market_data import fetch_ohlcv
    print("[1/5] Fetching OHLCV market data …")
    df = fetch_ohlcv(tickers=tickers, force_refresh=refresh)
    print(f"      {len(df):,} rows × {len(df.columns)} cols  |  "
          f"tickers: {df['ticker'].nunique()}")
    return df


def run_signals(market_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    from nylaris.signals.trend import compute_trend_score
    from nylaris.signals.momentum import compute_momentum_score
    from nylaris.signals.sentiment import compute_sentiment_score
    from nylaris.data.news_data import fetch_news_sentiment

    print("[2/5] Computing trend signals …")
    df = compute_trend_score(market_df)

    print("[3/5] Computing momentum signals …")
    df = compute_momentum_score(df)

    print("[4/5] Attaching sentiment scores …")
    news_df = fetch_news_sentiment(tickers=df["ticker"].unique().tolist())
    df = compute_sentiment_score(df, news_df)

    if mode == "full":
        from nylaris.data.fundamentals import load_fundamentals
        from nylaris.signals.fundamentals import (
            compute_fundamental_score,
            align_fundamentals_to_market,
        )
        print("      Loading fundamentals for full scoring mode …")
        fund_raw = load_fundamentals(tickers=df["ticker"].unique().tolist())
        if not fund_raw.empty:
            fund_scored = compute_fundamental_score(fund_raw)
            df = align_fundamentals_to_market(df, fund_scored)

    return df


def run_composite(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    from nylaris.scoring.composite import compute_composite_score
    print("[5/5] Computing composite scores …")
    df = compute_composite_score(df, mode=mode)
    return df


def run_backtest(df: pd.DataFrame) -> None:
    from nylaris.backtest.engine import (
        run_backtest as _run,
        print_backtest_report,
        run_regime_backtest,
        print_regime_report,
    )

    print("\n[backtest] Running rolling backtest …")
    results = _run(
        scored_df=df[["date", "ticker", "composite_score"]],
        price_df=df[["date", "ticker", "close"]],
    )
    print_backtest_report(results)

    # Regime breakdown
    regime_results = run_regime_backtest(results)
    print_regime_report(regime_results)


def run_snapshot(df: pd.DataFrame, mode: str = "phase1") -> None:
    from nylaris.scoring.composite import build_snapshot, save_snapshot
    from nylaris.scoring.history import save_daily_snapshot
    from nylaris.scoring.contributions import compute_contributions_table

    # Persist score history
    n_saved = save_daily_snapshot(df)
    print(f"\n[history] Saved {n_saved} rows to score history DB.")

    snap = build_snapshot(df)
    path = save_snapshot(snap)
    print(f"[snapshot] Saved → {path}\n")

    # Show ranked table with deltas and percentiles
    display_cols = ["ticker", "composite_score", "rank", "percentile_bucket"]
    optional = ["score_change_7d", "score_change_30d"]
    for c in optional:
        if c in snap.columns and snap[c].notna().any():
            display_cols.append(c)
    print(snap[display_cols].to_string(index=False))
    print()

    # Feature contributions
    print("[contributions] Feature breakdown:")
    contrib = compute_contributions_table(snap, mode=mode)
    print(contrib.to_string(index=False))
    print()


def run_diagnostics(df: pd.DataFrame) -> None:
    from nylaris.scoring.diagnostics import run_diagnostics as _diag, print_diagnostics

    diag = _diag(df)
    print_diagnostics(diag)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    stage = args.run
    tickers = args.tickers
    refresh = args.refresh
    mode = args.mode

    try:
        if stage in ("all", "ingest"):
            market_df = run_ingest(tickers, refresh)
        else:
            from nylaris.data.market_data import load_ohlcv
            market_df = load_ohlcv(tickers)
            if market_df.empty:
                print("ERROR: No cached market data found. Run with --run ingest first.")
                return 1

        if stage in ("all", "signals", "backtest", "snapshot", "diagnostics"):
            df = run_signals(market_df, mode)
            df = run_composite(df, mode)
        else:
            df = market_df  # ingest-only, nothing more to do

        if stage in ("all", "backtest"):
            run_backtest(df)

        if stage in ("all", "snapshot"):
            run_snapshot(df, mode=mode)

        if stage in ("all", "diagnostics"):
            run_diagnostics(df)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
