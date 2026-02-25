"""
main.py
-------
Nylaris – Phase 1 Signal Engine  (CLI entry point)

Usage
-----
  # Full pipeline: ingest → signals → score → backtest
  python -m nylaris.main --run all

  # Ingest market data only
  python -m nylaris.main --run ingest

  # Compute signals & scores (requires cached market data)
  python -m nylaris.main --run signals

  # Run backtest (requires scored data)
  python -m nylaris.main --run backtest

  # Print today's ranked snapshot
  python -m nylaris.main --run snapshot

Options
-------
  --tickers  AAPL MSFT ...   Override default universe
  --refresh                  Force refresh of cached data
  --mode     phase1|full     Scoring mode (default: phase1)
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nylaris",
        description="Nylaris Phase-1 Signal Engine",
    )
    p.add_argument(
        "--run",
        choices=["all", "ingest", "signals", "backtest", "snapshot"],
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
    from nylaris.backtest.engine import run_backtest as _run, print_backtest_report

    print("\n[backtest] Running rolling backtest …")
    results = _run(
        scored_df=df[["date", "ticker", "composite_score"]],
        price_df=df[["date", "ticker", "close"]],
    )
    print_backtest_report(results)


def run_snapshot(df: pd.DataFrame) -> None:
    from nylaris.scoring.composite import build_snapshot, save_snapshot

    snap = build_snapshot(df)
    path = save_snapshot(snap)
    print(f"\nSnapshot saved → {path}\n")
    print(snap.to_string(index=False))
    print()


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

        if stage in ("all", "signals", "backtest", "snapshot"):
            df = run_signals(market_df, mode)
            df = run_composite(df, mode)
        else:
            df = market_df  # ingest-only, nothing more to do

        if stage in ("all", "backtest"):
            run_backtest(df)

        if stage in ("all", "snapshot"):
            run_snapshot(df)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
