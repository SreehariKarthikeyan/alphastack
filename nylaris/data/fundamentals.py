"""
fundamentals.py
---------------
Pull the last 8 quarters of fundamental data (revenue, EPS, gross margin,
debt/equity, ROE) for each ticker via yfinance and persist as Parquet.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .market_data import DEFAULT_UNIVERSE

DATA_DIR = Path(os.environ.get("NYLARIS_DATA_DIR", "nylaris_data"))
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"

_QUARTERS = 8  # how many trailing quarters to retain


def _ensure_dirs() -> None:
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_float(value) -> float:
    """Convert a value to float, returning NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fetch_fundamentals(
    tickers: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download quarterly fundamentals for *tickers* and return a combined
    DataFrame.

    Columns: date, ticker, revenue, eps, gross_margin, debt_equity, roe
    """
    _ensure_dirs()
    tickers = tickers or DEFAULT_UNIVERSE

    frames: List[pd.DataFrame] = []

    for ticker in tickers:
        cache_path = FUNDAMENTALS_DIR / f"{ticker}.parquet"

        if cache_path.exists() and not force_refresh:
            df = pd.read_parquet(cache_path)
            frames.append(df)
            continue

        try:
            tk = yf.Ticker(ticker)

            # --- Quarterly income statement ---
            inc = tk.quarterly_income_stmt
            if inc is None or inc.empty:
                inc = pd.DataFrame()

            # --- Quarterly balance sheet ---
            bal = tk.quarterly_balance_sheet
            if bal is None or bal.empty:
                bal = pd.DataFrame()

            # Collect dates from income statement columns (most recent first)
            if not inc.empty:
                dates = sorted(inc.columns, reverse=True)[:_QUARTERS]
            elif not bal.empty:
                dates = sorted(bal.columns, reverse=True)[:_QUARTERS]
            else:
                print(f"[fundamentals] WARNING: no data for {ticker}")
                continue

            rows: List[dict] = []
            for date in dates:
                row: dict = {"date": pd.Timestamp(date), "ticker": ticker}

                # Revenue
                for label in ("Total Revenue", "Revenue"):
                    if not inc.empty and label in inc.index:
                        row["revenue"] = _safe_float(inc.loc[label, date])
                        break
                else:
                    row["revenue"] = float("nan")

                # EPS  (Net Income / shares outstanding as proxy)
                if not inc.empty and "Net Income" in inc.index:
                    net_income = _safe_float(inc.loc["Net Income", date])
                    info = tk.info or {}
                    shares = _safe_float(
                        info.get("sharesOutstanding", info.get("impliedSharesOutstanding", float("nan")))
                    )
                    row["eps"] = net_income / shares if shares and not np.isnan(shares) else float("nan")
                else:
                    row["eps"] = float("nan")

                # Gross Margin
                if not inc.empty and "Gross Profit" in inc.index:
                    gp = _safe_float(inc.loc["Gross Profit", date])
                    rev = row.get("revenue", float("nan"))
                    row["gross_margin"] = gp / rev if rev and not np.isnan(rev) else float("nan")
                else:
                    row["gross_margin"] = float("nan")

                # Debt / Equity
                for d_label in ("Total Debt", "Long Term Debt"):
                    if not bal.empty and d_label in bal.index:
                        total_debt = _safe_float(bal.loc[d_label, date])
                        break
                else:
                    total_debt = float("nan")

                for e_label in ("Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity"):
                    if not bal.empty and e_label in bal.index:
                        equity = _safe_float(bal.loc[e_label, date])
                        break
                else:
                    equity = float("nan")

                row["debt_equity"] = (
                    total_debt / equity
                    if not (np.isnan(total_debt) or np.isnan(equity) or equity == 0)
                    else float("nan")
                )

                # ROE
                if not inc.empty and "Net Income" in inc.index:
                    net_income = _safe_float(inc.loc["Net Income", date])
                    row["roe"] = (
                        net_income / equity
                        if not (np.isnan(equity) or equity == 0 or np.isnan(net_income))
                        else float("nan")
                    )
                else:
                    row["roe"] = float("nan")

                rows.append(row)

            if not rows:
                continue

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df.to_parquet(cache_path, index=False)
            frames.append(df)

        except Exception as exc:
            print(f"[fundamentals] ERROR fetching {ticker}: {exc}")
            continue

    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "revenue", "eps", "gross_margin", "debt_equity", "roe"]
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


def load_fundamentals(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Load previously cached fundamentals from Parquet files."""
    tickers = tickers or DEFAULT_UNIVERSE
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        cache_path = FUNDAMENTALS_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "revenue", "eps", "gross_margin", "debt_equity", "roe"]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined
