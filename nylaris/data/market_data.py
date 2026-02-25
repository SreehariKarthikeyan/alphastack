"""
market_data.py
--------------
Pull 5 years of daily OHLCV data for the universe of stocks using yfinance
and persist the data locally as Parquet files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Default universe – 15 US large-cap stocks
# ---------------------------------------------------------------------------
DEFAULT_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "UNH", "XOM", "PG", "HD",
]

DATA_DIR = Path(os.environ.get("NYLARIS_DATA_DIR", "nylaris_data"))
MARKET_DIR = DATA_DIR / "market"


def _ensure_dirs() -> None:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ohlcv(
    tickers: Optional[List[str]] = None,
    period: str = "5y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download OHLCV data for *tickers* and return a combined DataFrame.

    Columns: date, ticker, open, high, low, close, volume

    Data is cached to Parquet files under MARKET_DIR.  Pass
    ``force_refresh=True`` to bypass the cache.
    """
    _ensure_dirs()
    tickers = tickers or DEFAULT_UNIVERSE

    frames: List[pd.DataFrame] = []

    for ticker in tickers:
        cache_path = MARKET_DIR / f"{ticker}.parquet"

        if cache_path.exists() and not force_refresh:
            df = pd.read_parquet(cache_path)
        else:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                print(f"[market_data] WARNING: no data returned for {ticker}")
                continue

            # Flatten multi-level columns produced by yfinance ≥0.2
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            raw = raw[["open", "high", "low", "close", "volume"]].copy()
            raw.index.name = "date"
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            raw["ticker"] = ticker

            raw.to_parquet(cache_path)
            df = raw

        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )

    combined = pd.concat(frames)
    combined = combined.reset_index().rename(columns={"index": "date"})
    if "date" not in combined.columns and combined.index.name == "date":
        combined = combined.reset_index()

    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


def load_ohlcv(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Load previously cached OHLCV data from Parquet files."""
    tickers = tickers or DEFAULT_UNIVERSE
    frames: List[pd.DataFrame] = []
    for ticker in tickers:
        cache_path = MARKET_DIR / f"{ticker}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            df = df.reset_index().rename(columns={"index": "date"})
            if "date" not in df.columns and df.index.name == "date":
                df = df.reset_index()
            df["ticker"] = ticker
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )
    combined = pd.concat(frames)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined
