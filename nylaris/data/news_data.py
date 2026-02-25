"""
news_data.py
------------
Stub for news / sentiment data ingestion.

Phase 1 uses a neutral placeholder score (0.5) for every ticker so the
composite scoring pipeline can proceed without an external news API key.
A real implementation would query a financial news API (e.g. NewsAPI,
Benzinga, Alpha Vantage News) and compute an aggregated sentiment score.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .market_data import DEFAULT_UNIVERSE


def fetch_news_sentiment(
    tickers: Optional[List[str]] = None,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns [ticker, sentiment_score].

    In Phase 1 every ticker receives a neutral score of 0.5.
    Replace this function body with a real API call in Phase 2.

    Parameters
    ----------
    tickers:
        List of ticker symbols.  Defaults to the standard universe.
    lookback_days:
        Number of calendar days to look back for news (reserved for
        Phase 2 implementation).
    """
    tickers = tickers or DEFAULT_UNIVERSE

    rows = [{"ticker": t, "sentiment_score": 0.5} for t in tickers]
    df = pd.DataFrame(rows)
    return df
