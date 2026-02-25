"""
sentiment.py
------------
Convert news / sentiment data into a normalised score in [0, 1].

In Phase 1 the sentiment data is a neutral stub (0.5 for all tickers).
This module is ready to accept real sentiment scores once a news API is
integrated in Phase 2.
"""

from __future__ import annotations

import pandas as pd


def compute_sentiment_score(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge sentiment scores onto the daily market DataFrame.

    Parameters
    ----------
    market_df : daily market DataFrame (date, ticker, ...)
    news_df   : DataFrame with columns [ticker, sentiment_score]
                (one row per ticker, latest score)

    Returns
    -------
    market_df with ``sentiment_score`` column added.
    """
    df = market_df.copy()

    if "sentiment_score" in news_df.columns and "ticker" in news_df.columns:
        score_map = news_df.set_index("ticker")["sentiment_score"].to_dict()
        df["sentiment_score"] = df["ticker"].map(score_map).fillna(0.5)
    else:
        df["sentiment_score"] = 0.5

    return df
