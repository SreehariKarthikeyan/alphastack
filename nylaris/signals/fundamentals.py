"""
fundamentals.py  (signals layer)
---------------------------------
Convert raw fundamental data into a normalised score in [0, 1].

The fundamental score rewards:
  - High revenue growth (QoQ)
  - High EPS
  - High gross margin
  - Low debt/equity
  - High ROE
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _min_max_norm(series: pd.Series, clip_quantile: float = 0.01) -> pd.Series:
    lo = series.quantile(clip_quantile)
    hi = series.quantile(1 - clip_quantile)
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    clipped = series.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


def compute_fundamental_score(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a fundamental score in [0, 1] for each (ticker, date) row.

    Parameters
    ----------
    fund_df : DataFrame with columns
              [date, ticker, revenue, eps, gross_margin, debt_equity, roe]

    Returns
    -------
    DataFrame with all input columns plus ``fundamental_score``.
    """
    df = fund_df.copy()

    # Revenue growth QoQ
    df = df.sort_values(["ticker", "date"])
    df["revenue_growth"] = df.groupby("ticker")["revenue"].pct_change()

    # Normalise individual metrics
    rev_norm = _min_max_norm(df["revenue_growth"].fillna(0))
    eps_norm = _min_max_norm(df["eps"].fillna(df["eps"].median()))
    gm_norm = _min_max_norm(df["gross_margin"].fillna(df["gross_margin"].median()))

    # Low debt/equity is better → invert after normalising
    de_filled = df["debt_equity"].fillna(df["debt_equity"].median())
    de_norm = 1.0 - _min_max_norm(de_filled)

    roe_norm = _min_max_norm(df["roe"].fillna(df["roe"].median()))

    df["fundamental_score"] = (
        0.20 * rev_norm
        + 0.20 * eps_norm
        + 0.25 * gm_norm
        + 0.15 * de_norm
        + 0.20 * roe_norm
    ).clip(0, 1)

    return df


def align_fundamentals_to_market(
    market_df: pd.DataFrame,
    fund_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill quarterly fundamental scores onto the daily market data.

    Merges on (ticker, date) using an asof join so each market row gets the
    most recently available fundamental score.

    Parameters
    ----------
    market_df : daily market DataFrame (date, ticker, ...)
    fund_df   : quarterly fundamental DataFrame with ``fundamental_score`` column

    Returns
    -------
    market_df with ``fundamental_score`` column added.
    """
    market_df = market_df.copy().sort_values(["ticker", "date"])
    fund_df = fund_df.copy().sort_values(["ticker", "date"])

    result_frames = []
    for ticker, mkt_grp in market_df.groupby("ticker"):
        fund_grp = fund_df[fund_df["ticker"] == ticker][["date", "fundamental_score"]]
        if fund_grp.empty:
            mkt_grp = mkt_grp.copy()
            mkt_grp["fundamental_score"] = 0.5
        else:
            mkt_grp = pd.merge_asof(
                mkt_grp.reset_index(drop=True),
                fund_grp.reset_index(drop=True),
                on="date",
                direction="backward",
            )
            mkt_grp["fundamental_score"] = mkt_grp["fundamental_score"].fillna(0.5)
        result_frames.append(mkt_grp)

    return pd.concat(result_frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
