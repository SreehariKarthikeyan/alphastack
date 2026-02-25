"""
momentum.py
-----------
Momentum signal engineering.

Signals:
  1. rsi_14      – 14-period RSI
  2. return_3m   – 3-month (≈63 trading days) price return
  3. return_6m   – 6-month (≈126 trading days) price return

``compute_momentum_score`` is the main entry point; it returns a per-ticker,
per-date score in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_max_norm(series: pd.Series, clip_quantile: float = 0.01) -> pd.Series:
    lo = series.quantile(clip_quantile)
    hi = series.quantile(1 - clip_quantile)
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    clipped = series.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------

def rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add column ``rsi_14``: Wilder's RSI over *window* periods.

    RSI is already bounded in [0, 100]; we scale to [0, 1].
    """
    df = df.copy()

    def _calc_rsi(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    df[f"rsi_{window}"] = df.groupby("ticker")["close"].transform(_calc_rsi)
    return df


def return_nm(df: pd.DataFrame, trading_days: int = 63, label: str = "return_3m") -> pd.DataFrame:
    """
    Add a column *label* representing the price return over *trading_days*.

    Example: trading_days=63 → approximately 3 months.
    """
    df = df.copy()
    df[label] = df.groupby("ticker")["close"].transform(
        lambda s: s.pct_change(trading_days)
    )
    return df


# ---------------------------------------------------------------------------
# Composite momentum score
# ---------------------------------------------------------------------------

def compute_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite momentum score in [0, 1] for each (ticker, date) row.

    Adds columns:
      rsi_14, return_3m, return_6m, momentum_score

    Parameters
    ----------
    df : DataFrame with columns [date, ticker, open, high, low, close, volume]

    Returns
    -------
    DataFrame with all original columns plus the new signal columns.
    """
    df = rsi(df, window=14)
    df = return_nm(df, trading_days=63, label="return_3m")
    df = return_nm(df, trading_days=126, label="return_6m")

    # RSI is already in [0, 100]; normalise to [0, 1]
    rsi_norm = (df["rsi_14"].fillna(50) / 100).clip(0, 1)

    # Normalise return signals across the full dataset
    r3m_norm = _min_max_norm(df["return_3m"].fillna(0))
    r6m_norm = _min_max_norm(df["return_6m"].fillna(0))

    df["momentum_score"] = (
        0.40 * rsi_norm
        + 0.35 * r3m_norm
        + 0.25 * r6m_norm
    ).clip(0, 1)

    return df
