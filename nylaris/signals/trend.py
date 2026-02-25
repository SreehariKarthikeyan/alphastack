"""
trend.py
--------
Trend signal engineering.

Signals:
  1. price_vs_200dma  – close / 200-day SMA  (normalised 0-1)
  2. ma_crossover     – 50 DMA / 200 DMA     (normalised 0-1)
  3. atr_regime       – ATR-based volatility regime (low vol → higher score)

Each function accepts a DataFrame with at minimum columns
[date, ticker, open, high, low, close, volume] and returns the same
DataFrame with the new signal column(s) appended.

``compute_trend_score`` is the main entry point; it returns a per-ticker,
per-date score in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_max_norm(series: pd.Series, clip_quantile: float = 0.01) -> pd.Series:
    """
    Min-max normalise *series* to [0, 1].
    Winsorise at *clip_quantile* to reduce outlier distortion.
    """
    lo = series.quantile(clip_quantile)
    hi = series.quantile(1 - clip_quantile)
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    clipped = series.clip(lo, hi)
    return (clipped - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------

def price_vs_200dma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add column ``price_vs_200dma``: ratio of close price to its 200-day SMA.

    Values > 1 mean the stock is above its 200 DMA (bullish).
    """
    df = df.copy()
    df["sma_200"] = (
        df.groupby("ticker")["close"]
        .transform(lambda s: s.rolling(200, min_periods=50).mean())
    )
    df["price_vs_200dma"] = df["close"] / df["sma_200"]
    return df


def ma_crossover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add column ``ma_crossover``: ratio of 50-day SMA to 200-day SMA.

    Values > 1 indicate a 'golden cross' (bullish); < 1 a 'death cross'.
    """
    df = df.copy()
    df["sma_50"] = (
        df.groupby("ticker")["close"]
        .transform(lambda s: s.rolling(50, min_periods=20).mean())
    )
    if "sma_200" not in df.columns:
        df["sma_200"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.rolling(200, min_periods=50).mean())
        )
    df["ma_crossover"] = df["sma_50"] / df["sma_200"]
    return df


def atr_regime(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add column ``atr``: Average True Range over *window* days.

    Lower ATR relative to price → lower volatility → more stable trend.
    """
    df = df.copy()

    def _calc_atr(grp: pd.DataFrame) -> pd.Series:
        high = grp["high"]
        low = grp["low"]
        prev_close = grp["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window, min_periods=window // 2).mean()

    df["atr"] = df.groupby("ticker", group_keys=False).apply(_calc_atr)
    # Normalise ATR as a fraction of price (ATR%)
    df["atr_pct"] = df["atr"] / df["close"]
    return df


# ---------------------------------------------------------------------------
# Composite trend score
# ---------------------------------------------------------------------------

def compute_trend_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite trend score in [0, 1] for each (ticker, date) row.

    Adds columns:
      sma_50, sma_200, price_vs_200dma, ma_crossover, atr, atr_pct,
      trend_score

    Parameters
    ----------
    df : DataFrame with columns [date, ticker, open, high, low, close, volume]

    Returns
    -------
    DataFrame with all original columns plus the new signal columns.
    """
    df = price_vs_200dma(df)
    df = ma_crossover(df)
    df = atr_regime(df)

    # Normalise each raw signal across the *entire* dataset so scores are
    # comparable across tickers on any given day.
    p200_norm = _min_max_norm(df["price_vs_200dma"].fillna(1.0))
    mac_norm = _min_max_norm(df["ma_crossover"].fillna(1.0))

    # Low volatility → high score: invert the ATR percentage
    atr_norm = 1.0 - _min_max_norm(df["atr_pct"].fillna(df["atr_pct"].median()))

    df["trend_score"] = (
        0.40 * p200_norm
        + 0.40 * mac_norm
        + 0.20 * atr_norm
    ).clip(0, 1)

    return df
