"""
composite.py
------------
Combine individual signal scores into a single composite score.

Phase 1 formula (no fundamentals / sentiment yet):
    composite_score = 0.5 * trend_score + 0.5 * momentum_score

Full formula (activated when all scores are available):
    composite_score = 0.35 * fundamental_score
                    + 0.25 * trend_score
                    + 0.20 * momentum_score
                    + 0.10 * sentiment_score
                    + 0.10 * (1 - volatility_penalty)

Each call stores a daily snapshot JSON-serialisable dict and optionally
persists a Parquet file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("NYLARIS_DATA_DIR", "nylaris_data"))
SCORING_DIR = DATA_DIR / "scoring"


def _ensure_dirs() -> None:
    SCORING_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Volatility penalty
# ---------------------------------------------------------------------------

def _volatility_regime(df: pd.DataFrame) -> pd.Series:
    """
    Return a volatility penalty in [0, 1] derived from ``atr_pct`` if present,
    otherwise return 0 (no penalty).
    """
    if "atr_pct" not in df.columns:
        return pd.Series(0.0, index=df.index)
    atr = df["atr_pct"].fillna(df["atr_pct"].median())
    lo, hi = atr.quantile(0.01), atr.quantile(0.99)
    if hi == lo:
        return pd.Series(0.5, index=df.index)
    return ((atr.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_composite_score(df: pd.DataFrame, mode: str = "phase1") -> pd.DataFrame:
    """
    Compute composite scores and return the DataFrame with new columns.

    Parameters
    ----------
    df : DataFrame containing per-date, per-ticker signal scores.
         Expected columns vary by *mode*:
           phase1 : trend_score, momentum_score
           full   : trend_score, momentum_score, fundamental_score,
                    sentiment_score, atr_pct
    mode : "phase1" or "full"

    Returns
    -------
    Input DataFrame plus columns:
      volatility_regime, composite_score
    """
    df = df.copy()

    trend = df.get("trend_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    momentum = df.get("momentum_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    fundamentals = df.get("fundamental_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    sentiment = df.get("sentiment_score", pd.Series(0.5, index=df.index)).fillna(0.5)
    vol_penalty = _volatility_regime(df)

    df["volatility_regime"] = vol_penalty.apply(
        lambda v: "high" if v > 0.66 else ("medium" if v > 0.33 else "low")
    )

    if mode == "phase1":
        df["composite_score"] = (0.5 * trend + 0.5 * momentum).clip(0, 1)
    else:
        df["composite_score"] = (
            0.35 * fundamentals
            + 0.25 * trend
            + 0.20 * momentum
            + 0.10 * sentiment
            + 0.10 * (1 - vol_penalty)
        ).clip(0, 1)

    return df


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def build_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the latest-date snapshot for each ticker as a tidy DataFrame
    matching the schema:

      ticker, trend_score, momentum_score, fundamental_score,
      sentiment_score, volatility_regime, earnings_days, composite_score
    """
    latest = df.sort_values("date").groupby("ticker").last().reset_index()

    cols = [
        "ticker",
        "trend_score",
        "momentum_score",
        "fundamental_score",
        "sentiment_score",
        "volatility_regime",
        "composite_score",
    ]
    # Add placeholders for missing columns
    for col in cols:
        if col not in latest.columns:
            latest[col] = 0.5 if col not in ("ticker", "volatility_regime") else "n/a"

    # earnings_days is a placeholder (would require earnings calendar API)
    latest["earnings_days"] = None

    return latest[cols + ["earnings_days"]].sort_values("composite_score", ascending=False).reset_index(drop=True)


def save_snapshot(snapshot: pd.DataFrame, date_label: Optional[str] = None) -> Path:
    """Persist the snapshot DataFrame to Parquet and return the file path."""
    _ensure_dirs()
    label = date_label or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = SCORING_DIR / f"snapshot_{label}.parquet"
    snapshot.to_parquet(path, index=False)
    return path
