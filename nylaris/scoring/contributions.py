"""
contributions.py
-----------------
Feature-contribution breakdown for the composite score.

For each ticker, return the exact contribution of every signal component
so the score is fully transparent and auditable.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

# Weight definitions — must stay in sync with composite.py
WEIGHTS_PHASE1 = {
    "trend": 0.50,
    "momentum": 0.50,
    "fundamental": 0.00,
    "sentiment": 0.00,
    "risk": 0.00,
}

WEIGHTS_FULL = {
    "trend": 0.25,
    "momentum": 0.20,
    "fundamental": 0.35,
    "sentiment": 0.10,
    "risk": 0.10,
}


def _score_col(name: str) -> str:
    """Map short name → DataFrame column name."""
    return f"{name}_score" if name != "risk" else "risk_score"


def compute_contributions(
    row: pd.Series,
    mode: str = "phase1",
) -> Dict[str, float]:
    """
    Return a dict of signal contributions for a single ticker snapshot.

    Parameters
    ----------
    row  : Series/dict with keys trend_score, momentum_score,
           fundamental_score, sentiment_score, and optionally
           volatility_regime or atr_pct.
    mode : "phase1" or "full"

    Returns
    -------
    Dict with keys like ``trend_contribution``, ``momentum_contribution``, …
    """
    weights = WEIGHTS_PHASE1 if mode == "phase1" else WEIGHTS_FULL

    trend = float(row.get("trend_score", 0.5))
    momentum = float(row.get("momentum_score", 0.5))
    fundamental = float(row.get("fundamental_score", 0.5))
    sentiment = float(row.get("sentiment_score", 0.5))

    # Risk adjustment uses volatility_regime label → numeric
    regime = row.get("volatility_regime", "medium")
    if isinstance(regime, str):
        risk_score = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(regime, 0.5)
    else:
        risk_score = 1.0 - float(regime)  # numeric penalty → invert

    contributions = {
        "trend_contribution": weights["trend"] * trend,
        "momentum_contribution": weights["momentum"] * momentum,
        "fundamental_contribution": weights["fundamental"] * fundamental,
        "sentiment_contribution": weights["sentiment"] * sentiment,
        "risk_adjustment": weights["risk"] * risk_score,
    }
    contributions["total"] = sum(contributions.values())
    return contributions


def compute_contributions_table(
    snapshot: pd.DataFrame,
    mode: str = "phase1",
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per ticker and columns for each
    signal's contribution to the composite score.

    Parameters
    ----------
    snapshot : Latest-date snapshot (from ``build_snapshot``).
    mode     : Scoring mode.

    Returns
    -------
    DataFrame indexed by ticker with contribution columns.
    """
    records: List[Dict[str, float | str]] = []

    for _, row in snapshot.iterrows():
        contrib = compute_contributions(row, mode=mode)
        contrib["ticker"] = row["ticker"]
        records.append(contrib)

    df = pd.DataFrame(records)
    cols = ["ticker", "trend_contribution", "momentum_contribution",
            "fundamental_contribution", "sentiment_contribution",
            "risk_adjustment", "total"]
    return df[[c for c in cols if c in df.columns]]
