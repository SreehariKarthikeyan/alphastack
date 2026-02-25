"""
diagnostics.py
--------------
Distribution diagnostics for the signal engine.

Computes:
  - Mean, standard deviation, skewness of composite scores
  - Correlation matrix between features
  - Feature dominance checks (any single feature >70% correlated with composite)
  - Redundancy detection (pairwise feature correlation >0.85)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


_SIGNAL_COLS = [
    "trend_score",
    "momentum_score",
    "fundamental_score",
    "sentiment_score",
    "composite_score",
]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_score_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute distributional statistics for the composite score
    across the latest-date snapshot.

    Returns
    -------
    Dict with keys: mean, std, skewness, min, max, median.
    """
    scores = df["composite_score"].dropna()
    if scores.empty:
        return {k: float("nan") for k in
                ("mean", "std", "skewness", "min", "max", "median")}

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "skewness": float(scores.skew()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "median": float(scores.median()),
    }


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix between all signal columns
    present in *df*.

    Returns
    -------
    Square DataFrame of correlations.
    """
    available = [c for c in _SIGNAL_COLS if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return df[available].corr()


# ---------------------------------------------------------------------------
# Redundancy & dominance checks
# ---------------------------------------------------------------------------

def detect_redundancy(
    corr: pd.DataFrame,
    threshold: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """
    Identify pairs of features whose absolute correlation exceeds *threshold*.

    Returns a list of (feature_a, feature_b, correlation) triples.
    """
    pairs: List[Tuple[str, str, float]] = []
    cols = [c for c in corr.columns if c != "composite_score"]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append((a, b, float(r)))
    return pairs


def detect_dominance(
    corr: pd.DataFrame,
    threshold: float = 0.70,
) -> List[Tuple[str, float]]:
    """
    Identify features whose absolute correlation with composite_score
    exceeds *threshold* — possibly dominating the composite.

    Returns a list of (feature, correlation) pairs.
    """
    if "composite_score" not in corr.columns:
        return []

    dominant: List[Tuple[str, float]] = []
    for col in corr.columns:
        if col == "composite_score":
            continue
        r = corr.loc[col, "composite_score"]
        if abs(r) >= threshold:
            dominant.append((col, float(r)))
    return dominant


# ---------------------------------------------------------------------------
# Full diagnostic report
# ---------------------------------------------------------------------------

def run_diagnostics(df: pd.DataFrame) -> Dict:
    """
    Run all distribution diagnostics on a scored DataFrame.

    Parameters
    ----------
    df : DataFrame with per-ticker, per-date signal scores.

    Returns
    -------
    Dict with keys:
      statistics   – distributional stats of composite score
      correlation  – correlation matrix (as dict of dicts)
      redundancy   – list of redundant feature pairs
      dominance    – list of dominant features
    """
    # Use the latest-date snapshot per ticker for cross-sectional analysis
    latest = df.sort_values("date").groupby("ticker").last().reset_index()

    stats = compute_score_statistics(latest)
    corr = compute_correlation_matrix(latest)

    result = {
        "statistics": stats,
        "correlation": corr.to_dict() if not corr.empty else {},
        "redundancy": detect_redundancy(corr) if not corr.empty else [],
        "dominance": detect_dominance(corr) if not corr.empty else [],
    }
    return result


def print_diagnostics(diag: Dict) -> None:
    """Pretty-print the diagnostic results."""
    stats = diag["statistics"]
    print("\n" + "=" * 55)
    print("  DISTRIBUTION DIAGNOSTICS")
    print("=" * 55)
    print(f"  Composite Score  mean     : {stats['mean']:.4f}")
    print(f"                   std      : {stats['std']:.4f}")
    print(f"                   skewness : {stats['skewness']:.4f}")
    print(f"                   min      : {stats['min']:.4f}")
    print(f"                   max      : {stats['max']:.4f}")
    print(f"                   median   : {stats['median']:.4f}")

    corr_data = diag["correlation"]
    if corr_data:
        print("\n  Feature Correlation Matrix:")
        corr_df = pd.DataFrame(corr_data)
        # Format to 2 decimal places
        for col in corr_df.columns:
            corr_df[col] = corr_df[col].apply(lambda v: f"{v:.2f}")
        print(corr_df.to_string(index=True))

    redundancy = diag["redundancy"]
    if redundancy:
        print("\n  ⚠ Redundant feature pairs (|r| ≥ 0.85):")
        for a, b, r in redundancy:
            print(f"    {a}  ↔  {b}  :  r = {r:.3f}")
    else:
        print("\n  ✓ No redundant feature pairs detected.")

    dominance = diag["dominance"]
    if dominance:
        print("\n  ⚠ Dominant features (|r| ≥ 0.70 with composite):")
        for feat, r in dominance:
            print(f"    {feat}  :  r = {r:.3f}")
    else:
        print("\n  ✓ No single feature dominates the composite.")

    print("=" * 55 + "\n")
