"""
history.py
----------
Temporal score tracking — persist daily snapshots to SQLite and compute
score deltas over configurable lookback windows.

Schema
------
  date              TEXT  (ISO-8601)
  ticker            TEXT
  trend_score       REAL
  momentum_score    REAL
  fundamental_score REAL
  sentiment_score   REAL
  composite_score   REAL
  volatility_regime TEXT
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path(os.environ.get("NYLARIS_DATA_DIR", "nylaris_data"))
DB_PATH = DATA_DIR / "score_history.db"

_SCORE_COLS = [
    "trend_score",
    "momentum_score",
    "fundamental_score",
    "sentiment_score",
    "composite_score",
]

_TABLE_COLS = _SCORE_COLS + ["volatility_regime"]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_db() -> sqlite3.Connection:
    """Create the database and table if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            date              TEXT    NOT NULL,
            ticker            TEXT    NOT NULL,
            trend_score       REAL,
            momentum_score    REAL,
            fundamental_score REAL,
            sentiment_score   REAL,
            composite_score   REAL,
            volatility_regime TEXT,
            PRIMARY KEY (date, ticker)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_date
        ON score_history (ticker, date)
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def save_daily_snapshot(df: pd.DataFrame) -> int:
    """
    Persist the latest-date scores for each ticker into the SQLite database.

    Parameters
    ----------
    df : Scored DataFrame with columns [date, ticker, ...score columns...]

    Returns
    -------
    Number of rows upserted.
    """
    conn = _ensure_db()

    # Take the latest date per ticker
    latest = df.sort_values("date").groupby("ticker").last().reset_index()
    latest["date"] = pd.to_datetime(latest["date"]).dt.strftime("%Y-%m-%d")

    rows_written = 0
    for _, row in latest.iterrows():
        values = {
            "date": row["date"],
            "ticker": row["ticker"],
        }
        for col in _TABLE_COLS:
            if col not in row.index or pd.isna(row[col]):
                values[col] = None
            elif col == "volatility_regime":
                values[col] = str(row[col])
            else:
                values[col] = float(row[col])

        conn.execute(
            """
            INSERT OR REPLACE INTO score_history
                (date, ticker, trend_score, momentum_score,
                 fundamental_score, sentiment_score, composite_score,
                 volatility_regime)
            VALUES
                (:date, :ticker, :trend_score, :momentum_score,
                 :fundamental_score, :sentiment_score, :composite_score,
                 :volatility_regime)
            """,
            values,
        )
        rows_written += 1

    conn.commit()
    conn.close()
    return rows_written


def save_full_history(df: pd.DataFrame) -> int:
    """
    Persist *all* date×ticker rows (not just latest).  Useful for
    back-filling history from a full pipeline run.

    Returns
    -------
    Number of rows upserted.
    """
    conn = _ensure_db()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    rows_written = 0
    for _, row in df.iterrows():
        values = {
            "date": row["date"],
            "ticker": row["ticker"],
        }
        for col in _TABLE_COLS:
            if col not in row.index or pd.isna(row[col]):
                values[col] = None
            elif col == "volatility_regime":
                values[col] = str(row[col])
            else:
                values[col] = float(row[col])

        conn.execute(
            """
            INSERT OR REPLACE INTO score_history
                (date, ticker, trend_score, momentum_score,
                 fundamental_score, sentiment_score, composite_score,
                 volatility_regime)
            VALUES
                (:date, :ticker, :trend_score, :momentum_score,
                 :fundamental_score, :sentiment_score, :composite_score,
                 :volatility_regime)
            """,
            values,
        )
        rows_written += 1

    conn.commit()
    conn.close()
    return rows_written


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load_history(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load score history from the database.

    Returns a DataFrame with the same schema as the table.
    """
    conn = _ensure_db()

    query = "SELECT * FROM score_history WHERE 1=1"
    params: list = []

    if tickers:
        placeholders = ",".join("?" for _ in tickers)
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY ticker, date"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Score deltas
# ---------------------------------------------------------------------------

def compute_score_change(ticker: str, days: int = 7) -> Optional[float]:
    """
    Compute the composite score change for *ticker* over the last *days*
    calendar days.

    Returns
    -------
    float or None if insufficient history.
    """
    conn = _ensure_db()

    rows = pd.read_sql_query(
        """
        SELECT date, composite_score
        FROM score_history
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        conn,
        params=[ticker, days + 30],  # fetch extra rows to cover trading days
    )
    conn.close()

    if rows.empty or len(rows) < 2:
        return None

    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows.sort_values("date")

    latest_score = rows.iloc[-1]["composite_score"]
    cutoff = rows.iloc[-1]["date"] - pd.Timedelta(days=days)
    older = rows[rows["date"] <= cutoff]

    if older.empty:
        return None

    past_score = older.iloc[-1]["composite_score"]
    return latest_score - past_score


def compute_score_deltas(
    snapshot: pd.DataFrame,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """
    Add score_change_Nd columns to a snapshot DataFrame.

    Parameters
    ----------
    snapshot : DataFrame with at least a ``ticker`` column.
    windows  : List of lookback windows in days (default [7, 30]).

    Returns
    -------
    snapshot with additional columns ``score_change_7d``, ``score_change_30d``, etc.
    """
    windows = windows or [7, 30]
    snapshot = snapshot.copy()

    for d in windows:
        col_name = f"score_change_{d}d"
        snapshot[col_name] = snapshot["ticker"].apply(
            lambda t, _d=d: compute_score_change(t, days=_d)
        )

    return snapshot
