# alphastack

## Nylaris Signal Engine

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/SreehariKarthikeyan/alphastack/main/streamlit_app.py)

### Web UI

Launch the Streamlit dashboard to explore scores and stock details interactively:

```bash
pip install -r requirements.txt
streamlit run nylaris/ui.py
```

The UI opens in **demo mode** by default — it loads synthetic data instantly so you can see all the charts and tables without any setup.  Toggle off **Use demo data** in the sidebar to run the real pipeline against Yahoo Finance.

The UI lets you:
- View a ranked snapshot table of all tickers sorted by composite score
- Drill into any ticker to see price history with SMA overlays, signal scores, OHLCV data, and composite score over time
- Select tickers and scoring mode (phase1 / full) from the sidebar
- Click **▶ Run Pipeline** (demo mode off) to fetch and score real market data

### CLI Usage

```bash
pip install -r requirements.txt

# Full pipeline: ingest → signals → score → backtest
python -m nylaris.main --run all

# Ingest market data only
python -m nylaris.main --run ingest

# Compute signals & scores (requires cached market data)
python -m nylaris.main --run signals

# Print today's ranked snapshot
python -m nylaris.main --run snapshot
```
