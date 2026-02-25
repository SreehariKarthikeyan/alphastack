# alphastack

## Nylaris Signal Engine

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

### Web UI

Launch the Streamlit dashboard to explore scores and stock details interactively:

```bash
pip install -r requirements.txt
streamlit run nylaris/ui.py
```

The UI lets you:
- Select tickers and scoring mode from the sidebar
- Click **▶ Run Pipeline** to fetch data and compute scores
- View a ranked snapshot table sorted by composite score
- Drill into any ticker to see price history, SMA overlays, signal scores, and OHLCV data
