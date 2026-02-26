# alphastack — Nylaris Signal Engine

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/SreehariKarthikeyan/alphastack/main/streamlit_app.py)

A quantitative stock-scoring engine that ranks equities using trend, momentum, sentiment, and fundamental signals and exposes results via a web dashboard and a CLI.

---

## Local Setup

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 or higher |
| Git | any recent version |

### 1 — Clone the repository

```bash
git clone https://github.com/SreehariKarthikeyan/alphastack.git
cd alphastack
```

### 2 — Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the Web UI

```bash
streamlit run nylaris/ui.py
```

The dashboard opens at **http://localhost:8501** in your browser.

> **Demo mode is on by default** — synthetic data loads instantly so you can explore every chart and table without downloading anything.  
> To score real stocks, toggle off **🎲 Use demo data** in the sidebar and click **▶ Run Pipeline**.

### 5 — Run the CLI pipeline

Run each stage individually or the full pipeline in one command:

```bash
# Full pipeline: download data → compute signals → score → backtest → snapshot
python -m nylaris.main --run all

# Download market data only (cached to nylaris_data/market/)
python -m nylaris.main --run ingest

# Compute signals & composite scores (requires cached data)
python -m nylaris.main --run signals

# Run the rolling backtest (requires scored data)
python -m nylaris.main --run backtest

# Print today's ranked snapshot (requires scored data)
python -m nylaris.main --run snapshot
```

**Options**

| Flag | Description |
|------|-------------|
| `--tickers AAPL MSFT …` | Override the default 15-stock universe |
| `--refresh` | Force re-download of cached market data |
| `--mode phase1\|full` | `phase1` (trend + momentum) or `full` (adds fundamentals & sentiment) |

**Example — score three stocks in full mode:**
```bash
python -m nylaris.main --run all --tickers AAPL NVDA TSLA --mode full
```

---

## Project Structure

```
alphastack/
├── nylaris/
│   ├── main.py            # CLI entry point
│   ├── ui.py              # Streamlit web dashboard
│   ├── data/
│   │   ├── market_data.py # OHLCV download & cache (yfinance)
│   │   ├── news_data.py   # News sentiment fetch
│   │   └── fundamentals.py
│   ├── signals/
│   │   ├── trend.py       # SMA crossover, ATR regime
│   │   ├── momentum.py    # RSI, 3 m / 6 m returns
│   │   ├── sentiment.py   # Sentiment score aggregation
│   │   └── fundamentals.py
│   ├── scoring/
│   │   └── composite.py   # Weighted composite score
│   └── backtest/
│       └── engine.py      # Rolling monthly-rebalance backtest
├── streamlit_app.py       # Streamlit Cloud entry point
└── requirements.txt
```

> **Data cache** — downloaded market data is stored in `nylaris_data/` (git-ignored).  
> Delete this folder or use `--refresh` to force a fresh download.

---

## Free Hosting with Streamlit Community Cloud

GitHub does **not** host Python/Streamlit apps directly, but
[**Streamlit Community Cloud**](https://streamlit.io/cloud) is **completely free** and
deploys straight from your GitHub repository with no server management needed.

### Deploy in 3 steps

1. **Push to GitHub** — make sure your code is on the `main` branch of a public (or
   private) GitHub repository.

2. **Sign in to Streamlit Community Cloud** — go to
   [share.streamlit.io](https://share.streamlit.io) and click **"Sign in with GitHub"**.

3. **Create a new app** — click **"New app"** and fill in:

   | Field | Value |
   |-------|-------|
   | Repository | `<your-github-username>/alphastack` |
   | Branch | `main` |
   | Main file path | `streamlit_app.py` |

   Click **"Deploy!"** — your app will be live at a public URL within a minute.

### How it works

Streamlit Community Cloud reads `requirements.txt` to install dependencies and runs
`streamlit_app.py` as the entry point.  Every `git push` to `main` triggers an
automatic redeploy.

A `.streamlit/config.toml` file is included in this repo with the recommended cloud
settings (`headless = true`, CORS disabled).

### Continuous Integration (GitHub Actions)

A GitHub Actions workflow (`.github/workflows/ci.yml`) is included.  On every push or
pull request to `main` it installs dependencies, checks syntax, and verifies imports —
so broken code is caught before it reaches the live deployment.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'nylaris'` | Run commands from the repo root (`cd alphastack`) with the venv active |
| `streamlit: command not found` | Run `pip install -r requirements.txt` and make sure the venv is active |
| No data returned for a ticker | The ticker symbol may be delisted or misspelled; yfinance uses Yahoo Finance symbols |
| Slow first run | The pipeline downloads 5 years of daily OHLCV for each ticker on the first run; subsequent runs use the local cache |

