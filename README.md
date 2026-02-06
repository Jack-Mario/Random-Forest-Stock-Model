# Multi-stock ML backtest pipeline

This repository contains a modular pipeline for fetching stock data, computing features, training Random Forest models per stock, and running backtests.

Quick start
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Fetch data (edits `run_fetch.py` to change tickers):
```bash
python run_fetch.py
```
3. Train models (parallel):
```bash
python run_train.py
```
4. Run backtests and generate plots:
```bash
python run_backtest.py
```

Files
- `src/features.py` - feature engineering
- `src/data_fetch.py` - download & save parquet per ticker
- `src/model.py` - train model and save predictions
- `src/backtest.py` - backtest from saved predictions
- `src/plot.py` - plotting helpers
- `run_fetch.py`, `run_train.py`, `run_backtest.py` - runner scripts

Data & outputs
- `data/` - parquet per ticker
- `models/` - saved joblib models
- `results/` - predictions and plots

If you want, I can now run a small smoke-check (no external network here) or adjust parallel settings.
