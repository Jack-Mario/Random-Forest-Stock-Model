import yfinance as yf
import pandas as pd
from pathlib import Path
from .features import create_features
from typing import List
import datetime


def sanitize_name(ticker: str) -> str:
    return ticker.replace('.', '_').replace('-', '_')


def fetch_and_save(tickers: List[str], out_dir: str = 'data', start_date: datetime.datetime = None, end_date: datetime.datetime = None, period: int = 60, alfa: float = 0.1):
    """Fetch OHLCV for tickers, create features and save CSV files per ticker.

    - tickers: list of full tickers (e.g. 'VOLV-B.ST')
    - saves to out_dir/data_{sanitized_ticker}.csv
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if start_date is None:
        start_date = datetime.datetime(2014, 1, 1)
    if end_date is None:
        end_date = datetime.datetime.today()

    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, end=end_date)
            if df.empty:
                print(f"  No data for {ticker}")
                continue
            
            # Remove timezone info completely
            df.index = df.index.tz_localize(None)
            
            df = create_features(df, period=period, alfa=alfa)
            df = df.dropna()
            
            # Save Close prices separately
            close_df = pd.DataFrame({'Close': df['Close']})
            close_fname = out / f"prices_{sanitize_name(ticker)}.csv"
            close_df.to_csv(close_fname, index=True)
            
            # Keep the final feature vector + label (no Close)
            cols_to_keep = ['RSI', 'K(14)', 'R(14)', 'MACD', 'Signal', 'PROC', 'OBV', 'Increase']
            df = df[[c for c in cols_to_keep if c in df.columns]]
            
            # Keep index as datetime string format (don't convert to .date)
            # Pandas will naturally save datetime index as ISO string
            fname = out / f"data_{sanitize_name(ticker)}.csv"
            df.to_csv(fname, index=True)
            print(f"  Saved {fname} ({len(df)} rows) + prices file")
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
