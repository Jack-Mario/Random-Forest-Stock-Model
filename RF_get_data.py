import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# List of major Swedish companies
stocks = {
    "VOLV-B.ST": "Volvo Group",
    "VOLCAR-B.ST": "Volvo Cars",
    "ERIC-B.ST": "Ericsson",
    "HM-B.ST": "H&M",
    "SAND.ST": "Sandvik",
    "SKA-B.ST": "Skanska",
    "AZN.ST": "AstraZeneca",
    "ICA.ST": "ICA Gruppen"
}

start_date = datetime.datetime(2014, 1, 1)
end_date = datetime.datetime.today()
alfa = 0.1
period = 60

def create_features(df):
    """Create technical indicators from OHLCV data"""
    # Increase and smooth data
    df['Increase'] = np.where(df['Close'].shift(-period) > df['Close'], 1, -1)
    df['Smooth'] = df['Close'].ewm(alpha=alfa, adjust=False).mean()
    df['Delta_Smooth'] = df['Smooth'].diff()
    
    # RSI
    df['RSI_gain'] = df['Delta_Smooth'].clip(lower=0)
    df['RSI_loss'] = (-df['Delta_Smooth']).clip(lower=0)
    df['RSI_gain_mean'] = df['RSI_gain'].rolling(14).mean()
    df['RSI_loss_mean'] = df['RSI_loss'].rolling(14).mean()
    df['RSI'] = 100 - 100 / (1 + df['RSI_gain_mean'].div(df['RSI_loss_mean']))
    
    # Stochastic Oscillator
    df['L(14)'] = df['Low'].rolling(14).min()
    df['H(14)'] = df['High'].rolling(14).max()
    df['K(14)'] = 100 * ((df['Smooth'] - df['L(14)']).div(df['H(14)'] - df['L(14)']))
    
    # Williams
    df['R(14)'] = -100 * ((df['H(14)'] - df['Smooth']).div(df['H(14)'] - df['L(14)']))
    
    # EMA
    df['Smooth12'] = df['Smooth'].ewm(span=12, adjust=False).mean()
    df['Smooth26'] = df['Smooth'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['Smooth12'] - df['Smooth26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Price Rate of Change - PROC 10
    df['PROC'] = (df['Smooth'] - df['Smooth'].shift(10)) / df['Smooth'].shift(10)
    
    # On-Balance Volume - OBV
    df['OBV'] = (
        np.sign(df['Smooth'].diff())
        .fillna(0)
        .mul(df['Volume'])
        .cumsum()
    )
    
    return df

# Download and process data for each stock
for ticker, company_name in stocks.items():
    try:
        print(f"Downloading data for {company_name} ({ticker})...")
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if len(df) == 0:
            print(f"  No data found for {ticker}")
            continue
        
        # Create features
        df = create_features(df)
        
        # Drop NaN rows
        df = df.dropna()
        
        # Save to CSV
        filename = f"data_{ticker.replace('-', '_').replace('.', '_')}.csv"
        df.to_csv(filename, index=True)
        print(f"  Saved {len(df)} rows to {filename}")
        
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")

print("\nDone! All stock data has been downloaded and processed.")