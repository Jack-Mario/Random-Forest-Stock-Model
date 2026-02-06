"""
Live Trading System - Main Program

This is the complete real trading program:
1. Reads actual portfolio from trades.xlsx
2. Saves to trade_log.json for backup
3. Gets latest market data
4. Generates BUY/SELL signals based on current positions
5. Plots portfolio performance vs index from first trade
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance not installed. Run: pip install yfinance")
    exit(1)

# Configuration
TICKERS = [
    'ABB.ST', 'ADDT-B.ST', 'ALFA.ST', 'ASSA-B.ST', 'AZN.ST', 'ATCO-A.ST', 'BOL.ST',
    'EPI-A.ST', 'EQT.ST', 'ERIC-B.ST', 'ESSITY-B.ST', 'EVO.ST', 'SHB-A.ST', 'HM-B.ST',
    'HEXA-B.ST', 'INDU-C.ST', 'INVE-B.ST', 'LIFCO-B.ST', 'NIBE-B.ST', 'NDA-SE.ST',
    'SAAB-B.ST', 'SAND.ST', 'SCA-B.ST', 'SEB-A.ST', 'SKA-B.ST', 'SKF-B.ST', 'SWED-A.ST',
    'TEL2-B.ST', 'TELIA.ST', 'VOLV-B.ST'
]

BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.48
MIN_HOLD_DAYS = 14

MODELS_DIR = Path('models')
TRADE_LOG_FILE = Path('trade_log.json')
TRADES_EXCEL = Path('trades.xlsx')


def load_portfolio_from_trades():
    """Load current portfolio from trades.xlsx"""
    # Try to read from Excel first
    if TRADES_EXCEL.exists():
        try:
            df = pd.read_excel(TRADES_EXCEL, sheet_name="Trades", skiprows=0)
            df.columns = df.columns.str.strip().str.lower()
            df = df.dropna(subset=['date', 'ticker', 'action'])
            
            cash = 10_000.0
            holdings = {}
            trades = []
            
            for idx, row in df.iterrows():
                ticker = str(row['ticker']).upper()
                action = str(row['action']).upper()
                shares = int(row['shares'])
                price = float(row['price (kr)'])
                
                trade = {
                    'date': str(row['date']),
                    'ticker': ticker,
                    'action': action,
                    'shares': shares,
                    'price': price
                }
                trades.append(trade)
                
                if action == 'BUY':
                    holdings[ticker] = holdings.get(ticker, 0) + shares
                    cash -= shares * price
                elif action == 'SELL':
                    holdings[ticker] = max(0, holdings.get(ticker, 0) - shares)
                    cash += shares * price
            
            # Remove zero holdings
            holdings = {t: s for t, s in holdings.items() if s > 0}
            
            return {
                'cash': cash,
                'holdings': holdings,
                'trades': trades
            }
        except Exception as e:
            print(f"  ⚠️  Error reading trades.xlsx: {e}")
    
    # Fallback to trade_log.json if Excel doesn't exist
    if TRADE_LOG_FILE.exists():
        with open(TRADE_LOG_FILE, 'r') as f:
            trades = json.load(f)
        
        cash = 10_000.0
        holdings = {}
        
        for trade in trades:
            ticker = trade['ticker']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            
            if action == 'BUY':
                holdings[ticker] = holdings.get(ticker, 0) + shares
                cash -= shares * price
            elif action == 'SELL':
                holdings[ticker] = max(0, holdings.get(ticker, 0) - shares)
                cash += shares * price
        
        holdings = {t: s for t, s in holdings.items() if s > 0}
        
        return {
            'cash': cash,
            'holdings': holdings,
            'trades': trades
        }
    
    print("⚠️  No trade log found. Starting fresh with 10,000 kr cash.")
    return {
        'cash': 10_000.0,
        'holdings': {},
        'trades': []
    }


def get_latest_data(days_back=90):
    """Download latest OHLCV data for all tickers"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📥 Downloading data from {start_date.date()} to {end_date.date()}...")
    
    all_data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                all_data[ticker] = df
        except Exception as e:
            print(f"  ⚠️  Error downloading {ticker}: {e}")
    
    print(f"✅ Downloaded data for {len(all_data)} tickers")
    return all_data


def calculate_features(df):
    """Calculate technical indicators for feature data"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic K%
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['K%'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    # Stochastic R%
    df['R%'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Price rate of change
    df['PROC'] = ((df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)) * 100
    
    # On-Balance Volume
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df


def generate_signals(portfolio, all_data, models_dir):
    """Generate BUY/SELL signals based on trained models"""
    
    print("\n" + "=" * 80)
    print("📊 SIGNAL GENERATION")
    print("=" * 80)
    
    signals = []
    
    for ticker in TICKERS:
        if ticker not in all_data:
            continue
        
        # Load model
        model_file = models_dir / f"data_{ticker.replace('.ST', '')}.joblib"
        if not model_file.exists():
            continue
        
        try:
            model_data = joblib.load(model_file)
            model = model_data['model']
            features = model_data['features']
        except Exception as e:
            print(f"  ⚠️  Error loading model for {ticker}: {e}")
            continue
        
        # Get latest data
        df = all_data[ticker]
        df_features = calculate_features(df).copy()
        
        # Get latest row
        latest = df_features.iloc[-1]
        
        # Check for NaN values
        feature_cols = ['RSI', 'K%', 'R%', 'MACD', 'Signal', 'PROC', 'OBV']
        if latest[feature_cols].isna().any():
            continue
        
        # Prepare features
        X = latest[feature_cols].values.reshape(1, -1)
        
        try:
            # Get probability of "up"
            prob = model.predict_proba(X)[0][1]  # Probability of class 1 (up)
            price = latest['Close']
            
            # Get current holdings
            shares = portfolio['holdings'].get(ticker, 0)
            
            # Generate signal
            signal = {
                'ticker': ticker,
                'price': price,
                'probability': prob,
                'shares': shares,
                'action': None,
                'reason': ''
            }
            
            # BUY signal
            if prob > BUY_THRESHOLD and shares == 0:
                signal['action'] = 'BUY'
                signal['reason'] = f"High probability ({prob:.1%}) of price increase"
            
            # SELL signal
            elif prob < SELL_THRESHOLD and shares > 0:
                signal['action'] = 'SELL'
                signal['reason'] = f"Low probability ({prob:.1%}) of price increase"
            
            # HOLD
            else:
                if shares > 0:
                    signal['action'] = 'HOLD'
                    signal['reason'] = f"Currently holding {shares} shares at {prob:.1%} probability"
                else:
                    signal['action'] = 'SKIP'
                    signal['reason'] = f"Probability {prob:.1%} not strong enough to trade"
            
            signals.append(signal)
        
        except Exception as e:
            print(f"  ⚠️  Error predicting {ticker}: {e}")
            continue
    
    return signals


def print_signals(signals, portfolio):
    """Print trading signals in a nice format"""
    
    if not signals:
        print("\n⚠️  No signals generated (models not found?)")
        print("   Run: python run_train.py")
        return
    
    # Sort by action priority
    action_priority = {'BUY': 0, 'SELL': 1, 'HOLD': 2, 'SKIP': 3}
    signals.sort(key=lambda x: action_priority.get(x['action'], 4))
    
    # Buy signals
    buy_signals = [s for s in signals if s['action'] == 'BUY']
    if buy_signals:
        print("\n" + "🟢" * 40)
        print("BUY SIGNALS")
        print("🟢" * 40)
        for s in buy_signals:
            print(f"{s['ticker']:10s} | {s['price']:8.2f} kr | Prob: {s['probability']:.1%} | {s['reason']}")
    
    # Sell signals
    sell_signals = [s for s in signals if s['action'] == 'SELL']
    if sell_signals:
        print("\n" + "🔴" * 40)
        print("SELL SIGNALS")
        print("🔴" * 40)
        for s in sell_signals:
            print(f"{s['ticker']:10s} | {s['price']:8.2f} kr | Prob: {s['probability']:.1%} | {s['reason']}")
    
    # Holdings
    hold_signals = [s for s in signals if s['action'] == 'HOLD']
    if hold_signals:
        print("\n" + "🟡" * 40)
        print("CURRENT HOLDINGS")
        print("🟡" * 40)
        for s in hold_signals:
            print(f"{s['ticker']:10s} | {s['shares']:3d} aktier @ {s['price']:8.2f} kr | Prob: {s['probability']:.1%}")
    
    # No signal
    skip_signals = [s for s in signals if s['action'] == 'SKIP']
    if skip_signals:
        print("\n" + "⚪" * 40)
        print("NO ACTION (Probability too low)")
        print("⚪" * 40)
        for s in skip_signals[:10]:  # Show first 10
            print(f"{s['ticker']:10s} | {s['price']:8.2f} kr | Prob: {s['probability']:.1%}")
        if len(skip_signals) > 10:
            print(f"... and {len(skip_signals) - 10} more")


def print_portfolio_status(portfolio):
    """Print current portfolio status"""
    print("\n" + "=" * 80)
    print("💰 PORTFOLIO STATUS")
    print("=" * 80)
    print(f"Cash available: {portfolio['cash']:10,.2f} kr")
    
    if portfolio['holdings']:
        print(f"\nHoldings:")
        for ticker, shares in sorted(portfolio['holdings'].items()):
            print(f"  {ticker:10s}: {shares:5d} shares")
    else:
        print("No holdings currently")
    
    print(f"\nTotal trades executed: {len(portfolio['trades'])}")


if __name__ == '__main__':
    print("=" * 80)
    print("🚀 LIVE TRADING SYSTEM - SIGNAL GENERATOR")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load portfolio from Excel trades
    print("\n📂 Loading portfolio from trades.xlsx...")
    portfolio = load_portfolio_from_trades()
    print_portfolio_status(portfolio)
    
    # Save to trade_log.json for backup
    print("\n💾 Saving trades to trade_log.json...")
    with open(TRADE_LOG_FILE, 'w') as f:
        json.dump(portfolio['trades'], f, indent=2)
    print(f"✅ Saved {len(portfolio['trades'])} trades")
    
    # Download latest data
    print()
    all_data = get_latest_data(days_back=90)
    
    if not all_data:
        print("❌ No data downloaded. Check your internet connection.")
        exit(1)
    
    # Generate signals
    signals = generate_signals(portfolio, all_data, MODELS_DIR)
    
    # Print signals
    print_signals(signals, portfolio)
    
    # Plot portfolio performance vs index
    if portfolio['trades']:
        print("\n📈 Plotting portfolio performance...")
        
        first_trade_date = pd.to_datetime(portfolio['trades'][0]['date'])
        end_date = datetime.now()
        
        # Try to get OMXS30 index data (Swedish stock market index)
        index_data = None
        try:
            # Try different index symbols
            for index_ticker in ['^OMXS30', 'OMXSPI.ST', '^OMXSPI']:
                try:
                    index_data = yf.download(index_ticker, start=first_trade_date, end=end_date, progress=False)
                    if len(index_data) > 0:
                        print(f"✅ Using index: {index_ticker}")
                        break
                except:
                    continue
            
            if index_data is not None and len(index_data) > 0:
                # Calculate portfolio value over time (cash + stock values)
                portfolio_values = []
                index_values = []
                dates = []
                
                cash = 10_000.0
                holdings = {}
                
                for trade in portfolio['trades']:
                    trade_date = pd.to_datetime(trade['date'])
                    ticker = trade['ticker']
                    action = trade['action']
                    shares = trade['shares']
                    price = trade['price']
                    
                    # Update holdings
                    if action == 'BUY':
                        holdings[ticker] = holdings.get(ticker, 0) + shares
                        cash -= shares * price
                    elif action == 'SELL':
                        holdings[ticker] = max(0, holdings.get(ticker, 0) - shares)
                        cash += shares * price
                    
                    # Calculate total portfolio value = cash + stock values at trade price
                    portfolio_value = cash
                    for held_ticker, held_shares in holdings.items():
                        portfolio_value += held_shares * price  # Use latest trade price
                    
                    portfolio_values.append(portfolio_value)
                    dates.append(trade_date)
                    
                    # Get index value at this date
                    closest_date = index_data.index[index_data.index.get_indexer([trade_date], method='nearest')[0]]
                    index_values.append(index_data.loc[closest_date, 'Close'])
                
                # Normalize to 10,000 at start
                index_values_normalized = [v / index_values[0] * 10_000 for v in index_values]
                
                # Plot
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(dates, portfolio_values, marker='o', linewidth=2.5, label='Your Portfolio (Cash + Stocks)', color='steelblue')
                ax.plot(dates, index_values_normalized, marker='s', linewidth=2, label='Market Index (normalized)', color='orange', alpha=0.7)
                
                ax.axhline(y=10_000, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Value (kr)', fontsize=12)
                ax.set_title('Portfolio Performance vs Market Index', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                plt.tight_layout()
                
                plot_file = Path('results/portfolio_vs_index.png')
                plot_file.parent.mkdir(exist_ok=True)
                plt.savefig(plot_file, dpi=150)
                print(f"✅ Saved plot to {plot_file}")
                plt.show()
                
                # Print performance
                final_portfolio = portfolio_values[-1]
                final_index = index_values_normalized[-1]
                portfolio_return = (final_portfolio / 10_000 - 1) * 100
                index_return = (final_index / 10_000 - 1) * 100
                
                print(f"\n📊 Performance Summary:")
                print(f"   Portfolio: {final_portfolio:,.2f} kr ({portfolio_return:+.2f}%)")
                print(f"   Index:     {final_index:,.2f} kr ({index_return:+.2f}%)")
                print(f"   Difference: {portfolio_return - index_return:+.2f}%")
                
                # Show current holdings
                print(f"\n📈 Current Holdings:")
                print(f"   Cash: {cash:,.2f} kr")
                for ticker, shares in sorted(holdings.items()):
                    print(f"   {ticker}: {shares} shares")
            else:
                print("⚠️  Could not fetch index data - skipping index plot")
                print("   But portfolio data is still valid")
        
        except Exception as e:
            print(f"⚠️  Could not plot index comparison: {e}")
            print("   But your portfolio data is still loaded")
    
    print("\n" + "=" * 80)
    print("📋 NEXT STEPS:")
    print("=" * 80)
    print("1. Review the signals above")
    print("2. Execute any BUY or SELL trades you want to make")
    print("3. Update trades.xlsx with your executed trades")
    print("4. Run: python live_trading.py (this script) again")
    print("=" * 80)
