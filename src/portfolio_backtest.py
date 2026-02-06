import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from joblib import load
from datetime import timedelta

# Trading policy parameters
COURTAGE = 0.0025    # 0.25%
SPREAD = 0.0005      # 0.05% slippage
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.48
MIN_HOLD_DAYS = 14
INITIAL_CASH = 10_000.0


def load_all_data(data_dir: str, model_dir: str, tickers: List[str]) -> Dict:
    """Load price data and generate probability predictions using trained models.
    
    Returns dict with 'prices' (dict of Series per ticker), 'probs' (dict per ticker), 
    'common_idx' (intersection of all indices)
    """
    prices = {}
    probs = {}
    data_dfs = {}
    indices = []
    
    data_p = Path(data_dir)
    model_p = Path(model_dir)
    
    for ticker in tickers:
        sanitized = ticker.replace('.', '_').replace('-', '_')
        price_file = data_p / f"prices_{sanitized}.csv"
        data_file = data_p / f"data_{sanitized}.csv"
        model_file = model_p / f"data_{sanitized}.joblib"
        
        if not price_file.exists():
            print(f"  Warning: {price_file} not found, skipping {ticker}")
            continue
        if not data_file.exists():
            print(f"  Warning: {data_file} not found, skipping {ticker}")
            continue
        if not model_file.exists():
            print(f"  Warning: {model_file} not found, skipping {ticker}")
            continue
        
        price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        data_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        
        if price_df.empty or data_df.empty:
            print(f"  Warning: empty data for {ticker}, skipping")
            continue
        
        # Generate probabilities using trained model
        model_obj = load(model_file)
        clf = model_obj['model']
        feature_names = model_obj['features']
        
        X = data_df[feature_names]
        proba = clf.predict_proba(X)
        pos_idx = np.where(clf.classes_ == 1)[0][0]
        prob_positive = proba[:, pos_idx]
        prob_series = pd.Series(prob_positive, index=data_df.index, name='Prob')
        
        prices[ticker] = price_df['Close'].astype(float)
        probs[ticker] = prob_series
        data_dfs[ticker] = data_df
        indices.append(price_df.index)
    
    if not indices:
        raise ValueError("No valid ticker data found")
    
    # Intersection of all dates where all tickers have data
    common_idx = indices[0]
    for idx in indices[1:]:
        common_idx = common_idx.intersection(idx)
    
    # Align all to common index
    for ticker in prices:
        prices[ticker] = prices[ticker].loc[prices[ticker].index.isin(common_idx)]
        probs[ticker] = probs[ticker].loc[probs[ticker].index.isin(common_idx)]
    
    return {'prices': prices, 'probs': probs, 'common_idx': common_idx, 'tickers': list(prices.keys())}


def portfolio_backtest(data: Dict):
    """
    Trading policy backtest with realistic costs (spread + courtage).
    
    Policy:
    - Go LONG if P(up) > BUY_THRESHOLD
    - Go FLAT if P(up) < SELL_THRESHOLD
    - Min hold time: MIN_HOLD_DAYS
    
    Costs:
    - Spread: SPREAD % (slippage)
    - Courtage: COURTAGE % (commission)
    
    Returns:
        metrics: {final_value, return_pct, n_trades, max_drawdown_pct}
        equity_curve: DataFrame with date and portfolio value
        trades_log: list of trade dicts
    """
    tickers = data['tickers']
    prices = data['prices']  # dict ticker -> Series
    probs = data['probs']    # dict ticker -> Series
    common_idx = data['common_idx']
    
    # Initialize
    cash = INITIAL_CASH
    holdings = {t: 0 for t in tickers}
    last_trade_date = {t: None for t in tickers}
    equity_curve = []
    trades_log = []
    n_trades = 0
    
    # Day-by-day loop
    idx_list = sorted(common_idx)
    for i, date in enumerate(idx_list[:-1]):  # Can't trade on last day (no next day)
        next_date = idx_list[i + 1]
        
        # Get today's probabilities
        prob_dict = {}
        for t in tickers:
            if date in probs[t].index:
                prob_dict[t] = probs[t].loc[date]
            else:
                prob_dict[t] = 0.5  # neutral if missing
        
        # Trading policy: determine positions for next day
        to_buy = []
        to_sell = []
        
        for t in tickers:
            holding = holdings[t] > 0
            last_trade = last_trade_date[t]
            
            # Check min hold constraint
            if last_trade is not None:
                days_since = (pd.to_datetime(date) - pd.to_datetime(last_trade)).days
            else:
                days_since = 999  # first trade is always allowed
            
            p_up = prob_dict[t]
            
            # Decision logic
            if not holding and p_up > BUY_THRESHOLD and days_since >= MIN_HOLD_DAYS:
                to_buy.append(t)
            elif holding and p_up < SELL_THRESHOLD and days_since >= MIN_HOLD_DAYS:
                to_sell.append(t)
        
        # Execute sells first (at next_date close)
        for t in to_sell:
            if holdings[t] <= 0:
                continue
            
            price_next = prices[t].loc[next_date]
            if np.isnan(price_next):
                continue
            
            # Capture shares count before selling
            shares_to_sell = holdings[t]
            
            # Sell at (price - spread)
            sell_price = price_next * (1 - SPREAD)
            sale_value = shares_to_sell * sell_price
            courtage_fee = sale_value * COURTAGE
            cash_received = sale_value - courtage_fee
            
            cash += cash_received
            holdings[t] = 0
            last_trade_date[t] = next_date
            n_trades += 1
            
            # Portfolio value after trade
            pv = cash + sum(holdings[tt] * prices[tt].loc[next_date] for tt in tickers if not np.isnan(prices[tt].loc[next_date]))
            
            date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
            trades_log.append({
                'date': date_str,
                'action': 'SELL',
                'ticker': t,
                'shares': shares_to_sell,
                'price': sell_price,
                'fee': courtage_fee,
                'pv': pv
            })
        
        # Execute buys: split cash equally
        if to_buy and cash > 0:
            cash_per_ticker = cash / len(to_buy)
            
            for t in to_buy:
                price_next = prices[t].loc[next_date]
                if np.isnan(price_next) or price_next <= 0:
                    continue
                
                # Buy at (price + spread)
                buy_price = price_next * (1 + SPREAD)
                
                # How many shares can we buy?
                # cost = shares * buy_price + courtage = shares * buy_price * (1 + COURTAGE)
                # cost <= cash_per_ticker
                # shares <= cash_per_ticker / (buy_price * (1 + COURTAGE))
                max_shares = int(cash_per_ticker / (buy_price * (1 + COURTAGE)))
                
                if max_shares <= 0:
                    continue
                
                purchase_value = max_shares * buy_price
                courtage_fee = purchase_value * COURTAGE
                total_cost = purchase_value + courtage_fee
                
                if total_cost > cash:
                    continue
                
                cash -= total_cost
                holdings[t] = max_shares
                last_trade_date[t] = next_date
                n_trades += 1
                
                pv = cash + sum(holdings[tt] * prices[tt].loc[next_date] for tt in tickers if not np.isnan(prices[tt].loc[next_date]))
                
                date_str = next_date.strftime('%Y-%m-%d') if hasattr(next_date, 'strftime') else str(next_date)
                trades_log.append({
                    'date': date_str,
                    'action': 'BUY',
                    'ticker': t,
                    'shares': max_shares,
                    'price': buy_price,
                    'fee': courtage_fee,
                    'pv': pv
                })
        
        # Log equity at next_date close
        pv_end = cash + sum(holdings[t] * prices[t].loc[next_date] for t in tickers if not np.isnan(prices[t].loc[next_date]))
        equity_curve.append({'date': next_date, 'pv': pv_end})
    
    # Metrics
    df_eq = pd.DataFrame(equity_curve)
    final_value = df_eq['pv'].iloc[-1] if len(df_eq) > 0 else cash
    return_pct = (final_value / INITIAL_CASH - 1) * 100
    
    # Max drawdown
    if len(df_eq) > 0:
        running_max = df_eq['pv'].cummax()
        drawdown = (df_eq['pv'] - running_max) / running_max
        max_dd_pct = drawdown.min() * 100
    else:
        max_dd_pct = 0.0
    
    # Calculate trades per week
    if len(idx_list) > 1:
        first_date = pd.to_datetime(idx_list[0])
        last_date = pd.to_datetime(idx_list[-1])
        total_weeks = (last_date - first_date).days / 7
        trades_per_week = n_trades / total_weeks if total_weeks > 0 else 0
    else:
        trades_per_week = 0
    
    df_eq = df_eq.set_index('date')
    
    metrics = {
        'final_value': final_value,
        'return_pct': return_pct,
        'n_trades': n_trades,
        'max_drawdown_pct': max_dd_pct,
        'trades_per_week': trades_per_week
    }
    
    return metrics, df_eq, trades_log


def index_buy_and_hold(data: Dict, initial_cash: float = 10_000.0):
    """Equal-weighted buy-and-hold index."""
    tickers = data['tickers']
    prices = data['prices']
    idx = data['common_idx']
    
    # At first date, buy equal-weighted
    first_dt = idx[0]
    cash = initial_cash
    holdings = {}
    
    alloc_per_ticker = cash / len(tickers)
    for ticker in tickers:
        price = prices[ticker].loc[first_dt]
        if np.isnan(price) or price <= 0:
            holdings[ticker] = 0
        else:
            holdings[ticker] = int(alloc_per_ticker / price)
    
    # Compute portfolio value over time
    equity = []
    for dt in idx:
        pv = 0
        for ticker in tickers:
            price = prices[ticker].loc[dt]
            if not np.isnan(price):
                pv += holdings[ticker] * price
        equity.append(pv)
    
    return pd.Series(equity, index=idx, name='Index_BH')
