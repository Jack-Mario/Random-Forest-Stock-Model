import pandas as pd
import numpy as np
from pathlib import Path


def backtest_from_preds(price_parquet: str, preds_parquet: str, initial_cash: float = 10_000.0, fee_rate: float = 0.001):
    """Simple long/flat backtest using integer whole-share purchases.

    Returns equity_curve (pd.Series) and trades (pd.DataFrame)
    """
    prices = pd.read_parquet(price_parquet)[['Close']]
    preds = pd.read_parquet(preds_parquet)['Pred']

    idx = preds.index.intersection(prices.index)
    prices = prices.loc[idx, 'Close'].astype(float)
    signal = preds.loc[idx].astype(int)

    cash = initial_cash
    shares = 0
    trades = []

    for dt, s, p in zip(idx, signal.values, prices.values):
        if np.isnan(p):
            continue
        if s == 1 and shares == 0:
            fee = cash * fee_rate
            invest = cash - fee
            if invest > p:
                buy_shares = int(invest / p)
                cash = invest - buy_shares * p
                shares = buy_shares
                trades.append((dt, 'BUY', p, shares, cash))
        elif s == -1 and shares > 0:
            notional = shares * p
            fee = notional * fee_rate
            cash = notional - fee
            trades.append((dt, 'SELL', p, shares, cash))
            shares = 0

    # Build equity
    cash = initial_cash
    shares = 0
    equity = []
    for dt, s, p in zip(idx, signal.values, prices.values):
        if s == 1 and shares == 0:
            fee = cash * fee_rate
            buy_shares = int((cash - fee) / p)
            cash = (cash - fee) - buy_shares * p
            shares = buy_shares
        elif s == -1 and shares > 0:
            notional = shares * p
            fee = notional * fee_rate
            cash = notional - fee
            shares = 0
        equity.append(cash + shares * p)

    equity_curve = pd.Series(equity, index=idx, name='Equity')
    trades_df = pd.DataFrame(trades, columns=['Date','Action','Price','Shares','CashAfter']).set_index('Date')
    return equity_curve, trades_df
