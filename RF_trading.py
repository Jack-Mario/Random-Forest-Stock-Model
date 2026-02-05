import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# List of stocks (same as in RF_get_data.py and RF_model.py)
stocks = {
    "VOLV_B_ST": "Volvo Group",
    "VOLCAR_B_ST": "Volvo Cars",
    "ERIC_B_ST": "Ericsson",
    "HM_B_ST": "H&M",
    "SAND_ST": "Sandvik",
    "SKA_B_ST": "Skanska",
    "AZN_ST": "AstraZeneca",
    "ICA_ST": "ICA Gruppen"
}


# ----------------------------
# 1) Load + prepare data
# ----------------------------
def load_features_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Drop rows with NaNs from rolling/shift etc.
    df = df.dropna()

    return df


def make_time_split(df: pd.DataFrame, target_col: str = "Increase", test_size: float = 0.2):
    # y must NOT be in X
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Drop non-numeric columns if any slipped in (e.g., strings)
    X = X.select_dtypes(include=[np.number])

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
# ----------------------------
# 2) Train + predict
# ----------------------------
def train_rf(X_train, y_train, n_estimators=300, random_state=42):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


# ----------------------------
# 3) Backtest (buy/sell on test dates)
# ----------------------------
def backtest_long_flat(
    df: pd.DataFrame,
    signal: pd.Series,
    price_col: str = "Close",
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
):
    """
    Long/flat backtest:
      signal = +1 => be long (buy if not long)
      signal = -1 => be flat (sell if long)

    Trades at the price_col on the same index as signal.
    """
    idx = signal.index.intersection(df.index)
    signal = signal.loc[idx].astype(int)

    prices = df.loc[idx, price_col].astype(float)

    cash = initial_cash
    shares = 0

    trades = []
    buy_dates = []
    sell_dates = []
    buy_prices = []
    sell_prices = []

    for dt, s, p in zip(idx, signal.values, prices.values):
        if np.isnan(p):
            continue

        # BUY if signal=+1 and not in position
        if s == 1 and shares == 0:
            notional = cash
            fee = notional * fee_rate
            investable = cash - fee
            if investable > 0:
                shares = int(investable / p)
                cash = investable - (shares * p)
                if shares > 0:
                    trades.append((dt, "BUY", p, shares, cash))
                    buy_dates.append(dt)
                    buy_prices.append(p)

        # SELL if signal=-1 and in position
        elif s == -1 and shares > 0:
            notional = shares * p
            fee = notional * fee_rate
            cash = notional - fee
            shares = 0
            trades.append((dt, "SELL", p, shares, cash))
            sell_dates.append(dt)
            sell_prices.append(p)

    # Build proper equity curve
    cash = initial_cash
    shares = 0
    equity = []
    for dt, s, p in zip(idx, signal.values, prices.values):
        if s == 1 and shares == 0:
            fee = cash * fee_rate
            shares = int((cash - fee) / p)
            cash = (cash - fee) - (shares * p)
        elif s == -1 and shares > 0:
            notional = shares * p
            fee = notional * fee_rate
            cash = notional - fee
            shares = 0
        equity.append(cash + shares * p)
    equity_curve = pd.Series(equity, index=idx, name="Equity")

    trades_df = pd.DataFrame(trades, columns=["Date", "Action", "Price", "Shares", "CashAfter"]).set_index("Date")

    return equity_curve, trades_df, buy_dates, buy_prices, sell_dates, sell_prices


def buy_and_hold(df: pd.DataFrame, idx: pd.Index, price_col: str, initial_cash: float = 10_000.0):
    prices = df.loc[idx, price_col].astype(float)
    entry = prices.iloc[0]
    shares = int(initial_cash / entry)
    curve = shares * prices
    return curve.rename("BuyHold")


# ----------------------------
# 4) Run backtest for all stocks
# ----------------------------
def main():
    print("=" * 70)
    print("BACKTESTING TRADING STRATEGY FOR ALL SWEDISH STOCKS")
    print("=" * 70)
    
    results = {}
    
    for ticker, company_name in stocks.items():
        filename = f"data_{ticker}.csv"
        try:
            print(f"\nProcessing {company_name}...")
            
            # Load data
            df = load_features_csv(filename)
            
            # Train/test split
            X_train, X_test, y_train, y_test = make_time_split(df, target_col="Increase", test_size=0.2)
            
            # Train model
            clf = train_rf(X_train, y_train, n_estimators=300, random_state=42)
            y_pred = pd.Series(clf.predict(X_test), index=X_test.index, name="Pred")
            
            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            print(f"  Accuracy: {acc:.4f}")
            
            # Backtest
            equity_curve, trades, buy_dates, buy_prices, sell_dates, sell_prices = backtest_long_flat(
                df=df,
                signal=y_pred.copy(),
                price_col="Close",
                initial_cash=10_000.0,
                fee_rate=0.001
            )
            
            bh_curve = buy_and_hold(df, equity_curve.index, price_col="Close", initial_cash=10_000.0)
            
            final_val = equity_curve.iloc[-1]
            bh_final = bh_curve.iloc[-1]
            
            print(f"  Final value (strategy): {final_val:,.2f}")
            print(f"  Final value (buy&hold): {bh_final:,.2f}")
            print(f"  Number of trades: {len(trades)}")
            
            results[ticker] = {
                'company': company_name,
                'df': df,
                'equity_curve': equity_curve,
                'bh_curve': bh_curve,
                'trades': trades,
                'buy_dates': buy_dates,
                'buy_prices': buy_prices,
                'sell_dates': sell_dates,
                'sell_prices': sell_prices,
                'final_val': final_val,
                'bh_final': bh_final,
                'accuracy': acc
            }
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Plot results for each stock
    print("\n" + "=" * 70)
    print("GENERATING PLOTS FOR ALL STOCKS")
    print("=" * 70)
    
    for ticker, data in results.items():
        try:
            print(f"\nPlotting {data['company']}...")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            fig.suptitle(f"{data['company']} - Trading Strategy Results", fontsize=14, fontweight='bold')
            
            # Plot 1: Price with buy/sell signals
            test_prices = data['df'].loc[data['equity_curve'].index, "Close"]
            ax1.plot(data['equity_curve'].index, test_prices.values, label="Stock Price", color="black", linewidth=1)
            ax1.scatter(data['buy_dates'], data['buy_prices'], marker="^", color="blue", s=100, label="Buy Signal", zorder=5)
            ax1.scatter(data['sell_dates'], data['sell_prices'], marker="v", color="red", s=100, label="Sell Signal", zorder=5)
            ax1.set_ylabel("Price (SEK)")
            ax1.set_title("Stock Price with Buy (Blue) and Sell (Red) Signals")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Portfolio value vs buy&hold
            ax2.plot(data['equity_curve'].index, data['equity_curve'].values, label="Strategy", linewidth=2)
            ax2.plot(data['bh_curve'].index, data['bh_curve'].values, label="Buy & Hold", linewidth=2)
            ax2.set_ylabel("Portfolio Value (SEK)")
            ax2.set_xlabel("Date")
            ax2.set_title(f"Portfolio Value (Strategy: {data['final_val']:,.0f} vs Buy&Hold: {data['bh_final']:,.0f})")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(f"trading_backtest_{ticker}.png", dpi=100)
            print(f"  Saved plot to trading_backtest_{ticker}.png")
            plt.show()
            
        except Exception as e:
            print(f"  Error plotting {ticker}: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ticker, data in results.items():
        print(f"{data['company']:30s} | Strategy: {data['final_val']:10,.0f} | Buy&Hold: {data['bh_final']:10,.0f} | Accuracy: {data['accuracy']:.2%}")


if __name__ == "__main__":
    main()
