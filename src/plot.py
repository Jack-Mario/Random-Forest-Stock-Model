import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_price_signals(price_index, prices, buy_dates, buy_prices, sell_dates, sell_prices, out_file=None, title=None):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(price_index, prices, color='black', label='Price')
    if buy_dates:
        ax.scatter(buy_dates, buy_prices, marker='^', color='blue', s=80, label='Buy')
    if sell_dates:
        ax.scatter(sell_dates, sell_prices, marker='v', color='red', s=80, label='Sell')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_title(title or 'Price & Signals')
    ax.set_ylabel('Price')
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    plt.close()


def plot_equity(equity_index, equity, bh_index, bh_curve, out_file=None, title=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Ensure indices are pandas DatetimeIndex and remove timezone info
    if not isinstance(equity_index, pd.DatetimeIndex):
        equity_index = pd.to_datetime(equity_index, utc=True).tz_localize(None)
    else:
        # Remove timezone if present
        if equity_index.tz is not None:
            equity_index = equity_index.tz_localize(None)
    
    ax.plot(equity_index, equity, label='Strategy', linewidth=2, color='steelblue')
    
    if bh_index is not None and bh_curve is not None:
        if not isinstance(bh_index, pd.DatetimeIndex):
            bh_index = pd.to_datetime(bh_index, utc=True).tz_localize(None)
        else:
            # Remove timezone if present
            if bh_index.tz is not None:
                bh_index = bh_index.tz_localize(None)
        ax.plot(bh_index, bh_curve, label='Buy & Hold Index', linewidth=2, color='orange')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_title(title or 'Equity Curve', fontsize=14)
    ax.set_ylabel('Portfolio Value (kr)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=100)
    plt.close()
