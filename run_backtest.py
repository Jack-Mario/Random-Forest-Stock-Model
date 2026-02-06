from pathlib import Path
from src.portfolio_backtest import load_all_data, portfolio_backtest, COURTAGE, SPREAD, BUY_THRESHOLD, SELL_THRESHOLD, MIN_HOLD_DAYS
from src.plot import plot_equity
import pandas as pd
import numpy as np

if __name__ == '__main__':
    tickers = [
        'ABB.ST', 'ADDT-B.ST', 'ALFA.ST', 'ASSA-B.ST', 'AZN.ST', 'ATCO-A.ST', 'BOL.ST',
        'EPI-A.ST', 'EQT.ST', 'ERIC-B.ST', 'ESSITY-B.ST', 'EVO.ST', 'SHB-A.ST', 'HM-B.ST',
        'HEXA-B.ST', 'INDU-C.ST', 'INVE-B.ST', 'LIFCO-B.ST', 'NIBE-B.ST', 'NDA-SE.ST',
        'SAAB-B.ST', 'SAND.ST', 'SCA-B.ST', 'SEB-A.ST', 'SKA-B.ST', 'SKF-B.ST', 'SWED-A.ST',
        'TEL2-B.ST', 'TELIA.ST', 'VOLV-B.ST'
    ]
    
    res_dir = Path('results')
    res_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("PORTFOLIO BACKTEST - MULTI-STOCK TRADING WITH COSTS")
    print("=" * 80)
    
    try:
        # Load all data
        print("\nLoading data for all tickers...")
        data = load_all_data('data', 'models', tickers)
        print(f"  Loaded {len(data['tickers'])} tickers with {len(data['common_idx'])} common dates")
        print(f"  Tickers: {data['tickers']}")
        
        # Run portfolio backtest
        print("\nRunning portfolio backtest...")
        print(f"  Parameters: Courtage={COURTAGE*100:.2f}%, Spread={SPREAD*100:.2f}%, BUY_threshold={BUY_THRESHOLD}, SELL_threshold={SELL_THRESHOLD}, MIN_hold_days={MIN_HOLD_DAYS}")
        metrics, equity_curve, trades_log = portfolio_backtest(data)
        
        # Compute equal-weight buy-and-hold index
        print("Computing equal-weight buy-and-hold index...")
        INITIAL_CASH = 10_000.0
        prices = data['prices']
        common_idx_list = sorted(data['common_idx'])
        
        # First date: buy equal shares
        first_date = common_idx_list[0]
        holdings_bh = {}
        cash_bh = INITIAL_CASH
        
        for ticker in data['tickers']:
            price = prices[ticker].loc[first_date]
            if not np.isnan(price) and price > 0:
                shares = int(cash_bh / len(data['tickers']) / price)
                if shares > 0:
                    holdings_bh[ticker] = shares
                    cash_bh -= shares * price
                else:
                    holdings_bh[ticker] = 0
            else:
                holdings_bh[ticker] = 0
        
        # Hold till end
        bh_equity_list = []
        for date in common_idx_list:
            pv = cash_bh + sum(holdings_bh[t] * prices[t].loc[date] for t in data['tickers'] if not np.isnan(prices[t].loc[date]))
            bh_equity_list.append({'date': date, 'pv': pv})
        
        bh_equity_curve = pd.DataFrame(bh_equity_list).set_index('date')
        
        # Print summary
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Strategy:")
        print(f"  Final portfolio value: {metrics['final_value']:,.2f} kr")
        print(f"  Return: {metrics['return_pct']:.2f}%")
        print(f"  Number of trades: {metrics['n_trades']}")
        print(f"  Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"\nBuy & Hold Index:")
        print(f"  Final value: {bh_equity_curve['pv'].iloc[-1]:,.2f} kr")
        print(f"  Return: {(bh_equity_curve['pv'].iloc[-1] / INITIAL_CASH - 1) * 100:.2f}%")
        
        # Print all trades
        print("\n" + "=" * 80)
        print("ALL TRADES")
        print("=" * 80)
        if trades_log:
            for trade in trades_log:
                print(f"{trade['date']} | {trade['action']:4s} | {trade['ticker']:10s} | {trade['shares']:4d} @ {trade['price']:8.2f} kr | Fee: {trade['fee']:7.2f} kr | PV: {trade['pv']:10.2f} kr")
            print(f"\nTrades per week (average): {metrics['trades_per_week']:.2f}")
        else:
            print("No trades made")
        
        # Plot equity curve
        print("\n" + "=" * 80)
        print("Generating plots...")
        plot_equity(equity_curve.index, equity_curve['pv'].values, 
                    bh_equity_curve.index, bh_equity_curve['pv'].values,
                    out_file=res_dir / 'portfolio_equity.png', 
                    title='Portfolio Value: ML Trading Strategy vs Equal-Weight B&H Index')
        print(f"  Saved plot to {res_dir / 'portfolio_equity.png'}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        traceback.print_exc()

