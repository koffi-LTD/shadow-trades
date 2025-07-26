# %%
from lightweight_charts import Chart
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Hourly-specific equivalents of the daily MACD processing functions from macd_strategy.py
# Implemented locally to avoid daily-date assumptions.
from datetime import timedelta
# Import talib directly
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib is installed and will be used for MACD calculation.")
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib is NOT installed. MACD calculation cannot proceed without TA-Lib.")
    print("Please install TA-Lib (e.g., pip install TA-Lib) for this script to function correctly.")

# Keep all the existing functions as they are (plot_macd, _process_stock_data_for_macd, get_macd_lesser_red_buy_signals, get_macd_lesser_green_sell_signals)
# ... [Previous functions remain exactly the same until _process_stock_data] ...

def _process_stock_data(ticker_symbol, period, interval='1h'):
    """
    Helper function to fetch and process stock data for MACD calculation with hourly data.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        period (str): Time period for data (e.g., '60d', '1y')
        interval (str): Data interval, default is '1h' for hourly data
        
    Returns:
        pd.DataFrame: Processed stock data
    """
    # Use interval parameter to get hourly data
    data = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
    
    if data.empty:
        return None, f"No data found for {ticker_symbol} for the specified period."

    # Flatten columns if MultiIndex (e.g., [('close', 'AAPL'), ...])
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try to get the ticker-specific level first
            if ticker_symbol in data.columns.get_level_values(1):
                data = data.xs(ticker_symbol, axis=1, level=1, drop_level=False)
                data.columns = data.columns.droplevel(1)
            else: # Fallback if ticker not in level 1, assume level 0 is the main one
                data.columns = data.columns.droplevel(0)
        except Exception as e:
            # Fallback to just dropping the first level if multi-index is unexpected
            print(f"Warning: Could not process MultiIndex columns for {ticker_symbol} as expected: {e}. Attempting droplevel(0).")
            data.columns = data.columns.droplevel(0)

    return data

def _process_stock_data_for_macd_hourly(data: pd.DataFrame, ticker_symbol: str):
    """Calculate MACD columns on already-fetched hourly data using TA-Lib."""
    data = data.rename(columns=str.lower)
    if 'close' not in data.columns:
        return None, f"After processing, 'close' column not found in data for {ticker_symbol}. Available: {list(data.columns)}"

    if not TALIB_AVAILABLE:
        return None, "TA-Lib is not installed, cannot calculate MACD."

    macd_line, signal_line, hist = talib.MACD(
        data['close'].values,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )

    data['MACD_12_26_9'] = pd.Series(macd_line, index=data.index).fillna(0)
    data['MACDs_12_26_9'] = pd.Series(signal_line, index=data.index).fillna(0)
    data['MACDh_12_26_9'] = pd.Series(hist, index=data.index).fillna(0)

    if data['MACDh_12_26_9'].isnull().all():
        return None, f"Could not calculate MACD for {ticker_symbol}. MACDh_12_26_9 is all NaN."

    if len(data) < 10:
        return None, f"Not enough data points ({len(data)}) for MACD analysis."

    return data, None


def get_macd_lesser_red_buy_signals_hourly(data: pd.DataFrame):
    """Return timestamps (str) where 6 consecutive lesser-red histogram bars occur (hourly)."""
    all_dates = []
    streak = 0
    if 'MACDh_12_26_9' not in data:
        return all_dates

    for i in range(len(data) - 1, 6, -1):
        cur = data['MACDh_12_26_9'].iloc[i]
        prev = data['MACDh_12_26_9'].iloc[i - 1]
        is_lesser_red = (cur < 0) and (abs(cur) < abs(prev))
        streak = streak + 1 if is_lesser_red else 0
        if streak >= 6:
            ts = data.index[i]
            completion = ts.strftime('%Y-%m-%d %H:%M')
            if completion not in all_dates:
                all_dates.append(completion)
            streak = 0
    all_dates.sort()
    return all_dates


def get_macd_lesser_green_sell_signals_hourly(data: pd.DataFrame):
    """Return timestamps where 4 consecutive lesser-green histogram bars occur (hourly)."""
    all_dates = []
    streak = 0
    if 'MACDh_12_26_9' not in data:
        return all_dates

    for i in range(len(data) - 1, 4, -1):
        cur = data['MACDh_12_26_9'].iloc[i]
        prev = data['MACDh_12_26_9'].iloc[i - 1]
        is_lesser_green = (cur > 0) and (cur < prev)
        streak = streak + 1 if is_lesser_green else 0
        if streak >= 4:
            ts = data.index[i]
            completion = ts.strftime('%Y-%m-%d %H:%M')
            if completion not in all_dates:
                all_dates.append(completion)
            streak = 0
    all_dates.sort()
    return all_dates


def plot_macd_hourly(data: pd.DataFrame, ticker_symbol: str, buy_signal_dates=None, sell_signal_dates=None):
    """Plot MACD on hourly data keeping timestamp precision for signal markers."""
    print(f"\n=== Plotting Hourly MACD for {ticker_symbol} ===")
    required_cols = ['close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
    if any(col not in data for col in required_cols):
        print("Required MACD columns missing – skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Hourly MACD Analysis for {ticker_symbol}', fontsize=16)

    ax1.plot(data.index, data['close'], label='Close Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.legend(); ax1.grid(True)

    ax2.plot(data.index, data['MACD_12_26_9'], label='MACD', color='orange')
    ax2.plot(data.index, data['MACDs_12_26_9'], label='Signal', color='purple')
    hist_colors = []
    for i, val in enumerate(data['MACDh_12_26_9']):
        if i == 0:
            hist_colors.append('darkgreen' if val >= 0 else 'darkred')
            continue
        prev = data['MACDh_12_26_9'].iloc[i - 1]
        if val >= 0:
            hist_colors.append('darkgreen' if val > prev else 'lightgreen')
        else:
            hist_colors.append('darkred' if abs(val) > abs(prev) else 'lightcoral')
    # Use a bar width of ~40 minutes (0.03 days) to create some spacing between hourly bars
    ax2.bar(data.index, data['MACDh_12_26_9'], width=0.03, color=hist_colors, alpha=0.8, align='center')
    ax2.axhline(0, color='grey', ls='--', lw=0.8)
    ax2.set_ylabel('MACD'); ax2.legend(); ax2.grid(True)

    # Add buy/sell markers
    def _add_markers(sig_dates, marker, color):
        if not sig_dates:
            return
        for d in sig_dates:
            ts = pd.to_datetime(d)
            # handle timezone differences between index and ts
            index_compare = data.index
            if index_compare.tz is not None:
                index_compare = index_compare.tz_localize(None)
            ts_compare = ts.tz_localize(None) if ts.tzinfo is not None else ts
            # find nearest timestamp within 1 hour
            diffs = abs(index_compare - ts_compare)
            idx = diffs.argmin()
            if diffs[idx] <= timedelta(hours=1):
                ts_idx = data.index[idx]
                ax1.plot(ts_idx, data.loc[ts_idx, 'close'], marker=marker, ms=12, color=color, mec='black')
                ax2.plot(ts_idx, data.loc[ts_idx, 'MACDh_12_26_9'], marker=marker, ms=12, color=color, mec='black')
                ax1.axvline(ts_idx, color=color, ls='--', alpha=0.4)
                ax2.axvline(ts_idx, color=color, ls='--', alpha=0.4)

    _add_markers(buy_signal_dates, '^', 'lime')
    _add_markers(sell_signal_dates, 'v', 'red')

    ax2.set_xlabel('Timestamp')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outdir = "../data/yf/macd_charts_hourly"; os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{ticker_symbol}_macd_chart_hourly_combined.png")
    plt.savefig(fname); plt.close(fig)
    print(f"Chart saved to {fname}")

# %%
# --- Example Usage for Hourly Analysis ---
if __name__ == "__main__":
    # Define a list of tickers to analyze
    tickers_to_analyze = ["PLTR"]
    period = "60d"  # Shorter period for hourly data (e.g., 60 days)
    interval = "1h"  # Hourly interval

    # Check if TA-Lib is available before proceeding with the main logic
    if not TALIB_AVAILABLE:
        print("\nSkipping analysis as TA-Lib is not installed.")
    else:
        for ticker in tickers_to_analyze:
            print(f"\n{'='*80}")
            print(f"Performing hourly MACD signal analysis for {ticker} (period: {period}, interval: {interval})")

            # Fetch and process data with hourly interval
            # data_ = _process_stock_data(ticker, period, interval)
            raw_data = "../data/stock_bars.csv"
            df_csv = pd.read_csv(raw_data)
            df_csv = df_csv.rename(columns=str.lower)
            # filter symbol column to current ticker if present
            if 'symbol' in df_csv.columns:
                df_csv = df_csv[df_csv['symbol'] == ticker]
            # ensure timestamp is datetime and set as index
            if 'timestamp' in df_csv.columns:
                df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
                df_csv = df_csv.set_index('timestamp').sort_index()
            data_ = df_csv
            data, error_message = _process_stock_data_for_macd_hourly(data=data_, ticker_symbol=ticker)
            
            if data is None:
                print(f"❌ Error for {ticker}: {error_message}")
                # Plot an empty chart if data fetching fails
                plot_macd_hourly(pd.DataFrame(columns=['close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']), ticker)
            else:
                # Print the date range of the data
                print(f"Data fetched for {ticker} from {data.index[0].strftime('%Y-%m-%d %H:%M')} to {data.index[-1].strftime('%Y-%m-%d %H:%M')}")

                # Get all buy signals
                buy_signals = get_macd_lesser_red_buy_signals_hourly(data)
                if buy_signals:
                    print(f"\nFound BUY signals for {ticker}: {', '.join(buy_signals)}")
                else:
                    print(f"\nNo BUY signals found for {ticker}.")

                # Get all sell signals
                sell_signals = get_macd_lesser_green_sell_signals_hourly(data)
                if sell_signals:
                    print(f"\nFound SELL signals for {ticker}: {', '.join(sell_signals)}")
                else:
                    print(f"\nNo SELL signals found for {ticker}.")

                # Plot all signals on a single chart
                if buy_signals or sell_signals:
                    print(f"\nGenerating combined MACD chart for {ticker} with all detected signals...")
                    plot_macd_hourly(data, ticker, buy_signal_dates=buy_signals, sell_signal_dates=sell_signals)
                    print(f"\nCombined chart for {ticker} saved successfully.")
                else:
                    print(f"\nNo buy or sell signals found for {ticker} to plot.")
                    plot_macd_hourly(data, ticker)  # Plot without signals if none found

            print(f"\n{'='*80}\n")
            chart = Chart()
            chart.set(data_)
            chart.show(block=True)

