# %%
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np # Import numpy for NaN handling

# Import talib directly
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib is installed and will be used for MACD calculation.")
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib is NOT installed. MACD calculation cannot proceed without TA-Lib.")
    print("Please install TA-Lib (e.g., pip install TA-Lib) for this script to function correctly.")


# %%
def plot_macd(data, ticker_symbol, buy_signal_dates=None, sell_signal_dates=None):
    """
    Generates and saves a MACD plot for the given stock data,
    with optional markers for multiple buy and sell signal dates.
    The histogram will distinguish between four states:
    - Stronger Green (increasing positive momentum)
    - Lesser Green (decreasing positive momentum)
    - Stronger Red (increasing negative momentum)
    - Lesser Red (decreasing negative momentum)

    Args:
        data (pd.DataFrame): DataFrame containing 'Close', 'MACD_12_26_9',
                             'MACDs_12_26_9', and 'MACDh_12_26_9' columns.
        ticker_symbol (str): The stock ticker symbol.
        buy_signal_dates (list, optional): A list of date strings (YYYY-MM-DD) where buy signals were detected.
                                          If provided, upward triangle markers will be added for each.
        sell_signal_dates (list, optional): A list of date strings (YYYY-MM-DD) where sell signals were detected.
                                           If provided, downward triangle markers will be added for each.
    """
    print(f"\n=== Plotting MACD for {ticker_symbol} ===")
    print(f"Data index range: {data.index.min()} to {data.index.max()}")
    print(f"Buy signal dates for plot: {buy_signal_dates}")
    print(f"Sell signal dates for plot: {sell_signal_dates}")

    # Ensure all necessary MACD columns exist
    required_cols = ['Close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Error: Required columns {missing_cols} not found in data for {ticker_symbol}. Cannot plot.")
        print(f"Available columns: {list(data.columns)}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'MACD Analysis for {ticker_symbol}', fontsize=16)

    # Plotting Price
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # Plotting MACD Line and Signal Line
    ax2.plot(data.index, data['MACD_12_26_9'], label='MACD Line', color='orange')
    ax2.plot(data.index, data['MACDs_12_26_9'], label='Signal Line', color='purple')

    # Determine colors for histogram bars based on value and change
    histogram_colors = []
    for i in range(len(data)):
        current_val = data['MACDh_12_26_9'].iloc[i]
        if i > 0:
            previous_val = data['MACDh_12_26_9'].iloc[i-1]
            if current_val >= 0: # Positive histogram
                if current_val > previous_val:
                    histogram_colors.append('darkgreen') # Stronger Green
                else:
                    histogram_colors.append('lightgreen') # Lesser Green
            else: # Negative histogram
                if abs(current_val) > abs(previous_val):
                    histogram_colors.append('darkred') # Stronger Red
                else:
                    histogram_colors.append('lightcoral') # Lesser Red (less negative)
        else:
            # First bar, no previous to compare, default to standard green/red
            histogram_colors.append('darkgreen' if current_val >= 0 else 'darkred')

    ax2.bar(data.index, data['MACDh_12_26_9'], label='Histogram', color=histogram_colors, alpha=0.7)

    ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Zero line for histogram
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlabel('Date')

    # Add buy signal markers if provided
    if buy_signal_dates:
        for buy_date_str in buy_signal_dates:
            try:
                # Convert to pandas Timestamp and normalize to remove time component
                signal_date_ts = pd.to_datetime(buy_date_str).normalize()
                
                # Ensure the date is in the index
                if signal_date_ts in data.index:
                    signal_data = data.loc[signal_date_ts]
                    
                    # Add marker to price chart (upper subplot)
                    ax1.plot(signal_date_ts, signal_data['Close'],
                            marker='^', markersize=15, color='lime', mec='black', mew=1.5,
                            label='Buy Signal' if buy_date_str == buy_signal_dates[0] else None, # Label only for the first marker
                            zorder=5, alpha=0.9)
                    
                    # Add marker to MACD chart (lower subplot)
                    ax2.plot(signal_date_ts, signal_data['MACDh_12_26_9'],
                            marker='^', markersize=15, color='lime', mec='black', mew=1.5,
                            zorder=5, alpha=0.9)
                    
                    # Add vertical line for better visibility
                    ax1.axvline(x=signal_date_ts, color='lime', linestyle='--', alpha=0.5, linewidth=1)
                    ax2.axvline(x=signal_date_ts, color='lime', linestyle='--', alpha=0.5, linewidth=1)
                    
                    print(f"✅ Buy signal marked on plot at {buy_date_str}")
                else:
                    print(f"❌ Warning: Buy signal date {buy_date_str} not found in data index for plotting.")
                    
            except Exception as e:
                print(f"❌ Error marking buy signal on plot for {buy_date_str}: {e}")
                import traceback
                traceback.print_exc()

    # Add sell signal markers if provided
    if sell_signal_dates:
        for sell_date_str in sell_signal_dates:
            try:
                # Convert to pandas Timestamp and normalize to remove time component
                signal_date_ts = pd.to_datetime(sell_date_str).normalize()
                
                # Ensure the date is in the index
                if signal_date_ts in data.index:
                    signal_data = data.loc[signal_date_ts]
                    
                    # Add marker to price chart (upper subplot)
                    ax1.plot(signal_date_ts, signal_data['Close'],
                            marker='v', markersize=15, color='red', mec='black', mew=1.5,
                            label='Sell Signal' if sell_date_str == sell_signal_dates[0] else None, # Label only for the first marker
                            zorder=5, alpha=0.9)
                    
                    # Add marker to MACD chart (lower subplot)
                    ax2.plot(signal_date_ts, signal_data['MACDh_12_26_9'],
                            marker='v', markersize=15, color='red', mec='black', mew=1.5,
                            zorder=5, alpha=0.9)
                    
                    # Add vertical line for better visibility
                    ax1.axvline(x=signal_date_ts, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax2.axvline(x=signal_date_ts, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    
                    print(f"✅ Sell signal marked on plot at {sell_date_str}")
                else:
                    print(f"❌ Warning: Sell signal date {sell_date_str} not found in data index for plotting.")
                    
            except Exception as e:
                print(f"❌ Error marking sell signal on plot for {sell_date_str}: {e}")
                import traceback
                traceback.print_exc()


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    
    # Create a directory for plots if it doesn't exist
    plot_dir = "macd_charts"
    os.makedirs(plot_dir, exist_ok=True)
    
    filename = os.path.join(plot_dir, f'{ticker_symbol}_macd_chart_combined.png') # Changed filename
    plt.savefig(filename)
    plt.close(fig) # Close the plot to free memory
    print(f"\nMACD chart saved to: {filename}")


def _process_stock_data(ticker_symbol, period):
    """Helper function to fetch and process stock data for MACD calculation."""
    data = yf.download(ticker_symbol, period=period)

    if data.empty:
        return None, f"No data found for {ticker_symbol} for the specified period."

    # Flatten columns if MultiIndex (e.g., [('Close', 'AAPL'), ...])
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


    # Ensure 'Close' column exists after potential multi-index flattening
    if 'Close' not in data.columns:
        return None, f"After processing, 'Close' column not found in data for {ticker_symbol}. Available: {list(data.columns)}"

    if not TALIB_AVAILABLE:
        return None, "TA-Lib is not installed, cannot calculate MACD."

    # Calculate MACD directly using TA-Lib
    # TA-Lib returns numpy arrays, which will have NaNs at the beginning
    macd_line_talib, signal_line_talib, hist_talib = talib.MACD(
        data['Close'].values,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    # Convert numpy arrays to pandas Series with the correct index
    # Fill initial NaNs with 0 to ensure no gaps in plotting
    macd_series = pd.Series(macd_line_talib, index=data.index).fillna(0)
    signal_series = pd.Series(signal_line_talib, index=data.index).fillna(0)
    histogram_series = pd.Series(hist_talib, index=data.index).fillna(0)

    # Assign the correct column names
    data['MACD_12_26_9'] = macd_series
    data['MACDs_12_26_9'] = signal_series
    data['MACDh_12_26_9'] = histogram_series

    # Check if MACD columns were successfully added and are not entirely NaN
    if data['MACDh_12_26_9'].isnull().all():
        return None, f"Could not calculate MACD for {ticker_symbol}. 'MACDh_12_26_9' is all NaN after TA-Lib calculation."

    # Need at least 5 data points to check for 5 consecutive bars
    if len(data) < 5:
        return None, f"Not enough valid data points ({len(data)}) after MACD calculation to check for 5 consecutive bars."

    return data, None


def get_macd_lesser_red_buy_signals(data):
    """
    Identifies buy signals based on 5 consecutive "lesser red" MACD histogram bars.
    Does NOT plot.

    Args:
        data (pd.DataFrame): DataFrame containing 'MACDh_12_26_9' column.

    Returns:
        list: A list of date strings (YYYY-MM-DD) where buy signals were detected.
    """
    print(f"Identifying BUY signals (Lesser Red)...")
    
    all_detected_signal_dates = [] 
    consecutive_lesser_red_count = 0

    # Iterate backward from the most recent data point.
    # Loop from len(data)-1 down to 4 (inclusive) to ensure i-4 is a valid index.
    # This allows checking 5 consecutive bars ending at index 'i'.
    for i in range(len(data) - 1, 4, -1):
        current_hist = data['MACDh_12_26_9'].iloc[i]
        prev_hist = data['MACDh_12_26_9'].iloc[i-1]
        
        # Condition for "lesser red": negative and less negative than previous
        is_lesser_red = (current_hist < 0) and (abs(current_hist) < abs(prev_hist))
        
        if is_lesser_red:
            consecutive_lesser_red_count += 1
        else:
            consecutive_lesser_red_count = 0 # Reset streak if condition is broken

        # If 5 or more consecutive 'lesser red' bars are found
        if consecutive_lesser_red_count >= 5:
            # The pattern completes on the current date (data.index[i])
            completion_date = data.index[i].strftime('%Y-%m-%d')
            
            # Add to list only if not already added (prevents duplicates if patterns overlap slightly)
            if completion_date not in all_detected_signal_dates:
                all_detected_signal_dates.append(completion_date)
                print(f"🎯 BUY SIGNAL DETECTED, pattern completed on: {completion_date}")
            
            # Reset count to find subsequent, distinct 5-bar patterns
            consecutive_lesser_red_count = 0 

    all_detected_signal_dates.sort() # Sort dates chronologically
    return all_detected_signal_dates


def get_macd_lesser_green_sell_signals(data):
    """
    Identifies sell signals based on 5 consecutive "lesser green" MACD histogram bars.
    Does NOT plot.

    Args:
        data (pd.DataFrame): DataFrame containing 'MACDh_12_26_9' column.

    Returns:
        list: A list of date strings (YYYY-MM-DD) where sell signals were detected.
    """
    print(f"Identifying SELL signals (Lesser Green)...")

    all_detected_signal_dates = []
    consecutive_lesser_green_count = 0

    # Iterate backward from the most recent data point.
    # Loop from len(data)-1 down to 4 (inclusive) to ensure i-4 is a valid index.
    for i in range(len(data) - 1, 4, -1):
        current_histogram = data['MACDh_12_26_9'].iloc[i]
        previous_histogram = data['MACDh_12_26_9'].iloc[i-1]

        # Condition for "lesser green": positive and less positive than previous
        is_lesser_green = (current_histogram > 0) and (current_histogram < previous_histogram)
        
        if is_lesser_green:
            consecutive_lesser_green_count += 1
        else:
            consecutive_lesser_green_count = 0 # Reset streak if condition is broken

        # If 5 or more consecutive 'lesser green' bars are found
        if consecutive_lesser_green_count >= 4:
            # The pattern completes on the current date (data.index[i])
            completion_date = data.index[i].strftime('%Y-%m-%d')
            
            # Add to list only if not already added
            if completion_date not in all_detected_signal_dates:
                all_detected_signal_dates.append(completion_date)
                print(f"🎯 SELL SIGNAL DETECTED, pattern completed on: {completion_date}")
            
            # Reset count to find subsequent, distinct 5-bar patterns
            consecutive_lesser_green_count = 0

    all_detected_signal_dates.sort() # Sort dates chronologically
    return all_detected_signal_dates

# %%
# --- Example Usage ---
if __name__ == "__main__":
    # Define a list of tickers to analyze
    tickers_to_analyze = ["AAPL", "MSFT", "NVDA", "SYM"]
    period = "1y" # Use a longer period for more backtesting data

    # Check if TA-Lib is available before proceeding with the main logic
    if not TALIB_AVAILABLE:
        print("\nSkipping analysis as TA-Lib is not installed.")
    else:
        for ticker in tickers_to_analyze:
            print(f"\n{'='*80}")
            print(f"Performing combined MACD signal analysis for {ticker} (period: {period})")

            # Fetch and process data once
            data, error_message = _process_stock_data(ticker, period)
            if data is None:
                print(f"❌ Error for {ticker}: {error_message}")
                # Plot an empty chart if data fetching fails
                plot_macd(pd.DataFrame(columns=['Close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']), ticker)
            else:
                print(f"Data fetched for {ticker} from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

                # Get all buy signals
                buy_signals = get_macd_lesser_red_buy_signals(data)
                if buy_signals:
                    print(f"\nFound BUY signals for {ticker}: {', '.join(buy_signals)}")
                else:
                    print(f"\nNo BUY signals found for {ticker}.")

                # Get all sell signals
                sell_signals = get_macd_lesser_green_sell_signals(data)
                if sell_signals:
                    print(f"\nFound SELL signals for {ticker}: {', '.join(sell_signals)}")
                else:
                    print(f"\nNo SELL signals found for {ticker}.")

                # Plot all signals on a single chart
                if buy_signals or sell_signals:
                    print(f"\nGenerating combined MACD chart for {ticker} with all detected signals...")
                    plot_macd(data, ticker, buy_signal_dates=buy_signals, sell_signal_dates=sell_signals)
                    print(f"\nCombined chart for {ticker} saved successfully.")
                else:
                    print(f"\nNo buy or sell signals found for {ticker} to plot.")
                    plot_macd(data, ticker) # Plot without signals if none found

            print(f"\n{'='*80}\n")

# %%
