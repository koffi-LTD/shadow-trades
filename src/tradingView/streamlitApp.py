import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alpaca.historical import fetch_stock_data
from alpaca.data.timeframe import TimeFrame


# Compatibility: Some libraries still import `NaN` from NumPy, which was removed in newer versions
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # provide alias for backward compatibility
import pandas_ta as ta

COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350




def get_macd_lesser_red_buy_signals(data, consecutive_bars=6, hist_col=None):
    """
    Identifies buy signals based on consecutive "lesser red" MACD histogram bars.
    
    Args:
        data (pd.DataFrame): DataFrame containing MACD histogram column.
        consecutive_bars (int): Number of consecutive bars required for a signal.
        hist_col (str, optional): Name of the MACD histogram column. If None, will try to find it.
        
    Returns:
        list: A list of date strings (YYYY-MM-DD HH:MM:SS) where buy signals were detected.
    """
    # Try to find the histogram column if not provided
    if hist_col is None:
        hist_cols = [col for col in data.columns if col.startswith('MACDh_')]
        if not hist_cols:
            raise ValueError("No MACD histogram column found. Make sure to calculate MACD first.")
        hist_col = hist_cols[0]  # Use the first matching column
        print(f"Using MACD histogram column: {hist_col}")
    
    print(f"Identifying BUY signals ({consecutive_bars} consecutive Lesser Red)...")
    
    all_detected_signal_dates = [] 
    pattern_bars = []  # Store the bars that form the pattern

    # Iterate through the data
    for i in range(len(data) - 1):
        current_hist = data[hist_col].iloc[i]
        prev_hist = data[hist_col].iloc[i-1] if i > 0 else None
        
        # Condition for "lesser red": negative and less negative than previous
        if current_hist is not None and prev_hist is not None:
            is_lesser_red = (current_hist < 0) and (abs(current_hist) < abs(prev_hist))
            
            if is_lesser_red:
                pattern_bars.append(i)
                
                # Check if we have enough consecutive bars
                if len(pattern_bars) >= consecutive_bars:
                    # Verify the bars are truly consecutive
                    is_consecutive = all(pattern_bars[j+1] - pattern_bars[j] == 1 
                                      for j in range(len(pattern_bars)-1))
                    
                    if is_consecutive:
                        start_idx = pattern_bars[0]
                        end_idx = pattern_bars[-1]
                        
                        start_date = data.index[start_idx].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[start_idx], 'strftime') else str(data.index[start_idx])
                        completion_date = data.index[end_idx].strftime('%Y-%m-%d %H:%M:%S') if hasattr(data.index[end_idx], 'strftime') else str(data.index[end_idx])
                        
                        if completion_date not in all_detected_signal_dates:
                            all_detected_signal_dates.append(completion_date)
                            print(f"🎯 {data['symbol']} BUY SIGNAL DETECTED, pattern started on: {start_date} completed on: {completion_date}")
                        
                        pattern_bars = []  # Reset for next pattern
            else:
                pattern_bars = []  # Reset on non-lesser-red bar

    all_detected_signal_dates.sort()
    return all_detected_signal_dates

# def get_second_buy_entry(data, buy_signals, consecutive_bars=3):
#     """
#     Identifies second buy entry points based on consecutive green bars after initial buy signals.
    
#     Args:
#         data (pd.DataFrame): DataFrame containing price data with 'open', 'close', and 'macd' columns
#         buy_signals (list): List of datetime objects representing initial buy signals
#         consecutive_bars (int): Number of consecutive green bars with MACD > 0 required for second entry (default 3)
        
#     Returns:
#         tuple: A tuple containing:
#             - list: A list of date strings where second buy entries were detected
#             - pd.DataFrame: A DataFrame with detailed second entry information
#     """
#     print(f"Identifying SECOND BUY entries ({consecutive_bars} consecutive green bars with MACD > 0 after buy signals)...")
    
#     if not buy_signals:
#         print("No buy signals provided, skipping second entry detection.")
#         return [], pd.DataFrame(columns=[
#             'Initial Buy Date', 'Second Entry Date', 'Days After Initial',
#             'Initial Price', 'Second Entry Price', 'Price Change %',
#             'MACD Value'
#         ])

#     second_entry_dates = []
#     second_entries = []
    
#     # Convert buy signals to datetime and ensure timezone consistency
#     buy_dates = []
#     for date in buy_signals:
#         date_dt = pd.to_datetime(date)
#         # Convert to timezone-naive if needed
#         if date_dt.tz is not None:
#             date_dt = date_dt.tz_localize(None)
#         buy_dates.append(date_dt.normalize())
    
#     # Ensure data index timezone consistency
#     index_series = data.index
#     if isinstance(index_series, pd.DatetimeIndex):
#         if index_series.tz is not None:
#             current_dates = index_series.tz_localize(None)
#         else:
#             current_dates = index_series
#     else:
#         current_dates = pd.to_datetime(index_series).tz_localize(None)

#     # Track which initial buy dates already have a second entry
#     processed_buy_dates = set()
    
#     for buy_date in buy_dates:
#         # Skip if we've already found a second entry for this buy signal
#         if buy_date in processed_buy_dates:
#             continue
            
#         # Find the index of the buy date in the data
#         buy_idx = None
#         for i, date in enumerate(current_dates):
#             if date.normalize() == buy_date:
#                 buy_idx = i
#                 break
        
#         if buy_idx is None or buy_idx >= len(data) - consecutive_bars - 1:
#             continue
            
#         # Look for exactly 3 consecutive STRONG_GREEN histogram bars
#         # right after the initial buy signal (first 5 bars only)
#         max_bars_to_check = 12  # Only check first 12 bars after buy
#         strong_green_count = 0
        
#         for i in range(buy_idx + 1, min(buy_idx + 1 + max_bars_to_check, len(data))):
#             current_hist = data['hist'].iloc[i] if 'hist' in data.columns else None
#             prev_hist = data['hist'].iloc[i-1] if 'hist' in data.columns else None
            
#             is_strong_green = (current_hist is not None and prev_hist is not None and 
#                              current_hist > prev_hist and current_hist > 0)
            
#             if is_strong_green:
#                 strong_green_count += 1
                
#                 if strong_green_count == 3:
#                     entry_idx = i
#                     entry_date = current_dates[entry_idx]
#                     entry_price = data['close'].iloc[entry_idx]
#                     initial_price = data['close'].iloc[buy_idx]
#                     price_change = ((entry_price - initial_price) / initial_price) * 100
#                     days_after = (entry_date.normalize() - buy_date).days
                    
#                     second_entry_dates.append(entry_date.strftime('%Y-%m-%d %H:%M:%S'))
#                     second_entries.append({
#                         'Initial Buy Date': buy_date,
#                         'Second Entry Date': entry_date,
#                         'Days After Initial': days_after,
#                         'Initial Price': initial_price,
#                         'Second Entry Price': entry_price,
#                         'Price Change %': price_change,
#                         'MACD Histogram': current_hist,
#                         'Pattern': '3x STRONG_GREEN (First 5 bars)'
#                     })
#                     print(f"🎯 SECOND BUY ENTRY at {entry_date.strftime('%Y-%m-%d')}")
#                     print(f"   - {days_after} days after initial buy at {buy_date.strftime('%Y-%m-%d')}")
#                     print(f"   - Price change: {price_change:+.2f}%")
#                     print(f"   - MACD Histogram: {current_hist:.4f} (3x STRONG_GREEN in first 5 bars)")
                    
#                     processed_buy_dates.add(buy_date)
#                     break  # Exit after first valid pattern
#             else:
#                 strong_green_count = 0  # Reset counter if pattern breaks 
#     # Create DataFrame from second entries
#     second_entry_df = pd.DataFrame(second_entries)
    
#     # Sort by second entry date if we have multiple entries
#     if not second_entry_df.empty:
#         second_entry_df = second_entry_df.sort_values('Second Entry Date')
    
#     return second_entry_dates, second_entry_df

import pandas as pd
from typing import List, Tuple, Dict, Any
import numpy as np

import pandas as pd
from typing import List, Tuple, Dict, Any
import numpy as np

def get_second_buy_entry(data: pd.DataFrame, buy_signals: List[str], consecutive_bars=3) -> Tuple[List[str], pd.DataFrame]:
    """
    Identifies second buy entry points based on the FIRST 3 consecutive green MACD histogram bars
    that are all LESSER (smaller) than the previous bar (decelerating momentum), occurring after 
    an initial buy signal.

    Args:
        data (pd.DataFrame): DataFrame containing price data with 'close' and 'hist' (MACD Histogram) columns.
        buy_signals (list): List of date strings (e.g., 'YYYY-MM-DD') representing initial buy signals.

    Returns:
        tuple: A tuple containing:
            - list: A list of date strings where second buy entries were detected.
            - pd.DataFrame: A DataFrame with detailed second entry information.
    """
    REQUIRED_CONSECUTIVE_BARS = consecutive_bars
    
    if 'hist' not in data.columns or 'close' not in data.columns:
        raise ValueError("DataFrame must contain 'hist' (MACD Histogram) and 'close' columns.")
        
    print(f"Identifying SECOND BUY entries (First {REQUIRED_CONSECUTIVE_BARS} consecutive green bars, each lesser/smaller than the previous)...")
    
    if not buy_signals:
        print("No buy signals provided, skipping second entry detection.")
        return [], pd.DataFrame(columns=[
            'Initial Buy Date', 'Second Entry Date', 'Days After Initial',
            'Initial Price', 'Second Entry Price', 'Price Change %',
            'MACD Histogram'
        ])

    # --- Preprocessing Dates and Index Consistency ---
    
    # Ensure data index is datetime-like and timezone-naive
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
        
    if data.index.tz is not None:
        data = data.tz_localize(None)

    # Convert buy signals to datetime and normalize
    buy_dates = [pd.to_datetime(date).normalize() for date in buy_signals]
    
    # --- Second Entry Logic ---
    
    second_entries: List[Dict[str, Any]] = []
    
    # Loop through each initial buy signal
    for i, buy_date in enumerate(buy_dates):
        # 1. Find the index of the initial buy date
        try:
            buy_idx = data.index.get_loc(buy_date)
        except KeyError:
            # Skip if initial buy date is not in data index
            continue

        initial_price = data['close'].iloc[buy_idx]
        
        # 2. Define the search window
        # Start searching from the bar immediately *after* the initial buy
        start_idx = buy_idx + 1
        
        # Limit the search until the next initial buy signal (if one exists)
        end_idx = len(data)
        if i + 1 < len(buy_dates):
            next_buy_date = buy_dates[i+1]
            try:
                next_buy_idx = data.index.get_loc(next_buy_date)
                end_idx = next_buy_idx 
            except KeyError:
                pass # Search to the end if next buy date is invalid
        
        # 3. Look for the 3-bar lesser green pattern
        
        # Iterate over the search window
        for j in range(start_idx, end_idx):
            # Ensure we have enough history for 3 bars (j, j-1, j-2)
            if j < start_idx + REQUIRED_CONSECUTIVE_BARS - 1:
                continue
                
            # Get the three consecutive bars ending at index j
            hist_c = data['hist'].iloc[j]       # Current bar (The 3rd bar / Entry Bar)
            hist_p1 = data['hist'].iloc[j-1]    # Previous bar (The 2nd bar)
            hist_p2 = data['hist'].iloc[j-2]    # Bar before previous (The 1st bar)
            
            # Check the conditions for the lesser green pattern:
            # 1. All 3 are green (MACD Hist > 0)
            is_green = (hist_c > 0) and (hist_p1 > 0) and (hist_p2 > 0)
            
            # 2. All 3 are LESSER than the previous (i.e., decelerating/weaker momentum)
            is_lesser = (hist_c < hist_p1) and (hist_p1 < hist_p2)
            
            if is_green and is_lesser:
                # Pattern found! This is the 'first 3 green MACD histogram' sequence with decreasing strength.
                entry_idx = j
                entry_date = data.index[entry_idx]
                entry_price = data['close'].iloc[entry_idx]
                price_change = ((entry_price - initial_price) / initial_price) * 100
                days_after = (entry_date.normalize() - buy_date).days
                
                second_entries.append({
                    'Initial Buy Date': buy_date.strftime('%Y-%m-%d'),
                    'Second Entry Date': entry_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'Days After Initial': days_after,
                    'Initial Price': initial_price,
                    'Second Entry Price': entry_price,
                    'Price Change %': price_change,
                    'MACD Histogram': hist_c,
                })
                print(f"🎯 SECOND BUY ENTRY found (Lesser Green Pattern) at {entry_date.strftime('%Y-%m-%d')}")
                print(f"   - {days_after} days after initial buy at {buy_date.strftime('%Y-%m-%d')}")
                break  # Stop searching after the *first* pattern is found for this buy_date


    # --- Final Output Formatting ---
    
    second_entry_df = pd.DataFrame(second_entries)
    
    if not second_entry_df.empty:
        second_entry_df = second_entry_df.sort_values('Second Entry Date')
        second_entry_dates = second_entry_df['Second Entry Date'].tolist()
    else:
        second_entry_dates = []
    
    return second_entry_dates, second_entry_df
def get_macd_lesser_green_sell_signals(data, consecutive_bars=5, hist_col=None):
    """
    Identifies sell signals based on consecutive "lesser green" MACD histogram bars.
    More stringent criteria to reduce false positives:
    - Each bar must be at least 20% smaller than the previous
    - The pattern must not be in a strong uptrend
    
    Args:
        data (pd.DataFrame): DataFrame containing MACD histogram column.
        consecutive_bars (int): Number of consecutive bars required for a signal (default 5).
        hist_col (str, optional): Name of the MACD histogram column. If None, will try to find it.
        
    Returns:
        list: A list of date strings (YYYY-MM-DD HH:MM:SS) where sell signals were detected.
    """
    # Try to find the histogram column if not provided
    if hist_col is None:
        hist_cols = [col for col in data.columns if col.startswith('MACDh_')]
        if not hist_cols:
            raise ValueError("No MACD histogram column found. Make sure to calculate MACD first.")
        hist_col = hist_cols[0]  # Use the first matching column
        print(f"Using MACD histogram column: {hist_col}")
    
    print(f"Identifying SELL signals ({consecutive_bars} consecutive Lesser Green with strict criteria)...")
    
    all_detected_signal_dates = []
    pattern_bars = []  # Store the bars that form the pattern

    # Define minimum decrease threshold (20%)
    MIN_DECREASE_PCT = 0.20

    # Get the MACD line column name based on histogram column
    macd_col = hist_col.replace('MACDh_', 'MACD_')
    
    # Iterate through the data
    for i in range(len(data) - 1):
        current_hist = data[hist_col].iloc[i]
        prev_hist = data[hist_col].iloc[i-1] if i > 0 else None
        
        # Skip if we don't have previous data for comparison
        if current_hist is None or prev_hist is None:
            continue
            
        # Conditions for a valid "lesser green" bar:
        # 1. Current bar must be positive
        # 2. Must be decreasing from previous bar
        # 3. Must decrease by at least MIN_DECREASE_PCT
        is_lesser_green = (
            current_hist > 0 and
            prev_hist > 0 and
            current_hist < prev_hist and
            (prev_hist - current_hist) / prev_hist >= MIN_DECREASE_PCT
        )
        
        if is_lesser_green:
            pattern_bars.append(i)
            
            # Check if we have enough consecutive bars
            if len(pattern_bars) >= consecutive_bars:
                # Verify the bars are truly consecutive
                is_consecutive = all(pattern_bars[j+1] - pattern_bars[j] == 1 
                                  for j in range(len(pattern_bars)-1))
                
                if is_consecutive:
                    # Additional check: Ensure we're not in a strong uptrend
                    # Look at the MACD line trend over the past 10 periods
                    start_idx = max(0, pattern_bars[0] - 10)
                    macd_trend = data[macd_col].iloc[start_idx:pattern_bars[-1]].mean()
                    
                    # Only generate sell signal if MACD trend is not strongly positive
                    if macd_trend <= data[macd_col].iloc[pattern_bars[-1]]:
                        completion_date = data.index[pattern_bars[-1]]
                        date_str = completion_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(completion_date, 'strftime') else str(completion_date)
                        
                        if date_str not in all_detected_signal_dates:
                            all_detected_signal_dates.append(date_str)
                            print(f"🎯 SELL SIGNAL DETECTED at {date_str}")
                            print(f"   - Pattern strength: {(prev_hist - current_hist) / prev_hist:.2%} decrease")
                            print(f"   - MACD trend: {macd_trend:.6f}")
                    
                pattern_bars = []  # Reset for next pattern
        else:
            pattern_bars = []  # Reset on non-lesser-green bar

    all_detected_signal_dates.sort()  # Sort dates chronologically
    return all_detected_signal_dates


def get_consecutive_red_candles_exit(data, consecutive_bars=3, hist_col=None, buy_signal_dates=None):
    """
    Identifies exit points based on consecutive red candles with decreasing closing prices.
    Only looks for exit points at least 5 days after a buy signal has occurred.
    Calculates percentage gain/loss from buy signal to exit point.
    
    Args:
        data (pd.DataFrame): DataFrame containing 'open' and 'close' columns
        consecutive_bars (int): Number of consecutive red candles required (default 3)
        hist_col (str, optional): Not used, kept for API consistency
        buy_signal_dates (list): List of buy signal dates to check against
        
    Returns:
        tuple: A tuple containing:
            - list: A list of date strings (YYYY-MM-DD HH:MM:SS) where exit signals were detected
            - pd.DataFrame: A DataFrame with detailed exit signal information
    """
    print(f"Identifying EXIT points ({consecutive_bars} consecutive red candles) at least 5 days after buy signals...")
    
    if not buy_signal_dates:
        print("No buy signals provided, skipping exit signal detection.")
        return [], pd.DataFrame(columns=[
            'Buy Date', 'Exit Date', 'Holding Days',
            'Entry Price', 'Exit Price', 'Total Gain %',
            'Pattern Decline %'
        ])

    all_detected_signal_dates = []
    pattern_bars = []  # Store the bars that form the pattern
    exit_signals = []
    
    # Convert buy signal dates to datetime and ensure timezone consistency
    buy_dates = []
    for date in buy_signal_dates:
        date_dt = pd.to_datetime(date)
        # Convert to timezone-naive if needed
        if date_dt.tz is not None:
            date_dt = date_dt.tz_localize(None)
        buy_dates.append(date_dt.normalize())
    
    earliest_buy_date = min(buy_dates)

    # Ensure data index timezone consistency
    index_series = data.index
    if isinstance(index_series, pd.DatetimeIndex):
        if index_series.tz is not None:
            # Convert index to timezone-naive for comparison
            current_dates = index_series.tz_localize(None)
        else:
            current_dates = index_series
    else:
        current_dates = pd.to_datetime(index_series).tz_localize(None)

    # Create a mapping of normalized dates to original indices
    date_to_index = {date.normalize(): i for i, date in enumerate(current_dates)}

    # Iterate through the data
    for i in range(len(data) - 1):
        current_date = current_dates[i].normalize()
        
        # Skip dates before the first buy signal
        if current_date < earliest_buy_date:
            continue
            
        # Check if we're after the most recent applicable buy signal
        # Find the most recent buy date before current date
        applicable_buys = [date for date in buy_dates if date <= current_date]
        if not applicable_buys:
            continue
            
        current_open = data['open'].iloc[i]
        current_close = data['close'].iloc[i]
        
        # Check if it's a red candle (close < open)
        is_red_candle = current_close < current_open
        
        if is_red_candle:
            # If this is the first red candle or continuing the pattern
            if not pattern_bars or (
                i > 0 and 
                current_close < data['close'].iloc[i-1]  # Decreasing closing price
            ):
                pattern_bars.append(i)
                
                # Check if we have enough consecutive bars
                if len(pattern_bars) >= consecutive_bars:
                    # Verify the bars are truly consecutive and closing prices are decreasing
                    is_consecutive = all(
                        pattern_bars[j+1] - pattern_bars[j] == 1 and
                        data['close'].iloc[pattern_bars[j+1]] < data['close'].iloc[pattern_bars[j]]
                        for j in range(len(pattern_bars)-1)
                    )
                    
                    if is_consecutive:
                        completion_date = data.index[pattern_bars[-1]]
                        # Convert to string in a consistent format
                        date_str = completion_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(completion_date, 'strftime') else str(completion_date)
                        
                        if date_str not in all_detected_signal_dates:
                            # Calculate immediate price decline in the pattern
                            pattern_decline = (data['close'].iloc[pattern_bars[-1]] - data['close'].iloc[pattern_bars[0]]) / data['close'].iloc[pattern_bars[0]] * 100
                            
                            # Find the most recent buy signal and calculate total gain/loss
                            most_recent_buy = max(applicable_buys)
                            
                            # Skip if less than 5 days have passed since the buy signal
                            days_since_buy = (completion_date.normalize() - most_recent_buy).days
                            if days_since_buy < 5:
                                continue
                            
                            # Find the buy index using our date_to_index mapping
                            buy_index = date_to_index.get(most_recent_buy)
                            if buy_index is not None:  # Only proceed if we found the index
                                buy_price = data['close'].iloc[buy_index]
                                exit_price = data['close'].iloc[pattern_bars[-1]]
                                total_gain = ((exit_price - buy_price) / buy_price) * 100
                                
                                # Add to exit signals list
                                exit_signals.append({
                                    'Buy Date': most_recent_buy,
                                    'Exit Date': completion_date,
                                    'Holding Days': days_since_buy,
                                    'Entry Price': buy_price,
                                    'Exit Price': exit_price,
                                    'Total Gain %': total_gain,
                                    'Pattern Decline %': abs(pattern_decline)
                                })
                                
                                all_detected_signal_dates.append(date_str)
                                pattern_bars = []  # Reset pattern after detection
            else:
                # Reset pattern if not consecutive
                pattern_bars = [i]
        else:
            # Reset pattern on non-red candle
            pattern_bars = []
    
    # Create DataFrame from exit signals
    exit_df = pd.DataFrame(exit_signals)
    
    # Sort by exit date if we have multiple signals
    if not exit_df.empty:
        exit_df = exit_df.sort_values('Exit Date')
    
    return all_detected_signal_dates, exit_df

def prepare_chart_data(df):
    """
    Prepare OHLCV dataframe for lightweight charts visualization.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    where 'timestamp' is datetime-like, plus MACD columns
    
    Returns:
        dict: Dictionary containing 'candles', 'volume', 'macd_fast', 'macd_slow', 'macd_hist' JSON data
    """
    try:
        # Find MACD columns in the dataframe
        macd_cols = {
            'macd': [col for col in df.columns if col.startswith('MACD_') and not col.startswith('MACDh_') and not col.startswith('MACDs_')],
            'signal': [col for col in df.columns if col.startswith('MACDs_')],
            'hist': [col for col in df.columns if col.startswith('MACDh_')]
        }
        
        # If we can't find the expected columns, try to find any MACD-related columns
        if not all(macd_cols.values()):
            all_macd_cols = [col for col in df.columns if 'MACD' in col.upper()]
            if len(all_macd_cols) >= 3:
                macd_cols = {
                    'macd': [col for col in all_macd_cols if 'MACD_' in col and 'MACDh_' not in col and 'MACDs_' not in col],
                    'signal': [col for col in all_macd_cols if 'MACDs_' in col],
                    'hist': [col for col in all_macd_cols if 'MACDh_' in col]
                }
        
        # If still no columns found, return empty data
        if not all(macd_cols.values()):
            st.warning("Could not find all required MACD columns in the data")
            return {
                'candles': [],
                'volume': [],
                'macd_fast': [],
                'macd_slow': [],
                'macd_hist': '[]'
            }
            
        # Use the first found columns of each type
        macd_col = macd_cols['macd'][0]
        signal_col = macd_cols['signal'][0]
        hist_col = macd_cols['hist'][0]
        
        # Keep only needed columns and ensure we have a datetimelike timestamp
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', macd_col, signal_col, hist_col]
        df = df[[col for col in required_cols if col in df.columns]].copy()
        
        # For the chart, provide a string 'time' column as originally expected
        df['time'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear

        # Replace NaN with None for JSON compatibility
        df.replace({np.nan: None}, inplace=True)
        
        # export to JSON format
        candles = json.loads(df[['time', 'open', 'high', 'low', 'close', 'volume', 'color']].to_json(orient="records"))
        volume = json.loads(df[['time', 'volume']].rename(columns={"volume": "value"}).to_json(orient="records"))
        
        # Handle MACD data
        macd_fast = json.loads(df[['time', macd_col]].rename(columns={macd_col: "value"}).to_json(orient="records"))
        macd_slow = json.loads(df[['time', signal_col]].rename(columns={signal_col: "value"}).to_json(orient="records"))
        
        # Enhanced MACD Histogram Coloring
        COLOR_STRONG_GREEN = 'rgba(38, 166, 154, 0.9)'
        COLOR_LESSER_GREEN = 'rgba(38, 166, 154, 0.4)'
        COLOR_STRONG_RED = 'rgba(239, 83, 80, 0.9)'
        COLOR_LESSER_RED = 'rgba(239, 83, 80, 0.4)'

        histogram_data = []

        for i in range(len(df)):
            current_val = df[hist_col].iloc[i] if hist_col in df.columns else None

            # Default color if current_val is None
            color = COLOR_STRONG_GREEN if current_val is not None and current_val >= 0 else COLOR_STRONG_RED

            if i > 0 and current_val is not None:
                previous_val = df[hist_col].iloc[i-1] if hist_col in df.columns else None
                if previous_val is not None and not pd.isna(previous_val):
                    if current_val >= 0:
                        color = COLOR_STRONG_GREEN if current_val > previous_val else COLOR_LESSER_GREEN
                    else:
                        color = COLOR_STRONG_RED if abs(current_val) > abs(previous_val) else COLOR_LESSER_RED
            
            histogram_data.append({
                'time': df['time'].iloc[i],
                'value': current_val,
                'color': color
            })

        macd_hist = json.dumps(histogram_data)
        
        return {
            'candles': candles,
            'volume': volume,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_hist': macd_hist
        }
        
    except Exception as e:
        st.error(f"Error preparing chart data: {str(e)}")
        return {
            'candles': [],
            'volume': [],
            'macd_fast': [],
            'macd_slow': [],
            'macd_hist': '[]'
        }


def create_chart_config(candles, volume, macd_fast, macd_slow, macd_hist, ticker_symbol='', buy_signal_dates=None, sell_signal_dates=None, exit_signal_dates=None, second_buy_signals_dates=None, second_sell_signals_dates=None):
    """
    Create complete lightweight charts configuration for multipane display.
    
    Args:
        candles: Candlestick data JSON
        volume: Volume data JSON
        macd_fast: MACD histogram data JSON
        macd_slow: MACD signal line data JSON
        macd_hist: MACD line data JSON
        ticker_symbol: Stock ticker symbol for watermark
    
    Returns:
        list: Chart configuration ready for renderLightweightCharts
    """
    chartMultipaneOptions = [
        {
            "width": 1000,
            "height": 600,
            "layout": {
                "background": {"type": "solid", "color": 'white'},
                "textColor": "black"
            },
            "grid": {
                "vertLines": {"color": "rgba(197, 203, 206, 0.5)"},
                "horzLines": {"color": "rgba(197, 203, 206, 0.5)"}
            },
            "crosshair": {
                "mode": 1,
                "vertLine": {
                    "visible": True,
                    "labelVisible": True,
                    "labelBackgroundColor": 'rgba(0, 120, 212, 0.8)'
                },
                "horzLine": {
                    "visible": True,
                    "labelVisible": True,
                    "labelBackgroundColor": 'rgba(0, 120, 212, 0.8)'
                }
            },
            "priceScale": {"borderColor": "rgba(197, 203, 206, 0.8)"},
            "timeScale": {
                "borderColor": "rgba(197, 203, 206, 0.8)",
                "barSpacing": 15,
                "timeVisible": True,
                "rightOffset": 1,
                "secondsVisible": False
            },
            "watermark": {
                "visible": True,
                "fontSize": 48,
                "horzAlign": 'center',
                "vertAlign": 'center',
                "color": 'rgba(171, 71, 188, 0.3)',
                "text": f'{ticker_symbol} - D1' if ticker_symbol else 'Stock - D1',
            }
        },
        {
            "width": 1000,
            "height": 150,
            "layout": {
                "background": {"type": 'solid', "color": 'transparent'},
                "textColor": 'black',
            },
            "grid": {
                "vertLines": {"color": 'rgba(42, 46, 57, 0)'},
                "horzLines": {"color": 'rgba(42, 46, 57, 0.6)'}
            },
            "timeScale": {
                "visible": False
            },
            "crosshair": {
                "mode": 1,
                "vertLine": {
                    "visible": True,
                    "labelVisible": True,
                    "labelBackgroundColor": 'rgba(0, 120, 212, 0.8)'
                },
                "horzLine": {
                    "visible": True,
                    "labelVisible": True,
                    "labelBackgroundColor": 'rgba(0, 120, 212, 0.8)'
                }
            },
            "watermark": {
                "visible": True,
                "fontSize": 18,
                "horzAlign": 'left',
                "vertAlign": 'top',
                "color": 'rgba(171, 71, 188, 0.7)',
                "text": 'Volume',
            }
        },
        {
            "width": 1000,
            "height": 300,
            "layout": {
                "background": {"type": "solid", "color": 'white'},
                "textColor": "black"
            },
            "timeScale": {
                "visible": False
            },
            "crosshair": {
                "mode": 1,
                "vertLine": {
                    "visible": True,
                    "labelVisible": True,
                    "labelBackgroundColor": 'rgba(0, 120, 212, 0.8)'
                },
                "horzLine": {
                    "visible": True,
                    "labelVisible": True,
                    "labelBackgroundColor": 'rgba(0, 120, 212, 0.8)'
                }
            },
            "watermark": {
                "visible": True,
                "fontSize": 18,
                "horzAlign": 'left',
                "vertAlign": 'center',
                "color": 'rgba(171, 71, 188, 0.7)',
                "text": 'MACD',
            }
        }
    ]

    markers = []
    if buy_signal_dates:
        for date in buy_signal_dates:
            markers.append({"time": date.split(' ')[0], "position": 'belowBar', "color": '#26a69a', "shape": 'arrowUp', "text": 'Buy'})
    if sell_signal_dates:
        for date in sell_signal_dates:
            markers.append({"time": date.split(' ')[0], "position": 'aboveBar', "color": '#ef5350', "shape": 'arrowDown', "text": 'Sell'})
    if exit_signal_dates:
        for date in exit_signal_dates:
            markers.append({"time": date.split(' ')[0], "position": 'aboveBar', "color": '#000000', "shape": 'square', "text": 'Exit'})
    if second_buy_signals_dates:
        for date in second_buy_signals_dates:
            markers.append({"time": date.split(' ')[0], "position": 'belowBar', "color": '#66a69a', "shape": 'arrowUp', "text": '2Buy'})
    if second_sell_signals_dates:
        for date in second_sell_signals_dates:
            # markers.append({"time": date.split(' ')[0], "position": 'aboveBar', "color": '#66a69a', "shape": 'arrowDown', "text": '2Sell'})
            markers.append({"time": date.split(' ')[0], "position": 'aboveBar', "color": '#66a69a', "shape": 'arrowDown', "text": '2Sell'})

    seriesCandlestickChart = [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": COLOR_BULL,
                "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL,
                "wickDownColor": COLOR_BEAR
            },
            "markers": markers
        }
    ]

    seriesVolumeChart = [
        {
            "type": 'Histogram',
            "data": volume,
            "options": {
                "priceFormat": {"type": 'volume'},
                "priceScaleId": ""
            },
            "priceScale": {
                "scaleMargins": {"top": 0, "bottom": 0},
                "alignLabels": False
            }
        }
    ]

    seriesMACDchart = [
        {
            "type": 'Line',
            "data": macd_fast,
            "options": {
                "color": 'blue',
                "lineWidth": 2,
                "crosshairMarkerVisible": True,
                "crosshairMarkerRadius": 4
            },
            "priceScaleId": 'left'
        },
        {
            "type": 'Line',
            "data": macd_slow,
            "options": {
                "color": 'green',
                "lineWidth": 2,
                "crosshairMarkerVisible": True,
                "crosshairMarkerRadius": 4
            },
            "priceScaleId": 'left'
        },
        {
            "type": 'Histogram',
            "data": json.loads(macd_hist),
            "options": {
                "lineWidth": 1,
                "crosshairMarkerVisible": True,
                "crosshairMarkerRadius": 4
            },
            "priceScaleId": 'left',
            "markers": markers
        }
    ]

    return [
        {"chart": chartMultipaneOptions[0], "series": seriesCandlestickChart},
        {"chart": chartMultipaneOptions[1], "series": seriesVolumeChart},
        {"chart": chartMultipaneOptions[2], "series": seriesMACDchart}
    ]


# Main execution block - only run when script is executed directly
if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # --- Sidebar for Controls ---
    st.sidebar.title("Chart Controls")
    
    # Date Range Selection with timezone handling
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = st.sidebar.date_input(
        "Start Date",
        value=end_date - timedelta(days=365),
        max_value=end_date - timedelta(days=1)
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=end_date,
        min_value=start_date + timedelta(days=1),
        max_value=end_date
    )

    # Convert date inputs to datetime with proper timezone handling
    start_date_dt = pd.Timestamp(start_date).normalize()
    end_date_dt = pd.Timestamp(end_date).normalize()

    # Timeframe Selection
    timeframe_map = {
        '1 Day': TimeFrame.Day,
        '1 Hour': TimeFrame.Hour,
        '4 Hours': TimeFrame.Hour,
        '1 Week': TimeFrame.Week
    }
    selected_timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=list(timeframe_map.keys()),
        index=0
    )
    
    # MACD Parameters
    st.sidebar.subheader("MACD Parameters")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        fast_period = st.number_input("Fast", min_value=1, max_value=50, value=12, step=1)
    with col2:
        slow_period = st.number_input("Slow", min_value=1, max_value=100, value=26, step=1)
    with col3:
        signal_period = st.number_input("Signal", min_value=1, max_value=50, value=9, step=1)
    
    # Signal Settings
    st.sidebar.subheader("Signal Settings")
    show_buy_signals = st.sidebar.checkbox("Show Buy Signals", value=True)
    show_second_buy_signals = st.sidebar.checkbox("Show Second Buy Signals", value=True)
    show_sell_signals = st.sidebar.checkbox("Show Sell Signals", value=True)
    show_second_sell_signals = st.sidebar.checkbox("Show Second Sell Signals", value=True)
    show_exit_signals = st.sidebar.checkbox("Show Exit Signals", value=True)
    consecutive_buy_bars = st.sidebar.slider("Consecutive Buy Bars", 1, 10, 6)
    consecutive_sell_bars = st.sidebar.slider("Consecutive Sell Bars", 1, 10, 5)
    consecutive_exit_bars = st.sidebar.slider("Consecutive Exit Bars", 1, 10, 3)

    # Stock Selection
    st.sidebar.subheader("Stock Selection")
    try:
        symbols_df = pd.read_csv('../data/most_active.csv')
        symbols_df = symbols_df[symbols_df['close'] > 5]
        symbols = symbols_df['name'].tolist()
    except FileNotFoundError:
        st.sidebar.error("Error: '../data/most_active.csv' not found.")
        symbols = ['AAPL', 'GOOG', 'MSFT']

    selected_symbol = st.sidebar.selectbox("Select Stock:", symbols, index=0)
    st.sidebar.markdown("""
    **Signal Types:**
    - 🟢 Buy: Lesser red MACD bars (decreasing negative)
    - 🟡 2nd Buy: Lesser green MACD bars (decreasing positive)
    - 🔴 Sell: Lesser green MACD bars (decreasing positive)
    - ⬛ Exit: Consecutive red candles with decreasing closes
    """)
    

    st.subheader(f"Multipane Chart for {selected_symbol}")

    # --- Data Fetching and Charting ---
    if selected_symbol:
        try:
            # Calculate date range in days using timezone-aware timestamps
            date_range_days = (end_date_dt - start_date_dt).days
            
            # Calculate number of days based on timeframe
            if selected_timeframe == '1 Day':
                n_days = date_range_days
            elif selected_timeframe == '1 Hour':
                n_days = min(date_range_days * 5, 30)  # Approximate trading days in a month
            elif selected_timeframe == '4 Hours':
                n_days = min(date_range_days * 5, 90)  # Approximate trading days in 3 months
            else:  # 1 Week
                n_days = date_range_days

            # Convert dates to strings for the API call
            start_date_str = start_date_dt.strftime("%Y-%m-%d")
            end_date_str = end_date_dt.strftime("%Y-%m-%d")
            
            df = fetch_stock_data(
                symbols=selected_symbol,
                start_date=start_date_str,
                end_date=end_date_str,
                timeframe=timeframe_map[selected_timeframe],
                n_days=n_days
            )

            if df is None or df.empty:
                st.warning(f"No data found for {selected_symbol}. Please select another symbol.")
            else:
                # Ensure proper timezone handling for timestamps
                df = df.reset_index().rename(columns={'index': 'timestamp', **{col: col.lower() for col in df.columns}})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # If timestamp has timezone info, standardize it
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')

                # Set normalized timestamp as index for TA calculations
                df.set_index('timestamp', inplace=True)
                
                # Define MACD column names
                macd_col = f"MACD_{fast_period}_{slow_period}_{signal_period}"
                macds_col = f"MACDs_{fast_period}_{slow_period}_{signal_period}"
                macdh_col = f"MACDh_{fast_period}_{slow_period}_{signal_period}"
                
                # Check if MACD columns already exist to avoid duplicates
                existing_cols = set(df.columns)
                new_macd_cols = {macd_col, macds_col, macdh_col}
                
                # Only calculate MACD if not all columns exist
                if not new_macd_cols.issubset(existing_cols):
                    # Drop any existing MACD columns to prevent duplicates
                    df = df.drop(columns=list(new_macd_cols.intersection(existing_cols)), errors='ignore')
                    
                    # Calculate MACD
                    macd_result = df.ta.macd(
                        close='close',
                        fast=fast_period,
                        slow=slow_period,
                        signal=signal_period,
                        append=True
                    )

                    # Ensure the MACD columns exist in the dataframe
                    if macd_result is not None:
                        if isinstance(macd_result, pd.DataFrame):
                            # Only keep columns that don't already exist
                            new_cols = [col for col in macd_result.columns if col not in df.columns]
                            if new_cols:
                                df = pd.concat([df, macd_result[new_cols]], axis=1)
                        elif isinstance(macd_result, pd.Series) and macd_result.name not in df.columns:
                            df[macd_result.name] = macd_result
                
                # Verify MACD columns exist
                missing_cols = [col for col in [macd_col, macds_col, macdh_col] if col not in df.columns]
                if missing_cols:
                    st.error(f"Missing MACD columns: {', '.join(missing_cols)}. Available columns: {', '.join(df.columns)}")
                    st.stop()
                
                # Initialize signals lists
                buy_signals = []
                second_buy_signals = []
                sell_signals = []
                exit_signals = []

                # Get signals with user-defined consecutive bars
                if show_buy_signals:
                    buy_signals = get_macd_lesser_red_buy_signals(
                        df, 
                        consecutive_bars=consecutive_buy_bars,
                        hist_col=macdh_col
                    )
                    if buy_signals:
                        st.success(f"Found {len(buy_signals)} buy signals")


                if show_sell_signals:
                    sell_signals = get_macd_lesser_green_sell_signals(
                        df, 
                        consecutive_bars=consecutive_sell_bars,
                        hist_col=macdh_col
                    )
                    if sell_signals:
                        st.warning(f"Found {len(sell_signals)} sell signals")
                
                if show_exit_signals:
                    exit_signals, exit_hist= get_consecutive_red_candles_exit(
                        df,
                        consecutive_bars=consecutive_exit_bars,
                        buy_signal_dates=buy_signals
                    )
                    if exit_signals:
                        st.info(f"Found {len(exit_signals)} exit signals")
                        
                if show_second_buy_signals:
                    second_buy_signals, _ = get_second_buy_entry(
                        df,
                        consecutive_bars=consecutive_buy_bars,
                        buy_signal_dates=buy_signals,
                        hist_col=macdh_col
                    )
                    if second_buy_signals:
                        st.success(f"Found {len(second_buy_signals)} second buy signals")

                # TODO: Add second sell signals
                if show_second_sell_signals:
                    second_sell_signals = get_macd_lesser_green_sell_signals(
                        df,
                        consecutive_bars=consecutive_sell_bars,
                        # sell_signal_dates=sell_signals,
                        hist_col=macdh_col
                    )
                    if second_sell_signals:
                        st.warning(f"Found {len(second_sell_signals)} second sell signals")
                # TODO: Add exit signals
                # Reset index for chart preparation
                df = df.reset_index()

                chart_data = prepare_chart_data(df)
                chart_config = create_chart_config(
                    candles=chart_data['candles'],
                    volume=chart_data['volume'],
                    macd_fast=chart_data['macd_fast'],
                    macd_slow=chart_data['macd_slow'],
                    macd_hist=chart_data['macd_hist'],
                    ticker_symbol=selected_symbol,
                    buy_signal_dates=buy_signals,
                    sell_signal_dates=sell_signals,
                    exit_signal_dates=exit_signals,
                    second_buy_signals_dates=second_buy_signals,
                    second_sell_signals_dates=second_sell_signals
                )
                # Render the charts
                renderLightweightCharts(chart_config, 'multipane')
                
                # Add a section to display the data in a table
                st.subheader("Data Table")
                
                # Create a copy of the dataframe for display
                display_df = df.copy()
                
                # Format the timestamp for better readability
                if not display_df.empty and 'timestamp' in display_df.columns:
                    display_df['Date'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    display_df = display_df.drop(columns=['timestamp'])
                
                # Reorder columns to put Date first
                cols = ['Date'] + [col for col in display_df.columns if col != 'Date']
                display_df = display_df[cols]
                
                # Display the data table with pagination
                st.dataframe(
                    display_df,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(
                            "Date & Time",
                            format="YYYY-MM-DD HH:mm"
                        )
                    },
                    width="stretch",
                    hide_index=True,
                    height=400
                )
                
                # Add download button for the data
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"{selected_symbol}_data.csv",
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred while fetching or processing data for {selected_symbol}: {e}")
            st.exception(e)  # Show full traceback for debugging