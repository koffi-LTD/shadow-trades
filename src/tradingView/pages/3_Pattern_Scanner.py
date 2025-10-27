import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas_ta as ta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alpaca.historical import fetch_stock_data as _fetch_stock_data
from alpaca.data.timeframe import TimeFrame
from src.tradingView.streamlitApp import (
    get_macd_lesser_red_buy_signals,
    get_macd_lesser_green_sell_signals,
    get_consecutive_red_candles_exit,
    prepare_chart_data,
    create_chart_config,
)
from streamlit_lightweight_charts import renderLightweightCharts

# Cached version of fetch_stock_data with custom hashing for TimeFrame
@st.cache_data(
    ttl=3600,
    hash_funcs={TimeFrame: lambda x: str(x)}
)
def fetch_stock_data(symbols, start_date, end_date, _timeframe):
    """Cached version of fetch_stock_data with TTL of 1 hour."""
    return _fetch_stock_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=_timeframe
    )

def apply_signals_to_data(data, consecutive_buy_bars=6, consecutive_sell_bars=5, consecutive_exit_bars=3):
    """
    Apply buy, sell, and exit signals to the data and return the results.
    
    Args:
        data (pd.DataFrame): Input price data with MACD calculated
        consecutive_buy_bars (int): Number of consecutive bars for buy signal
        consecutive_sell_bars (int): Number of consecutive bars for sell signal
        consecutive_exit_bars (int): Number of consecutive red candles for exit signal
        
    Returns:
        dict: Dictionary containing buy_dates, sell_dates, and exit_dfs
    """
    # Initialize result dictionary
    result = {
        'buy_dates': [],
        'sell_dates': [],
        'exit_dfs': []
    }
    
    # Find MACD histogram column
    hist_col = next((col for col in data.columns if col.startswith('MACDh_')), None)
    if hist_col is None:
        return result
    
    # 1. Get buy signals
    result['buy_dates'] = get_macd_lesser_red_buy_signals(
        data, 
        consecutive_bars=consecutive_buy_bars,
        hist_col=hist_col
    )
    
    # 2. Get sell signals
    result['sell_dates'] = get_macd_lesser_green_sell_signals(
        data,
        consecutive_bars=consecutive_sell_bars,
        hist_col=hist_col
    )
    
    # 3. Get exit signals (only if we have buy signals)
    if result['buy_dates']:
        exit_dates, exit_df = get_consecutive_red_candles_exit(
            data,
            consecutive_bars=consecutive_exit_bars,
            hist_col=hist_col,
            buy_signal_dates=result['buy_dates']
        )
        result['exit_dfs'].append(exit_df)
    
    return result

def filter_results_by_period(signals, start_date, end_date):
    """
    Filter signals to only include those within the specified date range.
    
    Args:
        signals (dict): Dictionary containing buy_dates, sell_dates, and exit_dfs
        start_date (datetime): Start of the period to filter by
        end_date (datetime): End of the period to filter by
        
    Returns:
        dict: Filtered signals with the same structure as input
    """
    if not isinstance(signals, dict):
        return {'buy_dates': [], 'sell_dates': [], 'exit_dfs': []}
    
    # Safely get lists with defaults if keys don't exist
    buy_dates = signals.get('buy_dates', [])
    sell_dates = signals.get('sell_dates', [])
    exit_dfs = signals.get('exit_dfs', [])
    
    # Filter dates that fall within the specified range
    filtered = {
        'buy_dates': [
            d for d in buy_dates 
            if d and start_date <= pd.to_datetime(d).date() <= end_date
        ],
        'sell_dates': [
            d for d in sell_dates 
            if d and start_date <= pd.to_datetime(d).date() <= end_date
        ],
        'exit_dfs': []
    }
    
    # Filter exit dataframes
    for df in exit_dfs:
        try:
            # If it's a list, convert to DataFrame first
            if isinstance(df, list):
                if not df:  # Skip empty lists
                    continue
                df = pd.DataFrame(df)
                
            # If it's a DataFrame and not empty
            if hasattr(df, 'empty') and not df.empty:
                # Make sure 'Exit Date' exists and is datetime
                if 'Exit Date' in df.columns:
                    df['Exit Date'] = pd.to_datetime(df['Exit Date'])
                    mask = (df['Exit Date'].dt.date >= start_date) & \
                           (df['Exit Date'].dt.date <= end_date)
                    filtered_df = df[mask]
                    if not filtered_df.empty:
                        filtered['exit_dfs'].append(filtered_df)
        except Exception as e:
            print(f"Error filtering exit signals: {str(e)}")
            continue
    
    return filtered


# Function to process a single stock
def process_stock(symbol, start_date, end_date, fast_period, slow_period, signal_period, 
                 consecutive_buy_bars, consecutive_sell_bars, consecutive_exit_bars):
    try:
        # Convert dates to string format expected by fetch_stock_data
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        extended_start_date = (start_date - timedelta(days=365)).strftime('%Y-%m-%d')
        
        try:
            # Fetch data for the symbol with extended date range
            df = fetch_stock_data(
                symbols=symbol,
                start_date=extended_start_date,
                end_date=end_date_str,
                _timeframe=TimeFrame.Day
            )
            
            if df is None:
                return None
                
            # If we got a list, try to convert it to a DataFrame
            if isinstance(df, list):
                df = pd.DataFrame(df)
                
            # Check if DataFrame is empty
            if hasattr(df, 'empty') and df.empty:
                return None
                
            # Clean and prepare the data
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index(drop=False)
                
                # Try to find a datetime column
                datetime_col = None
                for col in ['timestamp', 'date', 'time', 'datetime']:
                    if col in df.columns:
                        datetime_col = col
                        break
                        
                if datetime_col is not None:
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                else:
                    # If no datetime column found, use the first column that looks like a date
                    for col in df.select_dtypes(include=['object', 'datetime64']).columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df.set_index(col, inplace=True)
                            break
                        except:
                            continue
            
            # Ensure we have a valid datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                return None
                
            # Ensure index is timezone-naive and sorted
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            df = df.sort_index()
            
            # Ensure we have the required columns
            if 'close' not in df.columns:
                # Try to find a close price column
                close_cols = [c for c in df.columns if 'close' in c.lower()]
                if not close_cols:
                    return None
                df = df.rename(columns={close_cols[0]: 'close'})
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            return None
        
        # Calculate MACD
        macd = ta.macd(df['close'], fast=fast_period, slow=slow_period, signal=signal_period)
        df = pd.concat([df, macd], axis=1)
        
        # Apply signals
        signals = apply_signals_to_data(
            df,
            consecutive_buy_bars=consecutive_buy_bars,
            consecutive_sell_bars=consecutive_sell_bars,
            consecutive_exit_bars=consecutive_exit_bars
        )
        
        # Filter by date range
        filtered_signals = filter_results_by_period(signals, start_date, end_date)
        
        # Prepare results
        exit_dfs = filtered_signals.get('exit_dfs', [])
        exit_count = len(exit_dfs[0]) if exit_dfs and isinstance(exit_dfs[0], pd.DataFrame) and not exit_dfs[0].empty else 0
        
        # Get second buy entries
        second_buy_dates, second_buy_entries = get_second_buy_entry(
            df,
            filtered_signals.get('buy_dates', []),
            consecutive_bars=3
        )
        
        # Track all signals with their dates and prices
        all_signals = []
        current_price = df['close'].iloc[-1] if not df.empty else None
        
        # Process buy signals with prices
        for date in filtered_signals.get('buy_dates', []):
            signal_date = pd.to_datetime(date).date()
            signal_price = df[df.index.date == signal_date]['close'].iloc[0] if not df[df.index.date == signal_date].empty else None
            if signal_price is not None:
                all_signals.append((signal_date, 'BUY', signal_price))
        
        # Process second buy entries
        if not second_buy_entries.empty:
            for _, row in second_buy_entries.iterrows():
                signal_date = row['Second Entry Date'].date()
                signal_price = row['Second Entry Price']
                all_signals.append((signal_date, 'SECOND_BUY', signal_price))
                
        # Process sell signals with prices
        for date in filtered_signals.get('sell_dates', []):
            signal_date = pd.to_datetime(date).date()
            signal_price = df[df.index.date == signal_date]['close'].iloc[0] if not df[df.index.date == signal_date].empty else None
            if signal_price is not None:
                all_signals.append((signal_date, 'SELL', signal_price))
                
        # Process exit signals with prices
        for exit_df in exit_dfs:
            if isinstance(exit_df, pd.DataFrame) and not exit_df.empty:
                for _, row in exit_df.iterrows():
                    exit_date = pd.to_datetime(row['Exit Date']).date()
                    exit_price = row.get('Exit Price', row.get('close', None))
                    if exit_price is not None:
                        all_signals.append((exit_date, 'EXIT', exit_price))
        
        # Calculate gain/loss for the latest signal
        gain_loss = None
        latest_signal_type = None
        latest_date = None
        signal_price = None
        days_since_signal = None
        
        if all_signals:
            # Sort signals by date in descending order and get the most recent one
            all_signals_sorted = sorted(all_signals, key=lambda x: x[0], reverse=True)
            latest_signal = all_signals_sorted[0]  # Get the most recent signal
            latest_date, latest_signal_type, signal_price = latest_signal
            
            if signal_price and signal_price > 0 and current_price is not None:
                gain_loss = ((current_price - signal_price) / signal_price) * 100
                days_since_signal = (pd.Timestamp.today().date() - latest_date).days
        
        # Count signals by type
        signal_counts = {'BUY': 0, 'SECOND_BUY': 0, 'SELL': 0, 'EXIT': 0}
        for _, signal_type, _ in all_signals:
            signal_counts[signal_type] += 1
                
        result = {
            'Symbol': symbol,
            'Buy Signals': signal_counts['BUY'],
            'Second Buy Signals': signal_counts['SECOND_BUY'],
            'Sell Signals': signal_counts['SELL'],
            'Exit Signals': signal_counts['EXIT'],
            'Last Close': f"${current_price:.2f}" if current_price is not None else "N/A",
            'Latest Signal': latest_signal_type if all_signals else None,
            'Signal Date': latest_date if all_signals else None,
            'Signal Price': f"${signal_price:.2f}" if signal_price is not None else "N/A",
            'Gain/Loss %': gain_loss,
            'Days Since Signal': days_since_signal if days_since_signal is not None else "N/A"
        }
        
        return result
        
    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
        return None

    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
        return None

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="Stock Pattern Scanner",
        page_icon="📈",
        layout="wide"
    )
    
    # Sidebar controls
    st.sidebar.title("🔍 Scanner Settings")
    
    # Date range
    st.sidebar.subheader("Date Range")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)  # Default to 30 days
    
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(start_date, end_date),
        help="Select the date range for the scan (up to 1 year)"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    
    # MACD Parameters
    st.sidebar.subheader("MACD Settings")
    
    with st.sidebar.expander("MACD Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fast_period = st.number_input("Fast EMA", min_value=1, max_value=50, value=12, 
                                       help="Fast EMA period (typically 12)")
        with col2:
            slow_period = st.number_input("Slow EMA", min_value=1, max_value=100, value=26, 
                                       help="Slow EMA period (typically 26)")
        with col3:
            signal_period = st.number_input("Signal", min_value=1, max_value=50, value=9, 
                                         help="Signal line period (typically 9)")
    
    # Signal Detection
    with st.sidebar.expander("Signal Detection"):
        st.caption("Adjust sensitivity for pattern detection")
        consecutive_buy_bars = st.slider("Buy Pattern Bars", 1, 10, 6, 
                                       help="Consecutive 'lesser red' bars for buy signal")
        consecutive_sell_bars = st.slider("Sell Pattern Bars", 1, 10, 5,
                                        help="Consecutive 'lesser green' bars for sell signal")
        consecutive_exit_bars = st.slider("Exit Pattern Bars", 1, 10, 3,
                                        help="Consecutive red candles for exit signal")
    
    # Load stock universe
    try:
        symbols_df = pd.read_csv('../data/most_active.csv') # TODO: Change to your own universe file
        # limit 10 symbols
        # TOP 200 by volume
        symbols_df = symbols_df.sort_values(by='volume', ascending=False)
        symbols = symbols_df['name'].tolist()[:200]
    except FileNotFoundError:
        st.warning("Could not load stock universe file. Using default symbols.")
        symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC']
    
    # Main content
    st.title("📈 Stock Pattern Scanner")
    st.write(f"Scanning {len(symbols)} stocks for patterns between {start_date} and {end_date}")
    
    # Scan for patterns when user clicks the button
    if st.sidebar.button("🚀 Start Scanning", use_container_width=True):
        # Clear previous results
        all_results = []
        
        # Create a placeholder for the results
        results_placeholder = st.empty()
        
        # Initialize progress
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Process each stock
        for idx, symbol in enumerate(symbols):
            progress = (idx + 1) / len(symbols)
            progress_placeholder.markdown(
                f'<p>Processing {symbol} ({idx + 1}/{len(symbols)})</p>',
                unsafe_allow_html=True
            )
            progress_bar.progress(progress)
            
            # Process the stock
            result = process_stock(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                consecutive_buy_bars=consecutive_buy_bars,
                consecutive_sell_bars=consecutive_sell_bars,
                consecutive_exit_bars=consecutive_exit_bars
            )
            
            if result is not None:
                all_results.append(result)
              
            # Update the results table after each stock is processed
            if all_results:
                results_df = pd.DataFrame(all_results)
                
                # Ensure all required columns exist before calculating total signals
                signal_columns = ['Buy Signals', 'Sell Signals', 'Exit Signals']
                if all(col in results_df.columns for col in signal_columns):
                    results_df['Total Signals'] = results_df[signal_columns].sum(axis=1)
                    sort_column = 'Total Signals'
                else:
                    # If any signal column is missing, sort by first available signal column
                    sort_column = next((col for col in signal_columns if col in results_df.columns), 'Symbol')
                
                # Sort the results
                results_df = results_df.sort_values(sort_column, ascending=False)
                
                # Select only columns that exist in the DataFrame
                display_columns = ['Symbol', 'Last Close', 'Buy Signals', 'Sell Signals', 
                                 'Exit Signals', 'Total Signals', 'Latest Signal', 'Signal Date']
                available_columns = [col for col in display_columns if col in results_df.columns]
                
                # Define display columns with the new gain/loss information
                display_columns = [
                    'Symbol', 'Latest Signal', 'Signal Date', 'Signal Price',
                    'Last Close', 'Gain/Loss %', 'Buy Signals', 'Sell Signals', 'Exit Signals'
                ]
                
                # Ensure all required columns exist in the results
                available_columns = [col for col in display_columns if col in results_df.columns]
                
                # Create a display version of the gain/loss with percentage sign
                if 'Gain/Loss %' in results_df.columns:
                    results_df['Gain/Loss Display'] = results_df['Gain/Loss %'].apply(
                        lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
                    )
                    # Sort by gain/loss percentage
                    results_df = results_df.sort_values('Gain/Loss %', ascending=False)
                
                # Define columns to display
                display_columns = [
                    'Symbol', 'Latest Signal', 'Signal Date', 'Signal Price',
                    'Last Close', 'Gain/Loss Display', 'Days Since Signal',
                    'Buy Signals', 'Sell Signals', 'Exit Signals'
                ]
                available_columns = [col for col in display_columns if col in results_df.columns]
                
                # Display the results in a nice table
                results_placeholder.dataframe(
                    results_df[available_columns],
                    column_config={
                        'Symbol': st.column_config.TextColumn('Symbol', width='small'),
                        'Last Close': st.column_config.TextColumn('Last Close'),
                        'Signal Price': st.column_config.TextColumn('Signal Price'),
                        'Gain/Loss Display': st.column_config.TextColumn(
                            'G/L %',
                            help='Percentage gain/loss since the last signal'
                        ),
                        'Days Since Signal': st.column_config.NumberColumn('Days'),
                        'Buy Signals': st.column_config.NumberColumn('Buy', width='small'),
                        'Sell Signals': st.column_config.NumberColumn('Sell', width='small'),
                        'Exit Signals': st.column_config.NumberColumn('Exit', width='small'),
                        'Latest Signal': st.column_config.TextColumn('Signal', width='small'),
                        'Signal Date': st.column_config.DateColumn('Date')
                    },
                    use_container_width=True,
                    hide_index=True
                )
        
        # Clear progress indicators when done
        progress_placeholder.empty()
        progress_bar.empty()
        
        # Show completion message
        if all_results:
            st.success(f"✅ Scan complete! Processed {len(symbols)} stocks. "
                      f"Found {len([r for r in all_results if r['Total Signals'] > 0])} stocks with signals.")
            
            # Add download button for results
            results_df = pd.DataFrame(all_results)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f'stock_scan_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )
            
            # Show detailed view for selected stock
            selected_symbol = st.selectbox("Select a stock to view details:", [""] + sorted(results_df['Symbol'].unique()))
            
            if selected_symbol:
                st.subheader(f"📊 {selected_symbol} - Detailed View")
                
                # Get the stock data
                stock_data = next((r for r in all_results if r['Symbol'] == selected_symbol), None)
                
                if stock_data is not None:
                    st.write(f"**Last Close:** ${stock_data['Last Close']:.2f}")
                    st.write(f"**Latest Signal:** {stock_data['Latest Signal']} on {stock_data['Signal Date']}")
                    st.write(f"**Buy Signals:** {stock_data['Buy Signals']} | "
                            f"**Sell Signals:** {stock_data['Sell Signals']} | "
                            f"**Exit Signals:** {stock_data['Exit Signals']}")
        else:
            st.info("No stocks with signals found in the selected period.")

if __name__ == "__main__":
    main()