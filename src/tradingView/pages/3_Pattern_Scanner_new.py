import asyncio
from concurrent.futures import ThreadPoolExecutor
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
@st.cache_data(ttl=3600, hash_funcs={TimeFrame: lambda x: str(x)})
def fetch_stock_data(symbols, start_date, end_date, _timeframe):
    return _fetch_stock_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=_timeframe,
    )

def apply_signals_to_data(data, consecutive_buy_bars=6, consecutive_sell_bars=5, consecutive_exit_bars=3):
    result = {'buy_dates': [], 'sell_dates': [], 'exit_dfs': []}
    hist_col = next((col for col in data.columns if col.startswith('MACDh_')), None)
    if hist_col is None:
        return result

    result['buy_dates'] = get_macd_lesser_red_buy_signals(data, consecutive_bars=consecutive_buy_bars, hist_col=hist_col)
    result['sell_dates'] = get_macd_lesser_green_sell_signals(data, consecutive_bars=consecutive_sell_bars, hist_col=hist_col)

    if result['buy_dates']:
        _, exit_df = get_consecutive_red_candles_exit(data, consecutive_bars=consecutive_exit_bars, hist_col=hist_col, buy_signal_dates=result['buy_dates'])
        result['exit_dfs'].append(exit_df)

    return result

def filter_results_by_period(signals, start_date, end_date):
    if not isinstance(signals, dict):
        return {'buy_dates': [], 'sell_dates': [], 'exit_dfs': []}
    
    buy_dates = signals.get('buy_dates', [])
    sell_dates = signals.get('sell_dates', [])
    exit_dfs = signals.get('exit_dfs', [])

    filtered = {
        'buy_dates': [d for d in buy_dates if d and start_date <= pd.to_datetime(d).date() <= end_date],
        'sell_dates': [d for d in sell_dates if d and start_date <= pd.to_datetime(d).date() <= end_date],
        'exit_dfs': []
    }

    for df in exit_dfs:
        try:
            if isinstance(df, list) and df:
                df = pd.DataFrame(df)
            if hasattr(df, 'empty') and not df.empty and 'Exit Date' in df.columns:
                df['Exit Date'] = pd.to_datetime(df['Exit Date'])
                mask = (df['Exit Date'].dt.date >= start_date) & (df['Exit Date'].dt.date <= end_date)
                filtered_df = df[mask]
                if not filtered_df.empty:
                    filtered['exit_dfs'].append(filtered_df)
        except Exception as e:
            print(f"Error filtering exit signals: {e}")
            continue

    return filtered

def process_stock(symbol, start_date, end_date, fast_period, slow_period, signal_period, consecutive_buy_bars, consecutive_sell_bars, consecutive_exit_bars):
    try:
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        extended_start_date = (start_date - timedelta(days=365)).strftime('%Y-%m-%d')

        df = fetch_stock_data(symbols=symbol, start_date=extended_start_date, end_date=end_date_str, _timeframe=TimeFrame.Day)
        if df is None or (hasattr(df, 'empty') and df.empty):
            return None
        if isinstance(df, list):
            df = pd.DataFrame(df)

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=False)
            for col in ['timestamp', 'date', 'time', 'datetime']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break

        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        if 'close' not in df.columns:
            close_cols = [c for c in df.columns if 'close' in c.lower()]
            if not close_cols:
                return None
            df = df.rename(columns={close_cols[0]: 'close'})

        macd = ta.macd(df['close'], fast=fast_period, slow=slow_period, signal=signal_period)
        df = pd.concat([df, macd], axis=1)
        signals = apply_signals_to_data(df, consecutive_buy_bars, consecutive_sell_bars, consecutive_exit_bars)
        filtered_signals = filter_results_by_period(signals, start_date, end_date)

        exit_dfs = filtered_signals.get('exit_dfs', [])
        exit_count = 0
        if exit_dfs and isinstance(exit_dfs[0], pd.DataFrame) and not exit_dfs[0].empty:
            exit_count = len(exit_dfs[0])

        # Calculate gains/loss since last signal
        gain_loss = None
        latest_signal = None
        latest_date = None
        all_dates = []
        
        # Process buy/sell signals
        for signal_type in ['buy_dates', 'sell_dates']:
            dates = filtered_signals.get(signal_type, [])
            if dates:
                for d in dates:
                    signal_date = pd.to_datetime(d).date()
                    signal_price = df[df.index.date == signal_date]['close'].iloc[0] if not df[df.index.date == signal_date].empty else None
                    if signal_price is not None:
                        all_dates.append((signal_date, signal_type.replace('_dates', '').upper(), signal_price))
        
        # Process exit signals
        if filtered_signals.get('exit_dfs') and not filtered_signals['exit_dfs'][0].empty:
            exit_df = filtered_signals['exit_dfs'][0]
            for _, row in exit_df.iterrows():
                exit_date = pd.to_datetime(row['Exit Date']).date()
                exit_price = row['Exit Price'] if 'Exit Price' in row else None
                if exit_price is not None:
                    all_dates.append((exit_date, 'EXIT', exit_price))
        
        # If we have signals, calculate gain/loss for the latest one
        if all_dates:
            latest_date, latest_signal, signal_price = max(all_dates, key=lambda x: x[0])
            current_price = df['close'].iloc[-1]
            if signal_price and signal_price > 0:
                gain_loss = ((current_price - signal_price) / signal_price) * 100
        
        result = {
            'Symbol': symbol,
            'Buy Signals': len(filtered_signals.get('buy_dates', [])),
            'Sell Signals': len(filtered_signals.get('sell_dates', [])),
            'Exit Signals': exit_count,
            'Last Close': df['close'].iloc[-1],
            'Latest Signal': latest_signal,
            'Signal Date': latest_date.date() if latest_date else None,
            'Gain/Loss %': f"{gain_loss:.2f}%" if gain_loss is not None else "N/A",
            'Signal Price': f"${signal_price:.2f}" if 'signal_price' in locals() and signal_price is not None else "N/A"
        }

        return result
    except Exception as e:
        st.error(f"Error processing {symbol}: {str(e)}")
        return None

# Assumes process_stock and supporting functions are imported from your existing code

async def async_scan_stocks(symbols, start_date, end_date, fast_period, slow_period, signal_period,
                           consecutive_buy_bars, consecutive_sell_bars, consecutive_exit_bars,
                           progress_callback=None):
    all_results = []
    total = len(symbols)
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                process_stock,
                symbol,
                start_date,
                end_date,
                fast_period,
                slow_period,
                signal_period,
                consecutive_buy_bars,
                consecutive_sell_bars,
                consecutive_exit_bars,
            ) for symbol in symbols
        ]

        for idx, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await coro
                if result is not None:
                    all_results.append(result)
                if progress_callback:
                    progress_callback((idx + 1) / total, symbols[idx])
            except Exception as e:
                st.error(f"Error processing symbol {symbols[idx]}: {str(e)}")
    return all_results


def main():
    st.set_page_config(page_title="Stock Pattern Scanner (Async)", page_icon="📈", layout="wide")
    st.sidebar.title("🔍 Scanner Settings")

    # Date selection
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    date_range = st.sidebar.date_input("Select date range", value=(start_date, end_date))
    if len(date_range) == 2:
        start_date, end_date = date_range

    # MACD Parameters
    with st.sidebar.expander("MACD Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fast_period = st.number_input("Fast EMA", min_value=1, max_value=50, value=12)
        with col2:
            slow_period = st.number_input("Slow EMA", min_value=1, max_value=100, value=26)
        with col3:
            signal_period = st.number_input("Signal", min_value=1, max_value=50, value=9)

    # Signal detection
    with st.sidebar.expander("Signal Detection"):
        consecutive_buy_bars = st.slider("Buy Bars", 1, 10, 6)
        consecutive_sell_bars = st.slider("Sell Bars", 1, 10, 5)
        consecutive_exit_bars = st.slider("Exit Bars", 1, 10, 3)

    # Symbols
    try:
        symbols_df = pd.read_csv('../data/most_active.csv')
        symbols = symbols_df['name'].tolist()
    except FileNotFoundError:
        st.warning("Default symbols used.")
        symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'AMD', 'INTC']

    st.title("📈 Async Stock Pattern Scanner")
    st.write(f"Scanning {len(symbols)} stocks between {start_date} and {end_date}")

    if st.sidebar.button("🚀 Start Async Scan", use_container_width=True):
        results_placeholder = st.empty()
        progress_placeholder = st.empty()
        progress_bar = st.progress(0)

        def update_progress(progress, current_symbol):
            progress_placeholder.markdown(f"Processing {current_symbol} ({progress*100:.1f}%)")
            progress_bar.progress(progress)

        all_results = asyncio.run(async_scan_stocks(
            symbols,
            start_date,
            end_date,
            fast_period,
            slow_period,
            signal_period,
            consecutive_buy_bars,
            consecutive_sell_bars,
            consecutive_exit_bars,
            progress_callback=update_progress
        ))

        progress_placeholder.empty()
        progress_bar.empty()

        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Convert string percentages to numeric for sorting
            results_df['Gain/Loss Num'] = results_df['Gain/Loss %'].replace('N/A', np.nan).str.rstrip('%').astype(float)
            
            # Sort by gain/loss percentage (descending)
            results_df = results_df.sort_values('Gain/Loss Num', ascending=False)
            
            # Format the display
            display_columns = ['Symbol', 'Latest Signal', 'Signal Date', 'Signal Price', 
                             'Last Close', 'Gain/Loss %', 'Buy Signals', 'Sell Signals', 'Exit Signals']
            
            # Display the results with formatted numbers
            results_placeholder.dataframe(
                results_df[display_columns],
                column_config={
                    'Gain/Loss %': st.column_config.NumberColumn(
                        'Gain/Loss %',
                        format='%.2f%%',
                        help='Percentage gain/loss since the last signal'
                    ),
                    'Last Close': st.column_config.NumberColumn(
                        'Price',
                        format='$%.2f'
                    ),
                    'Signal Price': st.column_config.NumberColumn(
                        'Signal Price',
                        format='$%.2f'
                    )
                },
                use_container_width=True,
                hide_index=True
            )

            # --- Metrics Summary ---
            st.subheader("📊 Summary Metrics")
            total_stocks = len(results_df)
            avg_gain = results_df['Gain/Loss Num'].mean()
            
            # Count profitable vs losing trades
            profitable = (results_df['Gain/Loss Num'] > 0).sum()
            losing = (results_df['Gain/Loss Num'] < 0).sum()
            neutral = len(results_df) - profitable - losing

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Stocks", total_stocks)
            c2.metric("Avg. Gain/Loss %", f"{avg_gain:.2f}%" if not pd.isna(avg_gain) else "N/A")
            c3.metric("Profitable/Losing", f"{profitable} / {losing}")

            st.success(f"✅ Scan complete! Processed {total_stocks} stocks with {total_signals} total signals.")

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f'stock_scan_async_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
            )

            selected_symbol = st.selectbox("Select stock for details", [""] + sorted(results_df['Symbol'].unique()))

            if selected_symbol:
                st.subheader(f"📊 {selected_symbol} - Details")
                stock_data = next((r for r in all_results if r['Symbol'] == selected_symbol), None)
                if stock_data:
                    st.write(f"**Last Close:** ${stock_data['Last Close']:.2f}")
                    st.write(f"**Latest Signal:** {stock_data['Latest Signal']} on {stock_data['Signal Date']}")
                    st.write(f"**Buy Signals:** {stock_data['Buy Signals']} | **Sell Signals:** {stock_data['Sell Signals']} | **Exit Signals:** {stock_data['Exit Signals']}")
        else:
            st.info("No results found.")


if __name__ == "__main__":
    main()
