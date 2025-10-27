import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alpaca.historical import fetch_stock_data
from tradingView.streamlitApp import (
    prepare_chart_data, 
    create_chart_config,
    get_macd_lesser_red_buy_signals,
    get_macd_lesser_green_sell_signals,
    get_consecutive_red_candles_exit,
    get_second_buy_entry
)
from streamlit_lightweight_charts import renderLightweightCharts

# Custom CSS for gaming theme
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --background-color: #1a1a2e;
            --secondary-bg: #16213e;
            --accent-color: #7b2cbf;
            --text-color: #e3e3e3;
            --success-color: #4ade80;
            --warning-color: #ff9800;
            --error-color: #ff4b4b;
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--secondary-bg);
            padding: 10px 0px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: var(--text-color);
            background-color: transparent;
            border-radius: 5px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color);
        }
        
        /* Signal cards */
        .signal-card {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .signal-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }
        
        .buy-signal {
            background-color: rgba(74, 222, 128, 0.1);
            border: 1px solid #4ade80;
        }
        
        .sell-signal {
            background-color: rgba(255, 75, 75, 0.1);
            border: 1px solid #ff4b4b;
        }
        
        .exit-signal {
            background-color: rgba(156, 163, 175, 0.1);
            border: 1px solid #9ca3af;
        }
        
        /* Metric animations */
        .metric-container {
            background: linear-gradient(45deg, #16213e, #1a1a2e);
            border: 1px solid var(--accent-color);
            border-radius: 10px;
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        .metric-container:hover {
            box-shadow: 0 0 15px rgba(123, 44, 191, 0.5);
            transform: translateY(-2px);
        }
        
        /* Chart container */
        .chart-container {
            background: var(--secondary-bg);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid var(--accent-color);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--secondary-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--accent-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #9d4edd;
        }
        
        /* Inputs and selectors */
        .stSelectbox [data-baseweb="select"] {
            background-color: var(--secondary-bg);
            border-color: var(--accent-color);
        }
        
        .stDateInput [data-baseweb="input"] {
            background-color: var(--secondary-bg);
            border-color: var(--accent-color);
            color: var(--text-color);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--secondary-bg);
            border: 1px solid var(--accent-color);
            border-radius: 5px;
            color: var(--text-color);
        }
        
        .streamlit-expanderContent {
            background-color: var(--background-color);
            border: 1px solid var(--accent-color);
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 MACD Pattern Analysis", width="stretch")

# Create tabs for different sections
tab1, tab2 = st.tabs(["📈 Chart Analysis", "⚙️ Settings"])

with tab2:
    st.header("Analysis Settings")
    
    with st.expander("📅 Date Range", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = st.date_input(
                "Start Date",
                value=end_date - timedelta(days=365),
                max_value=end_date - timedelta(days=1)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=end_date,
                min_value=start_date + timedelta(days=1),
                max_value=end_date
            )

    with st.expander("📊 MACD Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            fast_period = st.number_input("Fast Period", min_value=1, max_value=50, value=12)
        with col2:
            slow_period = st.number_input("Slow Period", min_value=1, max_value=100, value=26)
        with col3:
            signal_period = st.number_input("Signal Period", min_value=1, max_value=50, value=9)

    with st.expander("🎯 Pattern Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            consecutive_buy_bars = st.slider("Buy Signal Bars", 1, 10, 6)
        with col2:
            consecutive_sell_bars = st.slider("Sell Signal Bars", 1, 10, 5)
        with col3:
            consecutive_exit_bars = st.slider("Exit Signal Bars", 1, 10, 3)

with tab1:
    # Stock selection in the main area
    st.subheader("Select Stock")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            symbols_df = pd.read_csv('../data/most_active.csv')
            symbols_df = symbols_df[symbols_df['close'] > 5]
            symbols = symbols_df['name'].tolist()
        except FileNotFoundError:
            symbols = ['AAPL', 'GOOG', 'MSFT']
            
        selected_symbol = st.selectbox(
            "Choose a stock symbol:",
            symbols,
            index=0
        )
        
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("🔄 Refresh Data", width="stretch"):
            st.cache_data.clear()
    
    # Fetch and process data
    with st.spinner('Fetching data...'):
        df = fetch_stock_data(
            symbols=selected_symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe=TimeFrame.Day
        )

        if df is None or df.empty:
            st.warning(f"No data found for {selected_symbol}. Please select another symbol.")
        else:
            # Process data
            df = df.reset_index().rename(columns={'index': 'timestamp'})
            df.columns = df.columns.str.lower()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            df.set_index('timestamp', inplace=True)
            
            # Calculate MACD if needed
            if not all(col in df.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']):
                df.ta.macd(
                    close='close',
                    fast=fast_period,
                    slow=slow_period,
                    signal=signal_period,
                    append=True
                )
            
            # Ensure we have the MACD histogram column with the expected name
            if 'MACDh_12_26_9' in df.columns or 'macdh_12_26_9' in df.columns:
                df['hist'] = df.get('MACDh_12_26_9', df.get('macdh_12_26_9'))
                
                # Get signals only if we have valid MACD data
                buy_signals = get_macd_lesser_red_buy_signals(df, consecutive_bars=consecutive_buy_bars)
                sell_signals = get_macd_lesser_green_sell_signals(df, consecutive_bars=consecutive_sell_bars)
                exit_signals, exit_history = get_consecutive_red_candles_exit(df, consecutive_bars=consecutive_exit_bars, buy_signal_dates=buy_signals)
                second_buy_signals_dates, second_buy_entries = get_second_buy_entry(df, buy_signals, consecutive_bars=3)
                second_sell_signals_dates = get_macd_lesser_green_sell_signals(df, consecutive_bars=consecutive_sell_bars)
            else:
                st.error("Could not find MACD histogram data in the DataFrame")
                # Initialize empty signals if MACD data is not available
                buy_signals = []
                sell_signals = []
                exit_signals = []
                second_buy_signals_dates = []
                second_sell_signals_dates = []
                second_buy_entries = pd.DataFrame()
                exit_history = {}
            
            # Display stock metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current_price = df['close'].iloc[-1]
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "MACD",
                    f"{df['MACD_12_26_9'].iloc[-1]:.3f}",
                    f"{df['MACD_12_26_9'].iloc[-1] - df['MACD_12_26_9'].iloc[-2]:+.3f}"
                )
                
            with col3:
                st.metric(
                    "Signal",
                    f"{df['MACDs_12_26_9'].iloc[-1]:.3f}",
                    f"{df['MACDs_12_26_9'].iloc[-1] - df['MACDs_12_26_9'].iloc[-2]:+.3f}"
                )
                
            with col4:
                st.metric(
                    "Histogram",
                    f"{df['MACDh_12_26_9'].iloc[-1]:.3f}",
                    f"{df['MACDh_12_26_9'].iloc[-1] - df['MACDh_12_26_9'].iloc[-2]:+.3f}"
                )
            
            # Chart section
            st.subheader("Interactive Chart")
            df = df.reset_index()
            chart_data = prepare_chart_data(df)
            chart_config = create_chart_config(
                chart_data['candles'],
                chart_data['volume'],
                chart_data['macd_fast'],
                chart_data['macd_slow'],
                chart_data['macd_hist'],
                selected_symbol,
                buy_signals,
                second_buy_signals_dates,
                sell_signals,
                exit_signals,
                second_sell_signals_dates,
            )
            renderLightweightCharts(chart_config, 'multipane')
            
            # Signals section
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.subheader("")
                st.subheader("🟢 Buy Signals")
                if buy_signals:
                    for signal in buy_signals[-3:]:  # Show last 3 signals
                        st.write(f"• {signal}")
                    if len(buy_signals) > 3:
                        st.caption(f"... and {len(buy_signals) - 3} more")
                else:
                    st.write("None")
                
            with col2:
                st.subheader("🔵 2nd Buy")
                if second_buy_signals_dates:
                    for signal in second_buy_signals_dates[-3:]:
                        st.write(f"• {signal}")
                    if len(second_buy_signals_dates) > 3:
                        st.caption(f"... and {len(second_buy_signals_dates) - 3} more")
                else:
                    st.write("None")
                
            with col3:
                st.subheader("🔴 Sell")
                if sell_signals:
                    for signal in sell_signals[-3:]:
                        st.write(f"• {signal}")
                    if len(sell_signals) > 3:
                        st.caption(f"... and {len(sell_signals) - 3} more")
                else:
                    st.write("None")
                
            with col4:
                st.subheader("⬛ Exit")
                if exit_signals:
                    for signal in exit_signals[-3:]:
                        st.write(f"• {signal}")
                    if len(exit_signals) > 3:
                        st.caption(f"... and {len(exit_signals) - 3} more")
                else:
                    st.write("None")

            with col5:
                st.subheader("🟣 2nd Sell")
                if second_sell_signals_dates:
                    for signal in second_sell_signals_dates[-3:]:
                        st.write(f"• {signal}")
                    if len(second_sell_signals_dates) > 3:
                        st.caption(f"... and {len(second_sell_signals_dates) - 3} more")
                else:
                    st.write("None")

            # Trade history section display in data table
            st.subheader("Trade Exit History")
            
            if not exit_history.empty:
                st.dataframe(exit_history, width="stretch", height=10)