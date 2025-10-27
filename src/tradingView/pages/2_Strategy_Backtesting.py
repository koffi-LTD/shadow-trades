import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Initialize pandas_ta
# pd.DataFrame.ta = ta.Core

from src.tradingView.MACD_backtesting import MACDBacktester
from src.alpaca.historical import fetch_stock_data
from alpaca.data.timeframe import TimeFrame

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
            --chart-bg: #1a1a2e;
            --grid-color: rgba(123, 44, 191, 0.1);
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--secondary-bg);
            padding: 10px 0px;
            border-radius: 10px;
            border: 1px solid var(--accent-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            color: var(--text-color);
            background-color: transparent;
            border-radius: 5px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-color);
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: var(--accent-color);
        }
        
        /* Metric cards */
        .stat-card {
            background: linear-gradient(145deg, #16213e, #1a1a2e);
            border: 1px solid var(--accent-color);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }
        
        /* Trade history */
        .trade-history {
            background-color: var(--secondary-bg);
            border: 1px solid var(--accent-color);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
        }
        
        /* Performance metrics */
        .profit {
            color: var(--success-color);
            text-shadow: 0 0 5px rgba(74, 222, 128, 0.3);
        }
        
        .loss {
            color: var(--error-color);
            text-shadow: 0 0 5px rgba(255, 75, 75, 0.3);
        }
        
        /* Plotly chart customization */
        .js-plotly-plot .plotly .modebar {
            background-color: var(--secondary-bg) !important;
        }
        
        .js-plotly-plot .plotly .modebar-btn path {
            fill: var(--accent-color) !important;
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
        
        /* Input fields */
        .stNumberInput [data-baseweb="input"],
        .stSelectbox [data-baseweb="select"] {
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

st.title("📊 Strategy Backtesting")

# Create tabs for different sections
tab1, tab2 = st.tabs(["🔄 Backtest Runner", "⚙️ Settings"])

with tab2:
    st.header("Backtest Settings")
    
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

    with st.expander("💰 Capital Settings", expanded=True):
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=10000,
            step=1000,
            format="%d"
        )
        
        position_size = st.slider(
            "Position Size (% of Capital)",
            min_value=1,
            max_value=100,
            value=95,
            help="Percentage of capital to use per trade"
        )

    with st.expander("🎯 Strategy Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            consecutive_buy_bars = st.slider("Buy Signal Bars", 1, 10, 6)
        with col2:
            consecutive_sell_bars = st.slider("Sell Signal Bars", 1, 10, 5)
        with col3:
            consecutive_exit_bars = st.slider("Exit Signal Bars", 1, 10, 3)

with tab1:
    # Stock selection
    st.subheader("Select Stock to Test")
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
        run_backtest = st.button("🚀 Run Backtest", use_container_width=True)

    if run_backtest:
        st.subheader(f"Running Backtest for {selected_symbol}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Fetch historical data
            status_text.text("Fetching historical data...")
            progress_bar.progress(20)
            
            df = fetch_stock_data(
                symbols=selected_symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe=TimeFrame.Day
            )
            
            if df is None or df.empty:
                st.warning(f"No data found for {selected_symbol}. Please select another symbol.")
            else:
                # Process data for backtesting
                status_text.text("Processing data...")
                progress_bar.progress(40)
                
                df = df.reset_index().rename(columns={'index': 'timestamp'})
                df.columns = df.columns.str.lower()
                
                # Calculate MACD
                df.ta.macd(
                    close='close',
                    fast=12,
                    slow=26,
                    signal=9,
                    append=True
                )
                
                # Save to temporary CSV for backtester
                status_text.text("Running backtest...")
                progress_bar.progress(60)
                
                temp_csv = f"temp_{selected_symbol}_macd_data.csv"
                df.to_csv(temp_csv, index=False)
                
                # Run backtest
                backtester = MACDBacktester(temp_csv, initial_capital)
                backtester.run_backtest()
                metrics = backtester.get_performance_metrics()
                
                # Clean up temporary file
                os.remove(temp_csv)
                
                progress_bar.progress(80)
                status_text.text("Generating results...")
                
                # Display results in two columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create interactive Plotly chart
                    fig = make_subplots(
                        rows=3, 
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=('Price & Portfolio Value', 'Volume', 'MACD')
                    )
                    
                    # Price and Portfolio Value
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['close'],
                            name='Stock Price',
                            line=dict(color='blue', width=1)
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=backtester.portfolio_values[:len(df)],
                            name='Portfolio Value',
                            line=dict(color='green', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add buy/sell markers
                    for trade in backtester.trades:
                        marker_color = 'green' if trade['type'] == 'BUY' else 'red' if trade['type'] == 'SELL' else 'black'
                        marker_symbol = 'triangle-up' if trade['type'] == 'BUY' else 'triangle-down' if trade['type'] == 'SELL' else 'square'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=[trade['date']],
                                y=[trade['price']],
                                mode='markers',
                                name=trade['type'],
                                marker=dict(
                                    symbol=marker_symbol,
                                    color=marker_color,
                                    size=12
                                ),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                    
                    # Volume
                    fig.add_trace(
                        go.Bar(
                            x=df['timestamp'],
                            y=df['volume'],
                            name='Volume',
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                    
                    # MACD
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['MACD_12_26_9'],
                            name='MACD',
                            line=dict(color='blue', width=1)
                        ),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['MACDs_12_26_9'],
                            name='Signal',
                            line=dict(color='orange', width=1)
                        ),
                        row=3, col=1
                    )
                    
                    # MACD Histogram
                    colors = ['red' if val < 0 else 'green' for val in df['MACDh_12_26_9']]
                    fig.add_trace(
                        go.Bar(
                            x=df['timestamp'],
                            y=df['MACDh_12_26_9'],
                            name='Histogram',
                            marker_color=colors
                        ),
                        row=3, col=1
                    )
                    
                    # Update Plotly figure styling for dark theme
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="#1a1a2e",
                        plot_bgcolor="#1a1a2e",
                        title=dict(
                            text=f"Backtest Results for {selected_symbol}",
                            font=dict(color="#e3e3e3", size=20)
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(color="#e3e3e3"),
                            bgcolor="rgba(26, 26, 46, 0.8)",
                            bordercolor="#7b2cbf"
                        )
                    )
                    
                    # Update grid and axis colors
                    fig.update_xaxes(
                        gridcolor="rgba(123, 44, 191, 0.1)",
                        zerolinecolor="rgba(123, 44, 191, 0.2)",
                        tickfont=dict(color="#e3e3e3")
                    )
                    fig.update_yaxes(
                        gridcolor="rgba(123, 44, 191, 0.1)",
                        zerolinecolor="rgba(123, 44, 191, 0.2)",
                        tickfont=dict(color="#e3e3e3")
                    )
                    
                    # Style the volume bars
                    fig.update_traces(
                        marker_color="rgba(123, 44, 191, 0.5)",
                        selector=dict(name="Volume")
                    )
                    
                    # Style the MACD lines
                    fig.update_traces(
                        line_color="#9d4edd",
                        selector=dict(name="MACD")
                    )
                    fig.update_traces(
                        line_color="#c77dff",
                        selector=dict(name="Signal")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Performance Metrics Section
                    st.subheader("Performance Summary")
                    
                    # Key metrics in cards
                    total_return = (metrics['Final Portfolio Value'] - initial_capital) / initial_capital * 100
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <h3>Total Return</h3>
                                <p class="{'' if total_return >= 0 else 'loss'}">{total_return:,.2f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <h3>Win Rate</h3>
                                <p>{metrics['Win Rate']*100:.1f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <h3>Total Trades</h3>
                                <p>{metrics['Total Trades']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div class="stat-card">
                                <h3>Max Drawdown</h3>
                                <p class="loss">{metrics['Max Drawdown %']:.1f}%</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Detailed metrics table
                    st.markdown("### Detailed Metrics")
                    metrics_df = pd.DataFrame(
                        [[k, v] for k, v in metrics.items()],
                        columns=['Metric', 'Value']
                    )
                    
                    # Format numeric values
                    metrics_df['Value'] = metrics_df['Value'].apply(
                        lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) and "%" not in metrics_df.loc[metrics_df['Value'] == x, 'Metric'].values[0]
                        else f"{x:,.2%}" if isinstance(x, float) and "Rate" in metrics_df.loc[metrics_df['Value'] == x, 'Metric'].values[0]
                        else f"{x:,.2f}%" if isinstance(x, float)
                        else x
                    )
                    
                    st.table(metrics_df)
                    
                    # Trade History
                    with st.expander("📝 Trade History", expanded=False):
                        trades_df = pd.DataFrame(backtester.trades)
                        if not trades_df.empty:
                            trades_df['date'] = pd.to_datetime(trades_df['date'])
                            trades_df['profit'] = trades_df['profit'].fillna(0)
                            
                            # Format the trades dataframe
                            trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:,.2f}")
                            trades_df['value'] = trades_df['value'].apply(lambda x: f"${x:,.2f}")
                            
                            # Format profit with color coding based on value
                            def format_profit(x):
                                if isinstance(x, (int, float)):
                                    color_class = 'profit' if x > 0 else 'loss'
                                    return f"<span class='{color_class}'>${abs(x):,.2f}</span>"
                                return x
                            
                            trades_df['profit'] = trades_df['profit'].apply(format_profit)
                            
                            # Display trades with custom formatting
                            st.markdown(
                                trades_df.style\
                                    .format({'date': lambda x: x.strftime('%Y-%m-%d %H:%M')})\
                                    .hide(axis='index')\
                                    .to_html(),
                                unsafe_allow_html=True
                            )
                
                progress_bar.progress(100)
                status_text.text("Backtest completed successfully!")

        except Exception as e:
            st.error(f"An error occurred during backtesting: {str(e)}")
            st.exception(e)
            
    else:
        st.info("Click 'Run Backtest' to start the analysis.")