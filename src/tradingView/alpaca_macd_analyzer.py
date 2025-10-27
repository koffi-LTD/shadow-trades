import os
import sys
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
from tabulate import tabulate

# Configure logging
def setup_logging():
    """Configure logging to both console and file with timestamps."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join(log_dir, f'trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Set up root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress matplotlib debug logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Check if TA-Lib is available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib is not installed. Using custom MACD calculation.")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.alpaca.historical import fetch_stock_data
from alpaca.data.timeframe import TimeFrame

# Import MACD analysis functions from macd_test
from macd_strategy import (
    plot_macd,
    get_macd_lesser_red_buy_signals,
    get_macd_lesser_green_sell_signals,
    _process_stock_data_for_macd
)

# Import chart preparation functions from streamlit app

def format_table(data: List[Dict[str, Any]], headers: str = 'keys', tablefmt: str = 'grid') -> str:
    """Format data into a table.
    
    Args:
        data: List of dictionaries containing the data to display
        headers: Table headers ('keys' uses dict keys as headers)
        tablefmt: Table format ('grid', 'fancy_grid', 'pipe', 'html', etc.)
        
    Returns:
        Formatted table string
    """
    if not data:
        return "No data to display"
    return tabulate(data, headers=headers, tablefmt=tablefmt, showindex=False)

def save_to_csv(data: List[Dict], filename: str) -> str:
    """Save data to a CSV file in the output directory.
    
    Args:
        data: List of dictionaries containing the data to save
        filename: Name of the output file (without extension)
        
    Returns:
        Path to the saved file
    """
    if not data:
        logger.warning("No data provided to save to CSV")
        return ""
        
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Output directory: {output_dir}")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{filename}_{timestamp}.csv")
    logger.debug(f"Generated filepath: {filepath}")
    
    try:
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {len(df)} records to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving {filepath}: {str(e)}")
        raise

def display_analysis_summary(results: Dict[str, Dict]) -> None:
    """Display the analysis summary and save results to CSV files.
    
    Args:
        results: Dictionary containing analysis results for each symbol
    """
    if not results:
        logger.warning("No analysis results to display.")
        return
    
    # Prepare summary data
    summary_data = []
    all_signals = []
    
    for symbol, signals in results.items():
        # Add to summary data
        summary_data.append({
            'Symbol': symbol,
            'Buy_Signals': len(signals['buy_signals']),
            'Sell_Signals': len(signals['sell_signals']),
            'Latest_Buy': signals['buy_signals'][-1] if signals['buy_signals'] else 'None',
            'Latest_Sell': signals['sell_signals'][-1] if signals['sell_signals'] else 'None',
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # Prepare detailed signals
        for buy_date in signals['buy_signals']:
            all_signals.append({
                'Symbol': symbol,
                'Date': buy_date,
                'Signal': 'BUY',
                'Strength': 'High',
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

        for sell_date in signals['sell_signals']:
            all_signals.append({
                'Symbol': symbol,
                'Date': sell_date,
                'Signal': 'SELL',
                'Strength': 'High',
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    # Save to CSV files
    summary_file = save_to_csv(summary_data, 'macd_summary')
    signals_file = save_to_csv(all_signals, 'macd_signals')

    # Log summary information
    logger.info("="*80)
    logger.info("MACD Analysis Summary")
    logger.info("="*80)
    logger.info(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Symbols Analyzed: {len(results)}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Detailed signals saved to: {signals_file}")

    # Also print to console for immediate visibility
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)

    # Only add the handler if it's not already added
    if not any(isinstance(h, logging.StreamHandler) and h.level == logging.INFO 
              for h in logging.getLogger().handlers):
        logging.getLogger('').addHandler(console)
    
    logger.info("\n" + "="*80)
    logger.info("MACD Analysis Summary")
    logger.info("="*80)
    logger.info(f"\nSummary saved to: {summary_file}")
    logger.info(f"Detailed signals saved to: {signals_file}")
    
    # Display summary table
    # Log summary table
    display_summary = [{k: v for k, v in item.items() if k != 'Analysis_Date'} for item in summary_data]
    logger.info("\nSummary Table:" + "\n" + format_table(display_summary, tablefmt='grid'))
    
    # Log sample of detailed signals
    if all_signals:
        logger.info("\nSample of Detailed Signals (first 5):")
        logger.info("\n" + format_table(all_signals[:5], tablefmt='grid'))
        logger.info(f"... and {len(all_signals) - 5} more signals in the CSV file")

def get_stock_analysis(symbol: str, start_date: str, end_date: Optional[str] = None, n_days: Optional[int] = 0) -> Optional[Tuple[pd.DataFrame, List[str], List[str]]]:
    """Fetches and processes stock data for MACD analysis."""
    try:
        df = fetch_stock_data(
            symbols=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=TimeFrame.Day,
            n_days=n_days
        )
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                logger.error(f"Could not determine datetime column for {symbol}")
                return None

        df = df.reset_index().rename(columns={'index': 'timestamp', str.lower: str.lower})
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        processed_data = _process_stock_data_for_macd(df.set_index('timestamp'), symbol)
        if isinstance(processed_data, tuple) and processed_data[0] is None:
            logger.error(f"Error processing {symbol}: {processed_data[1]}")
            return None

        buy_signals = get_macd_lesser_red_buy_signals(processed_data)
        sell_signals = get_macd_lesser_green_sell_signals(processed_data)

        return df, buy_signals, sell_signals
    except Exception as e:
        logger.error(f"Error analyzing {symbol}", exc_info=True)
        return None


def analyze_stocks_with_macd(
    symbols: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    n_days: Optional[int] = 0,
    output_dir: str = "../data/macd_analysis"
) -> Dict[str, Dict[str, List[str]]]:
    """Analyzes multiple stocks using MACD indicators."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for symbol in symbols:
        logger.info(f"\nAnalyzing {symbol}...")
        analysis_result = get_stock_analysis(symbol, start_date, end_date, n_days)
        if analysis_result is None:
            continue

        df, buy_signals, sell_signals = analysis_result
        results[symbol] = {'buy_signals': buy_signals, 'sell_signals': sell_signals}

        # Matplotlib plot generation
        processed_data_for_plot = _process_stock_data_for_macd(df.set_index('timestamp'), symbol)
        if not (isinstance(processed_data_for_plot, tuple) and processed_data_for_plot[0] is None):
            plot_macd(processed_data_for_plot, symbol, buy_signals, sell_signals)
        
        # # Streamlit chart rendering
        # try:
        #     chart_data = prepare_chart_data(df)
        #     chart_config = create_chart_config(
        #         chart_data['candles'], chart_data['volume'], chart_data['macd_fast'],
        #         chart_data['macd_slow'], chart_data['macd_hist'], ticker_symbol=symbol,
        #         buy_signal_dates=buy_signals, sell_signal_dates=sell_signals
        #     )
        #     st.subheader(f"MACD Analysis for {symbol}")
        #     renderLightweightCharts(chart_config, f'multipane_{symbol}')
        #     if buy_signals:
        #         st.success(f"🟢 Buy Signals: {', '.join(buy_signals[:3])}{'...' if len(buy_signals) > 3 else ''}")
        #     if sell_signals:
        #         st.error(f"🔴 Sell Signals: {', '.join(sell_signals[:3])}{'...' if len(sell_signals) > 3 else ''}")
        # except Exception as chart_error:
        #     logger.warning(f"Could not display Streamlit chart for {symbol}: {chart_error}")

    return results

def main():
    # Example usage
    symbols = ["PLTR"]
    data = pd.read_csv('../data/most_active.csv')
    # filter where 1.M > 5
    data = data[data['Close'] > 5]
    symbols = data['name'].tolist()

    # Set date range (last 2 year )
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    print(start_date)
    print(f"\nRunning MACD analysis from {start_date} to {end_date}")
    print("=" * 50)

    results = analyze_stocks_with_macd(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        n_days=365
    )

    # Display formatted results
    # display_analysis_summary(results)

if __name__ == "__main__":
    main()
