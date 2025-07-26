import os
from datetime import datetime, timedelta
from typing import List, Optional, Union
from dotenv import load_dotenv
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_stock_data(
    symbols: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_days: Optional[int] = None,
    timeframe: TimeFrame = TimeFrame.Hour,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetches historical stock data from Alpaca API and optionally saves it to a CSV file.

    Args:
        symbols: Single ticker symbol or list of ticker symbols (e.g., 'AAPL' or ['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
                   If None and n_days is provided, it will be calculated from end_date.
        end_date: End date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format. 
                 If None, uses current date and time.
        n_days: Number of days of data to fetch. If provided along with end_date,
               calculates start_date as end_date - n_days.
        timeframe: TimeFrameUnit enum for the bar timeframe (default: Day)
        output_file: Optional path to save the data as CSV

    Returns:
        pd.DataFrame: DataFrame containing the historical stock data

    Raises:
        ValueError: If required environment variables are not set or if date parameters are invalid
        Exception: For any API-related errors
    """
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not all([api_key, api_secret]):
        raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env file")
    
    # Validate date parameters
    if start_date is None and n_days is None:
        raise ValueError("Either start_date or n_days must be provided")
    
    # Convert string dates to datetime objects
    end_dt = pd.to_datetime(end_date) if end_date else datetime.now()
    
    # Calculate start_date if n_days is provided
    if n_days is not None and n_days > 0:
        start_dt = end_dt - timedelta(days=n_days)
    else:
        start_dt = pd.to_datetime(start_date) if start_date else None
    
    # Ensure start date is before end date
    if start_dt and start_dt >= end_dt:
        raise ValueError("start_date must be before end_date")
    
    try:
        # Initialize Alpaca client
        client = StockHistoricalDataClient(api_key, api_secret)
        
        # Ensure symbols is a list
        if isinstance(symbols, str):
            symbols = [symbols]
            
        # Prepare request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt
        )
        
        # Get bars and convert to DataFrame
        bars = client.get_stock_bars(request_params)
        bars_df = bars.df
        
        # Reset index to make timestamp a column if multi-index
        if isinstance(bars_df.index, pd.MultiIndex):
            bars_df = bars_df.reset_index(level=1)
        
        # Save to CSV if output file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            bars_df.to_csv(output_file, index=True)
            print(f"Data successfully saved to {output_file}")
        
        return bars_df
    
    except Exception as e:
        raise Exception(f"Error fetching stock data: {str(e)}")


if __name__ == "__main__":
    from lightweight_charts import Chart
    # Example usage
    try:
        df = fetch_stock_data(
            symbols="PLTR",
            start_date="2025-01-03",
            end_date= datetime.now().strftime("%Y-%m-%d"),
            n_days=90,
            timeframe=TimeFrame.Hour,
            output_file="../data/stock_bars.csv"
        )
        chart = Chart()
        chart.set(df)
        chart.show(block=True)
        print(df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
