# # %%
# from macd_test import (get_macd_lesser_red_buy_signals, get_macd_lesser_green_sell_signals, plot_macd
# , _process_stock_data_for_macd)
# import pandas as pd

# # %%
# data = pd.read_csv('../data/stock_bars.csv')
# print(data.head())
# result = _process_stock_data_for_macd(data, 'PLTR')
# ticker = 'PLTR'

# # Check if result is a tuple and handle it properly
# if isinstance(result, tuple):
#     processed_data, error_message = result
#     if processed_data is None:
#         print(f"❌ Error for {ticker}: {error_message}")
#         # Plot an empty chart if data processing fails
#         plot_macd(pd.DataFrame(columns=['Close', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']), ticker)
#         exit(1)
# else:
#     processed_data = result

# # Ensure we have valid data before proceeding
# if processed_data is None or 'MACDh_12_26_9' not in processed_data.columns:
#     print(f"❌ Error: Could not process MACD data for {ticker}")
#     exit(1)

# # Get all buy signals
# buy_signals = get_macd_lesser_red_buy_signals(processed_data)
# if buy_signals:
#     print(f"\nFound BUY signals for {ticker}: {', '.join(buy_signals)}")
# else:
#     print(f"\nNo BUY signals found for {ticker}.")

# # Get all sell signals
# sell_signals = get_macd_lesser_green_sell_signals(processed_data)
# if sell_signals:
#     print(f"\nFound SELL signals for {ticker}: {', '.join(sell_signals)}")
# else:
#     print(f"\nNo SELL signals found for {ticker}.")

# # Plot all signals on a single chart
# if buy_signals or sell_signals:
#     print(f"\nGenerating combined MACD chart for {ticker} with all detected signals...")
#     plot_macd(processed_data, ticker, buy_signal_dates=buy_signals, sell_signal_dates=sell_signals)
#     print(f"\nCombined chart for {ticker} saved successfully.")
# else:
#     print(f"\nNo buy or sell signals found for {ticker} to plot.")
#     plot_macd(processed_data, ticker)  # Plot without signals if none found

# print(f"\n{'='*80}\n")

# # %%
