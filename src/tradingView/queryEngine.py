# %%
from tradingview_screener import Query, col
import pandas as pd


# Define sectors to include
# sectors = ['Finance', 'utilities']
exchanges = ['AMEX', 'CBOE', 'NASDAQ', 'NYSE']
sectors = ['Electronic Technology', 'Non-Energy Minerals', 'Technology Services', 'Finance', 'Communications', 'Producer Manufacturing']
# %%
top_50_bullish = (Query()
 .select('name', 'close','Perf.YTD','Perf.3M','Perf.Y', 'Value.Traded','sector')
 .where(
    #  col('market_cap_basic').between(1_000_000, 50_000_000),
     col('relative_volume_10d_calc') > 0.9,
     col('sector').isin(sectors),
     col('Perf.YTD') > 5,
     col('MACD.macd') >= col('MACD.signal')
 )
 .order_by('volume', ascending=False)
 .limit(500))

# %%
result = top_50_bullish.get_scanner_data()
data = pd.DataFrame(result[1])
data['extraction_day'] = pd.Timestamp.now().strftime('%Y-%m-%d')
data.to_csv('../data/top_50_bullish.csv', index=False, sep=',')


# %%
# Define the base query
def get_most_active_query(exchange):
    return (Query()
     .select(
         'exchange',
         'name',
         'description',
         'Perf.1M',
         'Perf.3M',
         'Perf.Y',
         'Perf.YTD',
         'RSI',
         'type',
         'Value.Traded',
         'close',
         'change',
         'volume',
         'relative_volume_10d_calc',
         'market_cap_basic',
         'price_earnings_ttm',
         'earnings_per_share_diluted_ttm',
         'earnings_per_share_diluted_yoy_growth_ttm',
         'dividends_yield_current',
         'sector'
     )
     .where(
         col('type')=='stock',
         col('exchange') == exchange,
         col('active_symbol') == True,
         col('sector').isin(sectors),
         
     )
     .order_by('volume', ascending=False, nulls_first=False)
     .limit(300)  # Get top 300 from each exchange
     .set_markets('america')
     .set_property('symbols', {'query': {'types': ['stock']}}))

# Fetch data for each exchange and combine
most_active_dfs = []
for exchange in exchanges:
    try:
        print(f"Fetching most active for {exchange}...")
        df = get_most_active_query(exchange).get_scanner_data()
        df = pd.DataFrame(df[1])
        if not df.empty:
            most_active_dfs.append(df)
            print(f"  - Found {len(df)} stocks from {exchange}")
    except Exception as e:
        print(f"Error fetching data for {exchange}: {str(e)}")

# Combine all results
if most_active_dfs:
    most_active = pd.concat(most_active_dfs, ignore_index=True)
    # Sort the combined results by volume
    most_active = most_active.sort_values('volume', ascending=False)
    # Save to CSV
    most_active.to_csv('../data/most_active.csv', index=False)
    print(f"Saved {len(most_active)} total stocks to most_active.csv")
else:
    print("No data was fetched from any exchange.")
    most_active = pd.DataFrame() 
     # Empty DataFrame if no data


# %%
# most_active_data = most_active.get_scanner_data()
# data = pd.DataFrame(most_active_data[1])
# # add extraction day
# data['extraction_day'] = pd.Timestamp.now().strftime('%Y-%m-%d')
# data.to_csv('../data/most_active.csv', index=False, sep=',')
