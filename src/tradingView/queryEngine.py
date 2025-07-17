# %%
from tradingview_screener import Query, col
import pandas as pd


# Define sectors to include
sectors = ['Health Technology', 'Consumer Non-Durables', 'Technology Services', 'Non-Energy Minerals', 'Consumer Durables', 'Commercial Services', 'Electronic Technology', 'Producer Manufacturing', 'Distribution Services', 'Retail Trade', 'Process Industries', 'Finance', 'Transportation', 'Miscellaneous']

# %%
top_50_bullish = (Query()
 .select('name', 'close','Perf.1M','Perf.3M','Perf.Y', 'sector')
 .where(
     col('market_cap_basic').between(1_000_000, 50_000_000),
     col('relative_volume_10d_calc') > 1.2,
     col('sector').isin(sectors),
     col('MACD.macd') >= col('MACD.signal')
 )
 .order_by('volume', ascending=False)
 .limit(100))

# %%
result = top_50_bullish.get_scanner_data()
data = pd.DataFrame(result[1])
data['extraction_day'] = pd.Timestamp.now().strftime('%Y-%m-%d')
data.to_csv('../data/top_50_bullish.csv', index=False, sep=',')


# %%
# most active
most_active = (Query()
 .select(
     'name',
     'description',
     'Perf.1M',
     'Perf.3M',
     'Perf.Y',
     'type',
     'typespecs',
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
     'sector',
     'recommendation_mark',
 )
 .where(
     col('exchange').isin(['NASDAQ', 'NYSE']),
     col('is_primary') == True,
     col('typespecs').has('common'),
     col('typespecs').has_none_of('preferred'),
     col('type') == 'stock',
     col('close').between(2, 10000),
     col('active_symbol') == True,
 )
 .order_by('Value.Traded', ascending=False, nulls_first=False)
 .limit(200)
 .set_markets('america')
 .set_property('symbols', {'query': {'types': ['stock', 'fund', 'dr', 'structured']}})
 .set_property('preset', 'volume_leaders'))

# %%
most_active_data = most_active.get_scanner_data()
data = pd.DataFrame(most_active_data[1])
# add extraction day
data['extraction_day'] = pd.Timestamp.now().strftime('%Y-%m-%d')
data.to_csv('../data/most_active.csv', index=False, sep=',')
# %%
