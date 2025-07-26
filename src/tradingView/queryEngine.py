# %%
from tradingview_screener import Query, col
import pandas as pd


# Define sectors to include
sectors = ['Finance', 'utilities']
# sectors = ['Technology Services', 'Non-Energy Minerals']

# %%
top_50_bullish = (Query()
 .select('name', 'close','Perf.YTD','Perf.3M','Perf.Y', 'sector')
 .where(
     col('market_cap_basic').between(1_000_000, 50_000_000),
     col('relative_volume_10d_calc') > 1.2,
     col('sector').isin(sectors),
     col('Perf.YTD') > 5,
     col('MACD.macd') >= col('MACD.signal')
 )
 .order_by('volume', ascending=False)
 .limit(200))

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
     'sector',
     'sector.tr'
 )
 .where(
     col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
    #  col('sector').isin(sectors),
     col('is_primary') == True,
     col('typespecs').has('common'),
     col('typespecs').has_none_of('preferred'),
     col('type') == 'stock',
     col('Perf.Y').between(5, 10000),
     col('Perf.YTD') > 5,
     col('RSI').between(47, 80),
     col('volume') >= 2000000,
     col('active_symbol') == True,
 )
 .order_by('volume', ascending=False, nulls_first=False)
#  .order_by('Value.Traded', ascending=False, nulls_first=False)
 .limit(500)
 .set_markets('america')
 .set_property('symbols', {'query': {'types': ['stock', 'fund', 'dr', 'structured']}})
 .set_property('preset', 'all_stocks'))

# %%
most_active_data = most_active.get_scanner_data()
data = pd.DataFrame(most_active_data[1])
# add extraction day
data['extraction_day'] = pd.Timestamp.now().strftime('%Y-%m-%d')
data.to_csv('../data/most_active.csv', index=False, sep=',')
# %%
