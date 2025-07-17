# %%
import pandas as pd

# %%
data = pd.read_csv('../data/most_active.csv')

# Clean the data
data = data.drop(['logoid', 'update_mode', 'typespecs', 'currency', 'pricescale', 'minmov', 'minmove2'], axis=1)


data = data[data['Perf.3M'] > 0]
perf = data.sort_values(by=['Perf.1M'], ascending=True)
print(perf.head(10))
# %%
