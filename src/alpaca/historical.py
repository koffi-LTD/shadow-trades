import requests
import pandas as pd

url = "https://data.alpaca.markets/v1beta3/crypto/us/bars?symbols=BTC%2FUSD&timeframe=4H&start=2025-05-01&end=2025-06-22&limit=10000&sort=asc"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)

# df = pd.DataFrame(response.json())
# print(df)