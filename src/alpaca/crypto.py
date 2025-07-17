import json, os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.crypto import CryptoDataStream

from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    StopLimitOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest
)
from alpaca.trading.enums import (
    AssetClass,
    AssetStatus,
    OrderSide,
    OrderType,
    TimeInForce,
    QueryOrderStatus
)
from alpaca.common.exceptions import APIError

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
paper = os.getenv("ALPACA_PAPER")
trade_api_url = os.getenv("ALPACA_TRADE_API_URL")

trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)

print(trade_client.get_account().__dict__)