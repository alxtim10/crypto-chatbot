"""Market data adapters, normalization, and quality checks."""

from crypto_chatbot.market_data.base import (
    DEFAULT_BACKFILL_AVAILABILITY_LAG,
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_FRESHNESS_GRACE,
    MAX_CANDLE_LIMIT,
    MIN_DEGRADED_HISTORY_CANDLES,
    MIN_FULL_HISTORY_CANDLES,
    MarketDataProvider,
    OhlcvBatch,
)
from crypto_chatbot.market_data.ccxt_provider import (
    DEFAULT_BINANCE_PUBLIC_API_URL,
    CcxtMarketDataProvider,
    CcxtOhlcvClient,
)
from crypto_chatbot.market_data.quality import (
    determine_quality_status,
    is_candle_stale,
    normalize_ohlcv_rows,
)
from crypto_chatbot.market_data.symbols import (
    SUPPORTED_MVP_BASES,
    SymbolError,
    SymbolMapping,
    resolve_mvp_spot_symbol,
    resolve_symbol,
)

__all__ = [
    "DEFAULT_BACKFILL_AVAILABILITY_LAG",
    "DEFAULT_BINANCE_PUBLIC_API_URL",
    "DEFAULT_CANDLE_LIMIT",
    "DEFAULT_FRESHNESS_GRACE",
    "MAX_CANDLE_LIMIT",
    "MIN_DEGRADED_HISTORY_CANDLES",
    "MIN_FULL_HISTORY_CANDLES",
    "SUPPORTED_MVP_BASES",
    "CcxtMarketDataProvider",
    "CcxtOhlcvClient",
    "MarketDataProvider",
    "OhlcvBatch",
    "SymbolError",
    "SymbolMapping",
    "determine_quality_status",
    "is_candle_stale",
    "normalize_ohlcv_rows",
    "resolve_mvp_spot_symbol",
    "resolve_symbol",
]
