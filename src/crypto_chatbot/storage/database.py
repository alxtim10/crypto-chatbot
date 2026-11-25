"""PostgreSQL engine creation and schema initialization."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, text

from crypto_chatbot.storage.models import Base


def create_db_engine(database_url: str) -> Engine:
    """Create a synchronous SQLAlchemy PostgreSQL engine."""
    if not database_url.startswith("postgresql"):
        raise ValueError("database_url must be a PostgreSQL URL (postgresql+psycopg://...)")
    return create_engine(database_url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create tables registered on `Base.metadata` and ensure backwards-compatible columns."""
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE market_candles "
                "ADD COLUMN IF NOT EXISTS taker_buy_base_volume DOUBLE PRECISION;"
            )
        )
        conn.execute(
            text("ALTER TABLE trade_setups ADD COLUMN IF NOT EXISTS market_type VARCHAR(16);")
        )
