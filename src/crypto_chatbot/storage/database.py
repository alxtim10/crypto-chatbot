"""PostgreSQL engine creation and schema initialization."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine

from crypto_chatbot.storage.models import Base


def create_db_engine(database_url: str) -> Engine:
    """Create a synchronous SQLAlchemy PostgreSQL engine."""
    if not database_url.startswith("postgresql"):
        raise ValueError("database_url must be a PostgreSQL URL (postgresql+psycopg://...)")
    return create_engine(database_url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create tables registered on `Base.metadata` if they do not exist."""
    Base.metadata.create_all(engine)
