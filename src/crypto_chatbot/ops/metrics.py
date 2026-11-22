"""PostgreSQL connectivity health probe."""

from __future__ import annotations

from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError


def check_database_health(engine: Engine) -> bool:
    """Execute a lightweight `SELECT 1` check without exposing credentials."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False
