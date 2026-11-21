"""Application composition root for closed-beta Telegram long polling (ADR 0001)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from telegram.ext import Application, ApplicationBuilder

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.handlers import (
    BotService,
    SecretRedactingFilter,
    register_handlers,
)
from crypto_chatbot.config import Settings
from crypto_chatbot.llm.gemini import (
    DailyQuotaTracker,
    GeminiNarrator,
    GoogleGenAiTransport,
)
from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.market_data.derivatives_provider import CcxtDerivativesLiquidityProvider
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.repositories import (
    PostgresDerivativesRepository,
    PostgresMarketRepository,
    PostgresUserMemoryRepository,
)


def configure_logging(secrets: Sequence[str] = ()) -> SecretRedactingFilter:
    """Configure structured logging with secret redaction and suppressed httpx URL logs."""
    redacting_filter = SecretRedactingFilter(secrets)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    root_logger = logging.getLogger()
    root_logger.addFilter(redacting_filter)
    for handler in root_logger.handlers:
        handler.addFilter(redacting_filter)
    # Prevent python-telegram-bot's underlying httpx client from logging full Bot API token URLs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return redacting_filter


def build_telegram_application(
    settings: Settings,
    *,
    initialize_db: bool = True,
) -> Application[Any, Any, Any, Any, Any, Any]:
    """Wire PostgreSQL repositories, CCXT provider, Gemini narrator, and Telegram handlers."""
    engine = create_db_engine(settings.database_url.get_secret_value())
    if initialize_db:
        init_db(engine)

    market_repo = PostgresMarketRepository(engine)
    derivatives_repo = PostgresDerivativesRepository(engine)
    memory_repo = PostgresUserMemoryRepository(engine)
    provider = CcxtMarketDataProvider()
    derivatives_provider = CcxtDerivativesLiquidityProvider()
    analysis_service = AnalysisService(
        provider=provider,
        repository=market_repo,
        derivatives_provider=derivatives_provider,
        derivatives_repository=derivatives_repo,
    )

    transport = GoogleGenAiTransport(
        api_key=settings.gemini_api_key,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    quota_tracker = DailyQuotaTracker(
        per_user_daily_cap=settings.gemini_user_daily_cap,
        global_daily_cap=settings.gemini_global_daily_cap,
    )
    narrator = GeminiNarrator(
        model=settings.gemini_model,
        transport=transport,
        timeout_seconds=settings.gemini_timeout_seconds,
        max_output_tokens=settings.gemini_max_output_tokens,
        quota_tracker=quota_tracker,
    )
    memory_service = MemoryService(memory_repo)
    access_controller = AccessController(
        settings.allowed_telegram_user_ids,
        rate_limit_per_minute=settings.rate_limit_per_minute,
    )
    bot_service = BotService(
        access=access_controller,
        analysis_service=analysis_service,
        narrator=narrator,
        memory=memory_service,
    )

    async def on_shutdown(app: Application[Any, Any, Any, Any, Any, Any]) -> None:
        del app
        await provider.close()
        await derivatives_provider.close()
        engine.dispose()

    application = (
        ApplicationBuilder()
        .token(settings.telegram_bot_token.get_secret_value())
        .post_shutdown(on_shutdown)
        .build()
    )
    register_handlers(application, bot_service)
    return application


def run_polling(settings: Settings) -> None:
    """Start closed-beta Telegram long polling with secret redaction active."""
    configure_logging(
        secrets=(
            settings.telegram_bot_token.get_secret_value(),
            settings.gemini_api_key.get_secret_value(),
            settings.database_url.get_secret_value(),
        )
    )
    application = build_telegram_application(settings)
    application.run_polling(drop_pending_updates=True)
