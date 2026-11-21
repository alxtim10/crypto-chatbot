from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.commands import (
    MAX_USER_MESSAGE_CHARS,
    AnalyzeIntent,
    CommandParseError,
    CompareIntent,
    RefusalIntent,
    parse_analyze_args,
    parse_compare_args,
    parse_watchlist_args,
    resolve_followup_intent,
)
from crypto_chatbot.bot.templates import (
    render_forget_message,
    render_memory_message,
    render_watchlist_message,
    split_telegram_message,
)
from crypto_chatbot.schemas.commands import WatchlistAction
from crypto_chatbot.schemas.market import Timeframe
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    StoredMessage,
    UserPreferences,
)


def test_access_controller_allowlist_and_sliding_rate_limit() -> None:
    now = [1000.0]
    access = AccessController(
        [111, 222],
        rate_limit_per_minute=10,
        window_seconds=60.0,
        clock=lambda: now[0],
    )

    assert access.check_access(None).reason == "not_allowlisted"
    assert access.check_access(999).reason == "not_allowlisted"
    assert 999 not in access._buckets

    for _ in range(10):
        decision = access.check_access(111)
        assert decision.allowed is True
        assert decision.reason is None

    blocked = access.check_access(111)
    assert blocked.allowed is False
    assert blocked.reason == "rate_limited"

    # Other allowlisted user still unaffected
    assert access.check_access(222).allowed is True

    # After 60 seconds, window slides open again
    now[0] = 1061.0
    assert access.check_access(111).allowed is True


def test_command_argument_parsers() -> None:
    a1 = parse_analyze_args(["btc", "4h"])
    assert a1.symbol == "BTC/USDT"
    assert a1.timeframe is Timeframe.H4

    a2 = parse_analyze_args(["ETH"], default_timeframe=Timeframe.D1)
    assert a2.symbol == "ETH/USDT"
    assert a2.timeframe is Timeframe.D1

    c1 = parse_compare_args(["BTC", "ETH", "1d"])
    assert c1.symbols == ("BTC/USDT", "ETH/USDT")
    assert c1.timeframe is Timeframe.D1

    c2 = parse_compare_args(["btc", "eth", "sol"])
    assert c2.symbols == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    assert c2.timeframe is Timeframe.H4

    w_list = parse_watchlist_args([])
    assert w_list.action is WatchlistAction.LIST

    w_add = parse_watchlist_args(["add", "sol"])
    assert w_add.action is WatchlistAction.ADD
    assert w_add.symbol == "SOL/USDT"

    w_rm = parse_watchlist_args(["remove", "BTC/USDT"])
    assert w_rm.action is WatchlistAction.REMOVE
    assert w_rm.symbol == "BTC/USDT"


@pytest.mark.parametrize(
    ("parser_fn", "args"),
    [
        (parse_analyze_args, []),
        (parse_analyze_args, ["BTC", "4h", "extra"]),
        (parse_analyze_args, ["BTC", "15m"]),
        (parse_analyze_args, ["DOGE", "4h"]),
        (parse_analyze_args, ["BTC-PERP", "4h"]),
        (parse_compare_args, ["BTC"]),
        (parse_compare_args, ["BTC", "ETH", "15m"]),
        (parse_compare_args, ["BTC", "BTC/USDT", "4h"]),
        (parse_compare_args, ["BTC", "DOGE", "4h"]),
        (parse_watchlist_args, ["add"]),
        (parse_watchlist_args, ["add", "DOGE"]),
        (parse_watchlist_args, ["clear"]),
    ],
)
def test_command_argument_parsers_reject_invalid_inputs(
    parser_fn: object,
    args: list[str],
) -> None:
    with pytest.raises(CommandParseError):
        parser_fn(args)  # type: ignore[operator]


def test_followup_intent_resolver_scenarios() -> None:
    now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
    history = (
        StoredMessage(
            role="user", content="/analyze BTC 4h", created_at=now - timedelta(minutes=2)
        ),
        StoredMessage(
            role="assistant",
            content="[BTC/USDT 4h] Harga {F1} di atas EMA 200 ({F6}).",
            snapshot_id="snap-btc-1",
            created_at=now - timedelta(minutes=1),
        ),
    )

    cmp_intent = resolve_followup_intent(
        "bandingkan momentumnya dengan ETH",
        recent_messages=history,
    )
    assert cmp_intent == CompareIntent(symbols=("BTC/USDT", "ETH/USDT"), timeframe=Timeframe.H4)

    ctx_intent = resolve_followup_intent(
        "bagaimana kondisi RSI dan momentumnya?",
        recent_messages=history,
    )
    assert ctx_intent == AnalyzeIntent(symbol="BTC/USDT", timeframe=Timeframe.H4)

    explicit_intent = resolve_followup_intent(
        "cek SOL pada 1d",
        recent_messages=history,
    )
    assert explicit_intent == AnalyzeIntent(symbol="SOL/USDT", timeframe=Timeframe.D1)

    order_refusal = resolve_followup_intent(
        "tolong pasang order beli sekarang", recent_messages=history
    )
    assert isinstance(order_refusal, RefusalIntent)

    perp_refusal = resolve_followup_intent("analisis BTC perpetual 4h", recent_messages=history)
    assert isinstance(perp_refusal, RefusalIntent)

    unsupported_refusal = resolve_followup_intent(
        "bandingkan dengan DOGE/USDT", recent_messages=history
    )
    assert isinstance(unsupported_refusal, RefusalIntent)

    oversized_refusal = resolve_followup_intent(
        "x" * (MAX_USER_MESSAGE_CHARS + 1), recent_messages=()
    )
    assert isinstance(oversized_refusal, RefusalIntent)


def test_template_escaping_and_splitting_on_hostile_input() -> None:
    now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
    prefs = UserPreferences(watchlist=("BTC/USDT",))
    hostile_msg = StoredMessage(
        role="user",
        content="<script>alert()</script> & <b>tag</b>",
        created_at=now,
    )
    mem_text = render_memory_message(prefs, [hostile_msg], html_mode=True)
    assert "<script>" not in mem_text
    assert "&lt;script&gt;alert()&lt;/script&gt;" in mem_text
    assert "&amp;" in mem_text

    wl_text = render_watchlist_message(["BTC/USDT"], note="<b>ok</b>", html_mode=True)
    assert "&lt;b&gt;ok&lt;/b&gt;" in wl_text

    forget_text = render_forget_message(
        ForgetResult(
            deleted_messages=4,
            deleted_conversations=1,
            deleted_preferences=1,
            deleted_interactions=2,
        ),
        html_mode=True,
    )
    assert "Pesan percakapan: 4" in forget_text

    chunks = split_telegram_message("\n".join(f"item {i}: baris panjang" for i in range(500)))
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
