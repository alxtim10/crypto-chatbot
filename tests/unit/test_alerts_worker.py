from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
import pytest

from crypto_chatbot.alerts.config import (
    WIB,
    QuietHoursWib,
    ScreenerConfig,
    load_screener_config,
    load_universe,
)
from crypto_chatbot.alerts.deadman import AlertDeadManMonitor
from crypto_chatbot.alerts.sender import TelegramOutboxSender
from crypto_chatbot.alerts.worker import (
    build_alert_inline_keyboard,
    render_settlement_reply,
    run_alert_cycle,
    run_alert_digest,
)
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.handlers import BotService
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DealingZone,
    Fact,
    InputRange,
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    SmcSummary,
    TechnicalMetrics,
    TradeSetup,
)
from crypto_chatbot.schemas.llm import (
    AnalysisResponse,
    LlmAnalysisNarration,
    NarrationStatus,
)
from crypto_chatbot.schemas.market import MarketType, QualityStatus, Timeframe
from crypto_chatbot.storage.repositories import (
    AlertRecord,
    AlertRunRecord,
    AlertSymbolState,
    TradeSetupRecord,
)

BASE_TIME = datetime(2026, 11, 28, 7, 0, tzinfo=UTC)  # 14:00 WIB (outside quiet hours)
SHA = "a" * 64


class _InMemoryAlertRepo:
    def __init__(self, *, lock_busy: bool = False) -> None:
        self.lock_busy = lock_busy
        self.locked_keys: set[int] = set()
        self.runs: dict[str, AlertRunRecord] = {}
        self.alerts: dict[str, AlertRecord] = {}
        self.states: dict[tuple[str, Timeframe], AlertSymbolState] = {}

    def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        if self.lock_busy or lock_key in self.locked_keys:
            return False
        self.locked_keys.add(lock_key)
        return True

    def release_advisory_lock(self, lock_key: int) -> None:
        self.locked_keys.discard(lock_key)

    def has_completed_run(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        return run is not None and run.status == "completed"

    def record_run_start(self, run: AlertRunRecord) -> bool:
        if run.run_id in self.runs:
            return False
        self.runs[run.run_id] = run
        return True

    def record_run_finish(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: datetime,
        scanned_count: int,
        ready_count: int,
        alerted_count: int,
        held_digest_count: int,
        failed_symbols: Sequence[str],
        symbol_errors: dict[str, str],
        armed_count: int = 0,
        expired_count: int = 0,
        stage_failures: dict[str, dict[str, str]] | None = None,
    ) -> None:
        prev = self.runs[run_id]
        self.runs[run_id] = prev.model_copy(
            update={
                "status": status,
                "finished_at": finished_at,
                "scanned_count": scanned_count,
                "ready_count": ready_count,
                "armed_count": armed_count,
                "expired_count": expired_count,
                "alerted_count": alerted_count,
                "held_digest_count": held_digest_count,
                "failed_symbols": tuple(failed_symbols),
                "symbol_errors": dict(symbol_errors),
                "stage_failures": {k: dict(v) for k, v in (stage_failures or {}).items()},
            }
        )

    def get_latest_completed_run(self) -> AlertRunRecord | None:
        completed = [
            r for r in self.runs.values() if r.status == "completed" and r.finished_at is not None
        ]
        if not completed:
            return None
        return max(completed, key=lambda r: r.finished_at or datetime.min.replace(tzinfo=UTC))

    def list_recent_runs(self, *, since: datetime) -> tuple[AlertRunRecord, ...]:
        return tuple(r for r in self.runs.values() if r.started_at >= since)

    def get_symbol_state(
        self, canonical_symbol: str, timeframe: Timeframe
    ) -> AlertSymbolState | None:
        return self.states.get((canonical_symbol, timeframe))

    def upsert_symbol_state(self, state: AlertSymbolState) -> None:
        self.states[(state.canonical_symbol, state.timeframe)] = state

    def mute_symbol(
        self,
        canonical_symbol: str,
        timeframe: Timeframe,
        muted_until: datetime,
        *,
        now: datetime,
    ) -> None:
        prev = self.states.get((canonical_symbol, timeframe))
        self.states[(canonical_symbol, timeframe)] = AlertSymbolState(
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            last_status=prev.last_status if prev else "wait",
            last_bias=prev.last_bias if prev else None,
            last_run_id=prev.last_run_id if prev else None,
            last_alert_time_long=prev.last_alert_time_long if prev else None,
            last_alert_time_short=prev.last_alert_time_short if prev else None,
            muted_until=muted_until,
            updated_at=now,
        )

    def create_alert(self, alert: AlertRecord) -> bool:
        if alert.alert_id in self.alerts:
            return False
        self.alerts[alert.alert_id] = alert
        return True

    def list_pending_alerts(self, *, limit: int = 50) -> tuple[AlertRecord, ...]:
        return tuple(a for a in self.alerts.values() if a.status == "pending")[:limit]

    def list_held_digest_alerts(self, *, since: datetime | None = None) -> tuple[AlertRecord, ...]:
        rows = [a for a in self.alerts.values() if a.status == "held_digest"]
        if since is not None:
            rows = [a for a in rows if a.created_at >= since]
        return tuple(rows)

    def list_recent_alerts(self, *, since: datetime) -> tuple[AlertRecord, ...]:
        return tuple(a for a in self.alerts.values() if a.created_at >= since)

    def mark_alert_sent(
        self,
        alert_id: str,
        *,
        telegram_message_id: int,
        sent_at: datetime,
        attempts: int,
    ) -> None:
        prev = self.alerts[alert_id]
        self.alerts[alert_id] = prev.model_copy(
            update={
                "status": "sent",
                "telegram_message_id": telegram_message_id,
                "sent_at": sent_at,
                "send_attempts": attempts,
                "last_error": None,
            }
        )

    def mark_alert_failed(self, alert_id: str, *, error: str, attempts: int) -> None:
        prev = self.alerts[alert_id]
        self.alerts[alert_id] = prev.model_copy(
            update={"status": "failed", "last_error": error, "send_attempts": attempts}
        )

    def mark_alerts_digest_sent(self, alert_ids: Sequence[str], *, sent_at: datetime) -> int:
        count = 0
        for aid in alert_ids:
            if aid in self.alerts:
                self.alerts[aid] = self.alerts[aid].model_copy(
                    update={"status": "digest_sent", "sent_at": sent_at}
                )
                count += 1
        return count

    def get_alert_by_id(self, alert_id: str) -> AlertRecord | None:
        return self.alerts.get(alert_id)

    def get_alert_by_snapshot_id(self, snapshot_id: str) -> AlertRecord | None:
        for al in self.alerts.values():
            if al.snapshot_id == snapshot_id:
                return al
        return None


class _InMemorySetupRepo:
    def __init__(self) -> None:
        self.saved: dict[str, dict[str, Any]] = {}
        self.user_actions: dict[str, str] = {}
        self.telegram_links: dict[str, tuple[int, int]] = {}

    def save_setup(
        self,
        snapshot: AnalysisSnapshot,
        setup: TradeSetup,
        *,
        params_sha256: str | None = None,
    ) -> bool:
        self.saved[snapshot.snapshot_id] = {
            "symbol": snapshot.canonical_symbol,
            "status": setup.status.value,
            "params_sha256": params_sha256,
        }
        return True

    def record_user_action(
        self,
        snapshot_id: str,
        *,
        action: Literal["ambil", "skip"],
        acted_at: datetime | None = None,
    ) -> bool:
        del acted_at
        self.user_actions[snapshot_id] = action
        return True

    def link_telegram_message(
        self,
        snapshot_id: str,
        *,
        chat_id: int,
        message_id: int,
    ) -> bool:
        self.telegram_links[snapshot_id] = (chat_id, message_id)
        return True

    def get_setup(self, snapshot_id: str) -> TradeSetupRecord | None:
        del snapshot_id
        return None

    def list_pending_setups(self, *, limit: int = 100) -> tuple[TradeSetupRecord, ...]:
        del limit
        return ()

    def list_recent_setups(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 20,
    ) -> tuple[TradeSetupRecord, ...]:
        del canonical_symbol, timeframe, limit
        return ()

    def update_settlement(
        self,
        snapshot_id: str,
        *,
        outcome_status: str,
        settled_at: datetime | None,
        realized_r: float | None,
    ) -> bool:
        del snapshot_id, outcome_status, settled_at, realized_r
        return False


def _make_snapshot(
    symbol: str,
    *,
    as_of: datetime,
    status: SetupStatus,
    bias: SetupBias = SetupBias.LONG,
    rr_tp2: float = 2.5,
    timeframe: Timeframe = Timeframe.H1,
) -> AnalysisSnapshot:
    base = symbol.split("/")[0]
    snap_id = f"snap-{base.lower()}-{int(as_of.timestamp())}"
    if status is SetupStatus.READY:
        if bias is SetupBias.SHORT:
            setup = TradeSetup(
                status=SetupStatus.READY,
                bias=SetupBias.SHORT,
                entry_low=101.0,
                entry_high=102.0,
                entry_mid=101.5,
                stop_loss=103.5,
                tp1=99.5,
                tp2=96.5,
                tp3=93.5,
                rr_tp1=1.0,
                rr_tp2=rr_tp2,
                rr_tp3=4.0,
                ltf_confirmation="15m CHoCH BEARISH",
            )
        else:
            setup = TradeSetup(
                status=SetupStatus.READY,
                bias=SetupBias.LONG,
                entry_low=98.0,
                entry_high=99.0,
                entry_mid=98.5,
                stop_loss=96.5,
                tp1=100.5,
                tp2=103.5,
                tp3=106.5,
                rr_tp1=1.0,
                rr_tp2=rr_tp2,
                rr_tp3=4.0,
                ltf_confirmation="15m BOS BULLISH",
            )
    else:
        setup = TradeSetup(
            status=SetupStatus.WAIT,
            bias=SetupBias.NEUTRAL,
            wait_reason="Ranging",
        )

    return AnalysisSnapshot(
        snapshot_id=snap_id,
        provider="gate_rest",
        venue="gate",
        canonical_symbol=symbol,
        provider_symbol=f"GATE:{base}USDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=timeframe,
        as_of=as_of,
        input_range=InputRange(
            first_open_time=as_of - timedelta(hours=200),
            last_open_time=as_of - timedelta(hours=1),
            candle_count=200,
            sha256=SHA,
        ),
        calc_version="1.7.0",
        technical=TechnicalMetrics(last_close=100.0),
        smc=SmcSummary(
            dealing_range_high=105.0,
            dealing_range_low=95.0,
            equilibrium=100.0,
            dealing_zone=DealingZone.DISCOUNT,
        ),
        trade_setup=setup,
        facts=(
            Fact(
                fact_id="F1",
                label="Harga penutupan terakhir",
                value=100.0,
                unit="USDT",
                display_format="price",
                source=f"gate:{symbol}:1h",
                as_of=as_of,
            ),
        ),
        regime=Regime.TRENDING_UP,
        market_structure=MarketStructure.BULLISH,
        quality_status=QualityStatus.READY,
    )


def _make_fake_sender(
    *,
    fail_429_times: int = 0,
    retry_after: int = 2,
) -> tuple[TelegramOutboxSender, list[dict[str, Any]], list[float]]:
    sent_payloads: list[dict[str, Any]] = []
    sleeps: list[float] = []
    calls = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content.decode("utf-8"))
        if calls <= fail_429_times:
            return httpx.Response(
                429,
                json={"ok": False, "parameters": {"retry_after": retry_after}},
            )
        sent_payloads.append(body)
        return httpx.Response(
            200,
            json={"ok": True, "result": {"message_id": 5000 + len(sent_payloads)}},
        )

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    sender = TelegramOutboxSender(
        bot_token="123456:fake-token",
        transport=httpx.MockTransport(_handler),
        sleep_fn=_sleep,
    )
    return sender, sent_payloads, sleeps


def test_universe_and_screener_yaml_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    universe = load_universe()
    assert len(universe) == 21
    assert universe[0] == "BTC/USDT:USDT"
    assert "INIT/USDT:USDT" in universe

    cfg = load_screener_config()
    assert cfg.is_scalp_mode is True
    assert cfg.is_mtf_mode is False
    assert cfg.timeframes == (Timeframe.M15,)
    assert cfg.cooldown_for(Timeframe.M15) == timedelta(hours=2)
    assert cfg.cooldown_for(Timeframe.H1) == timedelta(hours=6)
    assert cfg.max_alerts_per_run == 3
    assert cfg.setup_expiry_minutes == 90
    assert cfg.dead_man_hours == 1.0
    assert cfg.effective_dead_man_minutes == 15.0

    # Quiet hours WIB wrap (23:00 - 06:00 WIB)
    qh = QuietHoursWib(enabled=True, start="23:00", end="06:00")
    assert qh.is_quiet(datetime(2026, 11, 28, 23, 30, tzinfo=WIB)) is True
    assert qh.is_quiet(datetime(2026, 11, 28, 5, 59, tzinfo=WIB)) is True
    assert qh.is_quiet(datetime(2026, 11, 28, 6, 0, tzinfo=WIB)) is False
    assert qh.is_quiet(datetime(2026, 11, 28, 14, 0, tzinfo=WIB)) is False

    monkeypatch.setenv("QUIET_HOURS", "22:00-07:00")
    cfg_env = load_screener_config()
    assert cfg_env.quiet_hours_wib.start == "22:00"
    assert cfg_env.quiet_hours_wib.end == "07:00"


@pytest.mark.asyncio
async def test_idempotent_run_id_and_advisory_lock() -> None:
    snap_btc = _make_snapshot("BTC/USDT:USDT", as_of=BASE_TIME, status=SetupStatus.READY)

    class _Svc:
        async def analyze(self, sym: str, tf: Timeframe, **kw: Any) -> AnalysisSnapshot:
            del sym, tf, kw
            return snap_btc

    alert_repo = _InMemoryAlertRepo()
    setup_repo = _InMemorySetupRepo()
    sender, sent, _ = _make_fake_sender()
    cfg = ScreenerConfig(timeframes=(Timeframe.H1,), quiet_hours_wib=QuietHoursWib(enabled=False))

    try:
        # 1. Advisory lock busy -> skips immediately
        locked_repo = _InMemoryAlertRepo(lock_busy=True)
        locked_res = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=locked_repo,
            screener_config=cfg,
            symbols=("BTC",),
            clock=lambda: BASE_TIME,
        )
        assert locked_res.locked_out is True
        assert locked_res.results == ()

        # 2. First run processes & sends alert + links setup ledger with params_sha256
        res1 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            setup_repo=setup_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: BASE_TIME,
        )
        assert res1.results[0].sent_alerts == 1
        assert len(sent) == 1
        assert snap_btc.snapshot_id in setup_repo.saved
        assert setup_repo.saved[snap_btc.snapshot_id]["params_sha256"] == res1.params_sha256
        assert setup_repo.telegram_links[snap_btc.snapshot_id] == (999, 5001)

        # 3. Re-running on exact same candle close_time skips via run_id idempotency
        res2 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            setup_repo=setup_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: BASE_TIME + timedelta(minutes=5),
        )
        assert res2.results[0].skipped_reason == "already_processed"
        assert len(sent) == 1
    finally:
        await sender.close()


@pytest.mark.asyncio
async def test_state_change_and_per_symbol_direction_cooldown() -> None:
    alert_repo = _InMemoryAlertRepo()
    sender, sent, _ = _make_fake_sender()
    cfg = ScreenerConfig(
        timeframes=(Timeframe.H1,),
        cooldown_hours={"1h": 6.0},
        quiet_hours_wib=QuietHoursWib(enabled=False),
    )

    current_snaps: dict[str, AnalysisSnapshot] = {}

    class _Svc:
        async def analyze(self, sym: str, tf: Timeframe, **kw: Any) -> AnalysisSnapshot:
            del tf, kw
            return current_snaps[sym]

    try:
        # T+0h: BTC becomes READY LONG -> alert #1 sent
        t0 = BASE_TIME
        current_snaps["BTC/USDT:USDT"] = _make_snapshot(
            "BTC/USDT:USDT", as_of=t0, status=SetupStatus.READY, bias=SetupBias.LONG
        )
        r0 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: t0,
        )
        assert r0.results[0].sent_alerts == 1

        # T+1h: BTC STILL READY LONG -> NO alert (no state change!)
        t1 = t0 + timedelta(hours=1)
        current_snaps["BTC/USDT:USDT"] = _make_snapshot(
            "BTC/USDT:USDT", as_of=t1, status=SetupStatus.READY, bias=SetupBias.LONG
        )
        r1 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: t1,
        )
        assert r1.results[0].sent_alerts == 0

        # T+2h: BTC moves to WAIT -> state resets to wait
        t2 = t0 + timedelta(hours=2)
        current_snaps["BTC/USDT:USDT"] = _make_snapshot(
            "BTC/USDT:USDT", as_of=t2, status=SetupStatus.WAIT
        )
        await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: t2,
        )

        # T+3h: BTC transitions WAIT -> READY LONG inside 6h cooldown -> suppressed!
        t3 = t0 + timedelta(hours=3)
        current_snaps["BTC/USDT:USDT"] = _make_snapshot(
            "BTC/USDT:USDT", as_of=t3, status=SetupStatus.READY, bias=SetupBias.LONG
        )
        r3 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: t3,
        )
        assert r3.results[0].sent_alerts == 0

        # T+4h: BTC flips to READY SHORT (different direction!) -> allowed -> alert #2!
        t4 = t0 + timedelta(hours=4)
        current_snaps["BTC/USDT:USDT"] = _make_snapshot(
            "BTC/USDT:USDT", as_of=t4, status=SetupStatus.READY, bias=SetupBias.SHORT
        )
        r4 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: t4,
        )
        assert r4.results[0].sent_alerts == 1

        # T+7h: BTC flips back to READY LONG (>= 6h after T+0h) -> allowed -> alert #3!
        t7 = t0 + timedelta(hours=7)
        current_snaps["BTC/USDT:USDT"] = _make_snapshot(
            "BTC/USDT:USDT", as_of=t7, status=SetupStatus.READY, bias=SetupBias.LONG
        )
        r7 = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=("BTC",),
            chat_id=999,
            clock=lambda: t7,
        )
        assert r7.results[0].sent_alerts == 1
        assert len(sent) == 3
    finally:
        await sender.close()


@pytest.mark.asyncio
async def test_outbox_429_retry_per_symbol_isolation_max_cap_and_digest() -> None:
    # 6 symbols: 5 READY + 1 ERROR; max_alerts_per_run=4 -> 4 sent, 1 held_digest, 1 ERROR
    syms = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"]

    class _Svc:
        async def analyze(self, sym: str, tf: Timeframe, **kw: Any) -> AnalysisSnapshot:
            del tf, kw
            if sym == "DOGE/USDT:USDT":
                raise RuntimeError("gate timeout on DOGE")
            return _make_snapshot(sym, as_of=BASE_TIME, status=SetupStatus.READY, rr_tp2=2.5)

    alert_repo = _InMemoryAlertRepo()
    sender, sent, sleeps = _make_fake_sender(fail_429_times=2, retry_after=3)
    cfg = ScreenerConfig(
        timeframes=(Timeframe.H1,),
        max_alerts_per_run=4,
        quiet_hours_wib=QuietHoursWib(enabled=False),
    )

    try:
        res = await run_alert_cycle(
            analysis_service=_Svc(),  # type: ignore[arg-type]
            alert_repo=alert_repo,
            sender=sender,
            screener_config=cfg,
            symbols=syms,
            chat_id=999,
            clock=lambda: BASE_TIME,
        )
        tf_res = res.results[0]
        assert tf_res.sent_alerts == 4
        assert tf_res.held_digest_alerts == 1
        assert tf_res.failed_symbols == ("DOGE/USDT:USDT",)
        assert "ERROR" in tf_res.symbol_errors["DOGE/USDT:USDT"]
        # Verified 429 retry_after slept twice for 3.0s before delivering
        assert sleeps == [3.0, 3.0]
        assert len(sent) == 4
        # Inline buttons present on sent messages
        kb = sent[0]["reply_markup"]["inline_keyboard"]
        btn_texts = [btn["text"] for row in kb for btn in row]
        assert btn_texts == ["Detail", "Jelaskan", "Ambil", "Skip", "Mute 24j"]

        # Daily digest picks up the 1 held alert and marks it digest_sent
        digest = await run_alert_digest(
            alert_repo=alert_repo,
            sender=sender,
            chat_id=999,
            clock=lambda: BASE_TIME + timedelta(hours=2),
        )
        assert digest.sent is True
        assert digest.held_alerts_included == 1
        assert "Ringkasan Harian Alert Setup" in digest.message_text
        assert alert_repo.list_held_digest_alerts() == ()
    finally:
        await sender.close()


@pytest.mark.asyncio
async def test_inline_callbacks_settlement_reply_and_deadman_monitor() -> None:
    alert_repo = _InMemoryAlertRepo()
    setup_repo = _InMemorySetupRepo()
    snap = _make_snapshot("BTC/USDT:USDT", as_of=BASE_TIME, status=SetupStatus.READY)
    alert_id = "al-1234567890ab"
    alert_repo.create_alert(
        AlertRecord(
            alert_id=alert_id,
            run_id="1h:2026-11-28T07:00:00Z",
            snapshot_id=snap.snapshot_id,
            params_sha256="abc123",
            canonical_symbol="BTC/USDT:USDT",
            timeframe=Timeframe.H1,
            direction="long",
            venue="GATE",
            chat_id=999,
            status="sent",
            message_text="test",
            reply_markup_json=build_alert_inline_keyboard(alert_id),
            telegram_message_id=5001,
            created_at=BASE_TIME,
            sent_at=BASE_TIME,
        )
    )

    class _FakeAnalysis:
        async def analyze(self, sym: str, tf: Timeframe, **kw: Any) -> AnalysisSnapshot:
            del sym, tf, kw
            return snap

    class _FakeOutcome:
        def __init__(self, response: AnalysisResponse) -> None:
            self.response = response

    class _FakeNarrator:
        async def narrate_snapshot(self, snapshot: AnalysisSnapshot, **kw: Any) -> _FakeOutcome:
            del kw
            return _FakeOutcome(
                response=AnalysisResponse(
                    snapshot=snapshot,
                    narration_status=NarrationStatus.GENERATED,
                    narration=LlmAnalysisNarration(
                        summary="Penjelasan berbasis {F1}.",
                        bullish_scenario="Skenario naik mengikuti {F1}.",
                        bearish_scenario="Skenario turun jika struktur batal.",
                        fact_ids=("F1",),
                    ),
                )
            )

    class _FakeMemory:
        pass

    bot = BotService(
        access=AccessController((111,)),
        analysis_service=_FakeAnalysis(),  # type: ignore[arg-type]
        narrator=_FakeNarrator(),  # type: ignore[arg-type]
        memory=_FakeMemory(),  # type: ignore[arg-type]
        alert_repository=alert_repo,
        setup_repository=setup_repo,
        clock=lambda: BASE_TIME,
    )

    # [Ambil] & [Skip] stored in ledger
    toast_tk, chunks_tk = await bot.handle_alert_callback(111, f"al:tk:{alert_id}")
    assert "AMBIL" in toast_tk
    assert setup_repo.user_actions[snap.snapshot_id] == "ambil"
    assert "AMBIL" in chunks_tk[0]

    toast_sk, _ = await bot.handle_alert_callback(111, f"al:sk:{alert_id}")
    assert "SKIP" in toast_sk
    assert setup_repo.user_actions[snap.snapshot_id] == "skip"

    # [Mute 24j] mutes symbol for 24h
    toast_mt, _ = await bot.handle_alert_callback(111, f"al:mt:{alert_id}")
    assert "24" in toast_mt
    st = alert_repo.get_symbol_state("BTC/USDT:USDT", Timeframe.H1)
    assert st is not None
    assert st.muted_until == BASE_TIME + timedelta(hours=24)

    # [Detail] renders deterministic snapshot
    toast_dt, chunks_dt = await bot.handle_alert_callback(111, f"al:dt:{alert_id}")
    assert "Detail" in toast_dt
    assert "BTC/USDT:USDT" in chunks_dt[0]

    # Settlement thread reply text
    rec = TradeSetupRecord(
        snapshot_id=snap.snapshot_id,
        venue="gate",
        canonical_symbol="BTC/USDT:USDT",
        timeframe=Timeframe.H1,
        as_of=BASE_TIME,
        status="ready",
        bias="long",
        reference_price=100.0,
    )
    reply_txt = render_settlement_reply(
        rec,
        outcome_status="tp2_hit",
        realized_r=2.0,
        settled_at=BASE_TIME + timedelta(hours=3),
    )
    assert "TP2 TERCAPAI" in reply_txt
    assert "+2,00R" in reply_txt

    # Dead-man monitor sends 🔧 when no completed run in 2h, avoids duplicate spam
    sender, dm_sent, _ = _make_fake_sender()
    try:
        monitor = AlertDeadManMonitor(
            alert_repo=alert_repo,
            sender=sender,
            chat_id=999,
            max_silence_hours=2.0,
            clock=lambda: BASE_TIME + timedelta(hours=3),
        )
        assert await monitor.check_once() is True
        assert "🔧" in dm_sent[0]["text"]
        # Second check immediately after does NOT spam
        assert await monitor.check_once() is False
    finally:
        await sender.close()
