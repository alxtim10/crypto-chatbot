"""Unit tests for the 15-minute multi-timeframe top-down alert flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_chatbot.alerts.config import (
    MtfTimeframesConfig,
    QuietHoursWib,
    ScreenerConfig,
    load_screener_config,
)
from crypto_chatbot.alerts.deadman import AlertDeadManMonitor
from crypto_chatbot.alerts.mtf import (
    MixedVenueError,
    assert_single_venue_bundle,
    classify_setup_wait_stage,
    evaluate_5m_confirmation,
    make_mtf_run_id,
)
from crypto_chatbot.alerts.worker import run_alert_cycle, run_alert_digest
from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.schemas.analysis import (
    SetupBias,
    SetupStatus,
    TradeSetup,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    Timeframe,
)
from crypto_chatbot.storage.repositories import (
    AlertRecord,
    AlertRunRecord,
    AlertSymbolState,
)

BASE_TIME = datetime(2026, 11, 28, 7, 0, tzinfo=UTC)  # 14:00 WIB (outside quiet hours)


class _MemAlertRepo:
    def __init__(self) -> None:
        self.runs: dict[str, AlertRunRecord] = {}
        self.alerts: dict[str, AlertRecord] = {}
        self.states: dict[tuple[str, Timeframe], AlertSymbolState] = {}
        self.locked = False

    def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        del lock_key
        if self.locked:
            return False
        self.locked = True
        return True

    def release_advisory_lock(self, lock_key: int) -> None:
        del lock_key
        self.locked = False

    def has_completed_run(self, run_id: str) -> bool:
        r = self.runs.get(run_id)
        return r is not None and r.status == "completed"

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
        existing = self.runs[run_id]
        self.runs[run_id] = existing.model_copy(
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
        done = [r for r in self.runs.values() if r.status == "completed" and r.finished_at]
        if not done:
            return None
        done.sort(key=lambda r: r.finished_at or datetime.min.replace(tzinfo=UTC), reverse=True)
        return done[0]

    def list_recent_runs(self, *, since: datetime) -> tuple[AlertRunRecord, ...]:
        return tuple(
            sorted(
                (r for r in self.runs.values() if r.started_at >= since),
                key=lambda r: r.started_at,
                reverse=True,
            )
        )

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
        prev = self.get_symbol_state(canonical_symbol, timeframe)
        self.states[(canonical_symbol, timeframe)] = AlertSymbolState(
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            last_status=prev.last_status if prev else "wait",
            last_bias=prev.last_bias if prev else None,
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
        return tuple(
            a
            for a in self.alerts.values()
            if a.status == "held_digest" and (since is None or a.created_at >= since)
        )

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
        self.alerts[alert_id] = self.alerts[alert_id].model_copy(
            update={
                "status": "sent",
                "telegram_message_id": telegram_message_id,
                "sent_at": sent_at,
                "send_attempts": attempts,
            }
        )

    def mark_alert_failed(self, alert_id: str, *, error: str, attempts: int) -> None:
        self.alerts[alert_id] = self.alerts[alert_id].model_copy(
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
        for a in self.alerts.values():
            if a.snapshot_id == snapshot_id:
                return a
        return None


class _MemSetupRepo:
    def __init__(self) -> None:
        self.saved: dict[str, tuple[str | None, int | None]] = {}

    def save_setup(self, snap: Any, setup: Any, *, params_sha256: str | None = None) -> bool:
        del setup
        self.saved[snap.snapshot_id] = (params_sha256, None)
        return True

    def link_telegram_message(self, snapshot_id: str, *, chat_id: int, message_id: int) -> bool:
        del chat_id
        if snapshot_id in self.saved:
            sha, _ = self.saved[snapshot_id]
            self.saved[snapshot_id] = (sha, message_id)
            return True
        return False


class _FakeSender:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self._seq = 9000

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: dict[str, Any] | None = None,
        reply_to_message_id: int | None = None,
    ) -> tuple[int, int]:
        del parse_mode, reply_markup, reply_to_message_id
        self._seq += 1
        self.sent.append((chat_id, text))
        return self._seq, 1


def _make_trend_candles(
    *,
    symbol: str = "BTC/USDT:USDT",
    venue: str = "gate",
    timeframe: Timeframe,
    end_close: datetime,
    count: int = 80,
    start_price: float = 100.0,
    step: float = 0.5,
) -> tuple[MarketCandle, ...]:
    """Generate deterministic uptrend (step>0) or downtrend (step<0) closed candles."""
    dur = timeframe.duration
    candles: list[MarketCandle] = []
    for i in range(count):
        close_t = end_close - dur * (count - 1 - i)
        open_t = close_t - dur
        # Small wave pattern so swing highs/lows form cleanly for HH/HL structure
        wave = (i % 4) * (0.25 if step >= 0 else -0.25)
        base = start_price + i * step + wave
        o = base
        c = base + (0.4 if step >= 0 else -0.4)
        h = max(o, c) + 0.3
        lo = min(o, c) - 0.3
        candles.append(
            MarketCandle(
                provider="gate_direct" if venue == "gate" else venue,
                venue=venue,
                canonical_symbol=symbol,
                provider_symbol="BTC_USDT",
                market_type=MarketType.PERPETUAL,
                timeframe=timeframe,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=o,
                high=h,
                low=lo,
                close=c,
                base_volume=100.0,
                quote_volume=100.0 * c,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return tuple(candles)


def _make_batch(
    candles: tuple[MarketCandle, ...],
    *,
    venue: str,
    timeframe: Timeframe,
    symbol: str = "BTC/USDT:USDT",
) -> OhlcvBatch:
    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider="gate_direct" if venue == "gate" else venue,
            venue=venue,
            market_type=MarketType.PERPETUAL,
            base_unit="BTC",
            quote_currency="USDT",
        ),
        canonical_symbol=symbol,
        provider_symbol="BTC_USDT",
        timeframe=timeframe,
        received_time=candles[-1].close_time,
        candles=candles,
    )


def _make_5m_batch(
    *,
    end_close: datetime,
    confirmed_long: bool,
    venue: str = "gate",
    entry_low: float = 140.0,
    entry_high: float = 141.0,
) -> OhlcvBatch:
    dur = Timeframe.M5.duration
    c1 = MarketCandle(
        provider="gate_direct" if venue == "gate" else venue,
        venue=venue,
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.M5,
        open_time=end_close - dur * 2,
        close_time=end_close - dur,
        received_time=end_close - dur,
        available_time=end_close - dur,
        open=entry_high,
        high=entry_high + 0.4,
        low=entry_low,
        close=entry_low + 0.2,
        base_volume=50.0,
        quote_volume=50.0 * entry_low,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )
    if confirmed_long:
        # Bullish close above prev high
        c2_open = entry_low + 0.1
        c2_close = entry_high + 0.8
        c2_low = entry_low - 0.1
        c2_high = c2_close + 0.2
    else:
        # Bearish / doji bar — no bullish confirmation, above SL
        c2_open = entry_high + 0.2
        c2_close = entry_low + 0.2
        c2_low = entry_low + 0.1
        c2_high = entry_high + 0.3
    c2 = MarketCandle(
        provider="gate_direct" if venue == "gate" else venue,
        venue=venue,
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.M5,
        open_time=end_close - dur,
        close_time=end_close,
        received_time=end_close,
        available_time=end_close,
        open=c2_open,
        high=c2_high,
        low=c2_low,
        close=c2_close,
        base_volume=55.0,
        quote_volume=55.0 * c2_close,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )
    return _make_batch((c1, c2), venue=venue, timeframe=Timeframe.M5)


def test_screener_yaml_mtf_schema_and_defaults() -> None:
    cfg = load_screener_config("config/screener.yaml")
    assert cfg.is_mtf_mode is True
    assert cfg.mtf_timeframes is not None
    assert cfg.mtf_timeframes.bias == (Timeframe.H4, Timeframe.H1)
    assert cfg.mtf_timeframes.setup is Timeframe.M15
    assert cfg.mtf_timeframes.confirm is Timeframe.M5
    assert cfg.cadence_minutes == 15
    assert cfg.setup_expiry_minutes == 90
    assert cfg.cooldown_for(Timeframe.M15) == timedelta(hours=2)
    assert cfg.dead_man_hours == 1.0
    assert cfg.quiet_hours_wib.enabled is True
    assert cfg.quiet_hours_wib.start == "23:00"
    assert cfg.quiet_hours_wib.end == "06:00"


@pytest.mark.asyncio
async def test_mtf_armed_to_ready_and_idempotent_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run 1 stores ARMED; Run 2 promotes ARMED -> READY on 5m confirm; duplicate run_id skips."""
    import crypto_chatbot.alerts.mtf as mtf_mod

    # Stub HTF bias + 15m trade setup deterministically so we isolate the MTF state machine.
    def _fake_bias(*args: Any, **kwargs: Any) -> mtf_mod.MtfBiasEvaluation:
        del args, kwargs
        return mtf_mod.MtfBiasEvaluation(
            agreed_direction="long",
            per_tf_summary={"4h": "trending_up/bullish", "1h": "trending_up/bullish"},
            reason="HTF selaras LONG (4h=trending_up/bullish, 1h=trending_up/bullish)",
        )

    def _fake_build_setup(*args: Any, **kwargs: Any) -> TradeSetup:
        del args, kwargs
        return TradeSetup(
            status=SetupStatus.READY,
            bias=SetupBias.LONG,
            entry_timeframe=Timeframe.M15,
            entry_low=140.0,
            entry_high=141.0,
            entry_mid=140.5,
            stop_loss=136.0,
            tp1=145.0,
            tp2=150.0,
            tp3=155.0,
            rr_tp1=1.0,
            rr_tp2=2.0,
            rr_tp3=3.0,
        )

    monkeypatch.setattr(mtf_mod, "evaluate_htf_bias", _fake_bias)
    monkeypatch.setattr(mtf_mod, "build_trade_setup", _fake_build_setup)

    t1 = BASE_TIME  # 07:00 UTC
    t2 = BASE_TIME + timedelta(minutes=15)  # 07:15 UTC (< 90m expiry)

    class _FakeAnalysisService:
        def __init__(self) -> None:
            self.now = t1
            self.confirm_5m = False

        async def fetch_mtf_candle_bundle(
            self,
            symbol: str,
            timeframe_limits: Mapping[Timeframe, int],
            *,
            as_of: datetime | None = None,
            incremental: bool = True,
        ) -> tuple[dict[Timeframe, OhlcvBatch], str]:
            del timeframe_limits, as_of, incremental
            b4h = _make_batch(
                _make_trend_candles(symbol=symbol, timeframe=Timeframe.H4, end_close=self.now),
                venue="gate",
                timeframe=Timeframe.H4,
                symbol=symbol,
            )
            b1h = _make_batch(
                _make_trend_candles(symbol=symbol, timeframe=Timeframe.H1, end_close=self.now),
                venue="gate",
                timeframe=Timeframe.H1,
                symbol=symbol,
            )
            b15 = _make_batch(
                _make_trend_candles(symbol=symbol, timeframe=Timeframe.M15, end_close=self.now),
                venue="gate",
                timeframe=Timeframe.M15,
                symbol=symbol,
            )
            b5 = _make_5m_batch(end_close=self.now, confirmed_long=self.confirm_5m)
            return {
                Timeframe.H4: b4h,
                Timeframe.H1: b1h,
                Timeframe.M15: b15,
                Timeframe.M5: b5,
            }, "gate"

    cfg = ScreenerConfig(
        mtf_timeframes=MtfTimeframesConfig(
            bias=(Timeframe.H4, Timeframe.H1),
            setup=Timeframe.M15,
            confirm=Timeframe.M5,
        ),
        timeframes=(Timeframe.M15,),
        setup_expiry_minutes=90,
        quiet_hours_wib=QuietHoursWib(enabled=False),
    )
    repo = _MemAlertRepo()
    setup_repo = _MemSetupRepo()
    sender = _FakeSender()
    svc = _FakeAnalysisService()

    # Run 1 (07:00 UTC): 5m not confirmed -> ARMED, 0 alerts sent
    res1 = await run_alert_cycle(
        analysis_service=svc,  # type: ignore[arg-type]
        alert_repo=repo,
        setup_repo=setup_repo,  # type: ignore[arg-type]
        sender=sender,  # type: ignore[arg-type]
        screener_config=cfg,
        symbols=("BTC/USDT:USDT",),
        chat_id=999,
        clock=lambda: t1,
    )
    assert res1.results[0].run_id == make_mtf_run_id(t1)
    assert res1.results[0].run_id == "mtf:2026-11-28T07:00:00Z"
    assert res1.results[0].armed_count == 1
    assert res1.results[0].ready_count == 0
    assert res1.results[0].sent_alerts == 0
    assert len(sender.sent) == 0
    st1 = repo.get_symbol_state("BTC/USDT:USDT", Timeframe.M15)
    assert st1 is not None
    assert st1.last_status == "armed"
    assert st1.armed_expires_at == t1 + timedelta(minutes=90)

    # Idempotent replay on same 15m candle (07:00 UTC) -> skipped!
    res1_dup = await run_alert_cycle(
        analysis_service=svc,  # type: ignore[arg-type]
        alert_repo=repo,
        setup_repo=setup_repo,  # type: ignore[arg-type]
        sender=sender,  # type: ignore[arg-type]
        screener_config=cfg,
        symbols=("BTC/USDT:USDT",),
        chat_id=999,
        clock=lambda: t1 + timedelta(seconds=30),
    )
    assert res1_dup.results[0].skipped_reason == "already_processed"
    assert len(sender.sent) == 0

    # Run 2 (07:15 UTC): 5m confirms -> ARMED promoted to READY + alert sent + ledger linked
    svc.now = t2
    svc.confirm_5m = True
    res2 = await run_alert_cycle(
        analysis_service=svc,  # type: ignore[arg-type]
        alert_repo=repo,
        setup_repo=setup_repo,  # type: ignore[arg-type]
        sender=sender,  # type: ignore[arg-type]
        screener_config=cfg,
        symbols=("BTC/USDT:USDT",),
        chat_id=999,
        clock=lambda: t2,
    )
    assert res2.results[0].run_id == "mtf:2026-11-28T07:15:00Z"
    assert res2.results[0].ready_count == 1
    assert res2.results[0].sent_alerts == 1
    assert len(sender.sent) == 1
    st2 = repo.get_symbol_state("BTC/USDT:USDT", Timeframe.M15)
    assert st2 is not None
    assert st2.last_status == "ready"
    assert st2.armed_setup_json is None
    assert len(setup_repo.saved) == 1
    saved_sha, saved_msg_id = next(iter(setup_repo.saved.values()))
    assert saved_sha == res2.params_sha256
    assert saved_msg_id == 9001


@pytest.mark.asyncio
async def test_mtf_armed_setup_expires_after_setup_expiry_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ARMED setup that receives no 5m confirmation before expiry is cleared and counted."""
    import crypto_chatbot.alerts.mtf as mtf_mod

    state_flag = {"allow_setup": True}

    def _fake_bias(*args: Any, **kwargs: Any) -> mtf_mod.MtfBiasEvaluation:
        del args, kwargs
        return mtf_mod.MtfBiasEvaluation(
            agreed_direction="long" if state_flag["allow_setup"] else None,
            per_tf_summary={"4h": "trending_up/bullish", "1h": "trending_up/bullish"},
            reason="HTF selaras LONG" if state_flag["allow_setup"] else "Bias HTF tidak sepakat",
        )

    def _fake_build_setup(*args: Any, **kwargs: Any) -> TradeSetup:
        del args, kwargs
        return TradeSetup(
            status=SetupStatus.READY,
            bias=SetupBias.LONG,
            entry_timeframe=Timeframe.M15,
            entry_low=140.0,
            entry_high=141.0,
            entry_mid=140.5,
            stop_loss=136.0,
            tp1=145.0,
            tp2=150.0,
            tp3=155.0,
            rr_tp1=1.0,
            rr_tp2=2.0,
            rr_tp3=3.0,
        )

    monkeypatch.setattr(mtf_mod, "evaluate_htf_bias", _fake_bias)
    monkeypatch.setattr(mtf_mod, "build_trade_setup", _fake_build_setup)

    class _Svc:
        def __init__(self, now: datetime) -> None:
            self.now = now

        async def fetch_mtf_candle_bundle(
            self,
            symbol: str,
            timeframe_limits: Mapping[Timeframe, int],
            *,
            as_of: datetime | None = None,
            incremental: bool = True,
        ) -> tuple[dict[Timeframe, OhlcvBatch], str]:
            del timeframe_limits, as_of, incremental
            return {
                Timeframe.H4: _make_batch(
                    _make_trend_candles(symbol=symbol, timeframe=Timeframe.H4, end_close=self.now),
                    venue="gate",
                    timeframe=Timeframe.H4,
                    symbol=symbol,
                ),
                Timeframe.H1: _make_batch(
                    _make_trend_candles(symbol=symbol, timeframe=Timeframe.H1, end_close=self.now),
                    venue="gate",
                    timeframe=Timeframe.H1,
                    symbol=symbol,
                ),
                Timeframe.M15: _make_batch(
                    _make_trend_candles(symbol=symbol, timeframe=Timeframe.M15, end_close=self.now),
                    venue="gate",
                    timeframe=Timeframe.M15,
                    symbol=symbol,
                ),
                Timeframe.M5: _make_5m_batch(end_close=self.now, confirmed_long=False),
            }, "gate"

    cfg = ScreenerConfig(
        mtf_timeframes=MtfTimeframesConfig(),
        timeframes=(Timeframe.M15,),
        setup_expiry_minutes=90,
        quiet_hours_wib=QuietHoursWib(enabled=False),
    )
    repo = _MemAlertRepo()
    sender = _FakeSender()
    svc = _Svc(BASE_TIME)

    # Run 1 -> ARMED
    r1 = await run_alert_cycle(
        analysis_service=svc,  # type: ignore[arg-type]
        alert_repo=repo,
        sender=sender,  # type: ignore[arg-type]
        screener_config=cfg,
        symbols=("BTC/USDT:USDT",),
        chat_id=999,
        clock=lambda: BASE_TIME,
    )
    assert r1.results[0].armed_count == 1

    # Run 2 at +105 minutes (past 90m expiry), bias lost -> expired!
    t_exp = BASE_TIME + timedelta(minutes=105)
    svc.now = t_exp
    state_flag["allow_setup"] = False
    r2 = await run_alert_cycle(
        analysis_service=svc,  # type: ignore[arg-type]
        alert_repo=repo,
        sender=sender,  # type: ignore[arg-type]
        screener_config=cfg,
        symbols=("BTC/USDT:USDT",),
        chat_id=999,
        clock=lambda: t_exp,
    )
    assert r2.results[0].expired_count == 1
    assert r2.results[0].sent_alerts == 0
    st = repo.get_symbol_state("BTC/USDT:USDT", Timeframe.M15)
    assert st is not None
    assert st.last_status == "wait"
    assert st.armed_setup_json is None


@pytest.mark.asyncio
async def test_mtf_mixed_venue_prevention_and_digest_stage_breakdown() -> None:
    """Mixed venues across timeframes are rejected cleanly and reported in digest stage summary."""
    b4h = _make_batch(
        _make_trend_candles(timeframe=Timeframe.H4, end_close=BASE_TIME, venue="gate"),
        venue="gate",
        timeframe=Timeframe.H4,
    )
    b5m_other = _make_5m_batch(end_close=BASE_TIME, confirmed_long=True, venue="bybit")

    with pytest.raises(MixedVenueError):
        assert_single_venue_bundle({Timeframe.H4: b4h, Timeframe.M5: b5m_other})

    # Verify AnalysisService.fetch_mtf_candle_bundle also enforces single-venue guard.
    class _MixedProvider:
        async def get_closed_ohlcv(
            self, symbol: str, timeframe: Timeframe, limit: int = 100
        ) -> OhlcvBatch:
            del symbol, limit
            v = "bybit" if timeframe is Timeframe.M5 else "gate"
            return _make_batch(
                _make_trend_candles(timeframe=timeframe, end_close=BASE_TIME, venue=v),
                venue=v,
                timeframe=timeframe,
            )

    svc = AnalysisService(provider=_MixedProvider())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mixed_venue_detected"):
        await svc.fetch_mtf_candle_bundle(
            "BTC/USDT:USDT",
            {Timeframe.H4: 40, Timeframe.M15: 40, Timeframe.M5: 20},
            as_of=BASE_TIME,
            incremental=False,
        )

    # Full cycle isolates the mixed-venue error per symbol and digest reports stage breakdown.
    cfg = ScreenerConfig(
        mtf_timeframes=MtfTimeframesConfig(),
        timeframes=(Timeframe.M15,),
        quiet_hours_wib=QuietHoursWib(enabled=False),
    )
    repo = _MemAlertRepo()
    sender = _FakeSender()
    res = await run_alert_cycle(
        analysis_service=svc,
        alert_repo=repo,
        sender=sender,  # type: ignore[arg-type]
        screener_config=cfg,
        symbols=("BTC/USDT:USDT",),
        chat_id=999,
        clock=lambda: BASE_TIME,
    )
    assert res.results[0].failed_symbols == ("BTC/USDT:USDT",)
    assert res.results[0].stage_failures["BTC/USDT:USDT"]["stage"] == "error"

    digest = await run_alert_digest(
        alert_repo=repo,
        sender=sender,  # type: ignore[arg-type]
        chat_id=999,
        clock=lambda: BASE_TIME + timedelta(hours=1),
    )
    assert "Penyaringan 24j" in digest.message_text
    assert "Error Data/Venue=1" in digest.message_text


def test_cost_gate_classification_5m_confirmation_and_deadman_default_1h() -> None:
    assert classify_setup_wait_stage("Biaya eksekusi melebihi ambang 0.15R") == "cost_gate"
    assert classify_setup_wait_stage("R:R bersih di bawah minimum 1.5") == "cost_gate"
    assert classify_setup_wait_stage("Tidak ada zona order block") == "setup"

    # Unclosed future 5m candle must be ignored by evaluate_5m_confirmation
    b5 = _make_5m_batch(end_close=BASE_TIME, confirmed_long=True)
    future_cutoff = BASE_TIME - timedelta(minutes=1)
    confirmed, invalidated, _ = evaluate_5m_confirmation(
        b5.candles,
        direction="long",
        entry_low=140.0,
        entry_high=141.0,
        stop_loss=136.0,
        cutoff=future_cutoff,
    )
    assert confirmed is False
    assert invalidated is False

    # Dead-man default is 1 hour for the 15m cadence
    repo = _MemAlertRepo()
    sender = _FakeSender()
    monitor = AlertDeadManMonitor(
        alert_repo=repo,
        sender=sender,  # type: ignore[arg-type]
        chat_id=999,
    )
    assert monitor._max_silence == timedelta(hours=1)
