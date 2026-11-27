"""Bulk historical archive client, parser, and loader for download.gatedata.org (D-04)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import itertools
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from crypto_chatbot.market_data.gate_contract_stats import to_gate_contract_symbol
from crypto_chatbot.market_data.symbols import SymbolMapping
from crypto_chatbot.schemas.derivatives import FundingRateObservation
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityFlag,
    Timeframe,
)
from crypto_chatbot.storage.repositories import (
    DerivativesRepository,
    GateIngestRepository,
    MarketRepository,
)

DEFAULT_GATEDATA_BASE_URL = "https://download.gatedata.org/futures_usdt"
DEFAULT_GATE_RAW_DIR = Path("data/raw/gate")

# Empirically verified 200 OK candlestick intervals on download.gatedata.org/futures_usdt:
VERIFIED_GATEDATA_CANDLESTICK_TYPES: dict[str, Timeframe | None] = {
    "candlesticks_10s": None,  # Archived as raw CSV (no sub-1m MarketCandle timeframe)
    "candlesticks_1m": Timeframe.M1,
    "candlesticks_5m": Timeframe.M5,
    "candlesticks_1h": Timeframe.H1,
    "candlesticks_4h": Timeframe.H4,
    "candlesticks_1d": Timeframe.D1,
}

# Verified 404 on download.gatedata.org/futures_usdt (must resample from 1m/5m/1h instead):
UNAVAILABLE_GATEDATA_CANDLESTICK_TYPES: frozenset[str] = frozenset(
    {"candlesticks_15m", "candlesticks_30m", "candlesticks_8h"}
)

VERIFIED_GATEDATA_TYPES: frozenset[str] = frozenset(
    {
        *VERIFIED_GATEDATA_CANDLESTICK_TYPES.keys(),
        "trades",
        "mark_prices",
        "funding_applies",
    }
)

_YYYYMM_RE = re.compile(r"^(\d{4})(0[1-9]|1[0-2])$")


def parse_yyyymm(value: str) -> tuple[int, int]:
    """Validate YYYYMM string and return `(year, month)`."""
    match = _YYYYMM_RE.match(value.strip())
    if match is None:
        raise ValueError(f"invalid YYYYMM month '{value}' (expected e.g. 202610)")
    return int(match.group(1)), int(match.group(2))


def iter_months(from_ym: str, to_ym: str) -> tuple[str, ...]:
    """Generate inclusive sequence of `YYYYMM` strings from `from_ym` to `to_ym`."""
    sy, sm = parse_yyyymm(from_ym)
    ey, em = parse_yyyymm(to_ym)
    if (sy, sm) > (ey, em):
        raise ValueError(f"--from ({from_ym}) must not follow --to ({to_ym})")
    months: list[str] = []
    year, month = sy, sm
    while (year, month) <= (ey, em):
        months.append(f"{year:04d}{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return tuple(months)


def validate_dataset_type(dataset: str) -> str:
    """Validate requested download.gatedata.org dataset type against verified catalog."""
    cleaned = dataset.strip().lower()
    if cleaned in UNAVAILABLE_GATEDATA_CANDLESTICK_TYPES:
        raise ValueError(
            f"dataset '{cleaned}' does not exist (404) on download.gatedata.org; "
            "use candlesticks_1m / candlesticks_5m / candlesticks_1h and resample if needed"
        )
    if cleaned not in VERIFIED_GATEDATA_TYPES:
        raise ValueError(
            f"unsupported gatedata dataset '{dataset}'; "
            f"verified types: {sorted(VERIFIED_GATEDATA_TYPES)}"
        )
    return cleaned


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 checksum of an archive file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GateArchiveTrade:
    """Parsed trade row from `download.gatedata.org/futures_usdt/trades` or REST `/trades`."""

    timestamp: datetime
    trade_id: int
    price: float
    signed_contracts: float

    @property
    def is_taker_buy(self) -> bool:
        return self.signed_contracts > 0.0

    @property
    def abs_contracts(self) -> float:
        return abs(self.signed_contracts)


def parse_gatedata_candlesticks_csv(
    lines: Iterable[str],
    *,
    symbol: SymbolMapping,
    contract_id: str,
    timeframe: Timeframe,
    quanto_multiplier: Decimal | float | str,
    received_time: datetime | None = None,
    venue: str = "gate",
) -> tuple[MarketCandle, ...]:
    """Parse Gate bulk candlestick CSV lines into closed `MarketCandle` instances.

    Verified CSV layout: `[0]=ts_s, [1]=contracts, [2]=close, [3]=high, [4]=low, [5]=open`.
    Backfilled candles set `available_time = close_time` and `BACKFILL` + `QUOTE_VOL_ESTIMATED`.
    """
    mult = Decimal(str(quanto_multiplier))
    if mult <= 0:
        raise ValueError("quanto_multiplier must be positive")
    effective_received = (received_time or datetime.now(UTC)).astimezone(UTC)
    provider_symbol = f"GATE:{symbol.base}{symbol.quote}.P"

    by_open: dict[datetime, MarketCandle] = {}
    for row in csv.reader(lines):
        if not row or len(row) < 6:
            continue
        ts_val = int(float(row[0].strip()))
        contracts_dec = Decimal(row[1].strip())
        close_p = float(Decimal(row[2].strip()))
        high_p = float(Decimal(row[3].strip()))
        low_p = float(Decimal(row[4].strip()))
        open_p = float(Decimal(row[5].strip()))

        open_time = datetime.fromtimestamp(ts_val, tz=UTC)
        close_time = open_time + timeframe.duration
        if close_time > effective_received:
            continue

        base_vol = float(contracts_dec * mult)
        quote_vol = base_vol * close_p
        candle = MarketCandle(
            provider="gate_archive",
            venue=venue,
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=provider_symbol,
            market_type=MarketType.PERPETUAL,
            timeframe=timeframe,
            open_time=open_time,
            close_time=close_time,
            received_time=effective_received,
            available_time=close_time,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            raw_contracts=float(contracts_dec),
            base_volume=base_vol,
            quote_volume=quote_vol,
            base_unit=symbol.base,
            quote_currency=symbol.quote,
            is_closed=True,
            quality_flags=(QualityFlag.BACKFILL, QualityFlag.QUOTE_VOL_ESTIMATED),
        )
        by_open[open_time] = candle
    return tuple(by_open[ts] for ts in sorted(by_open))


def parse_gatedata_funding_applies_csv(
    lines: Iterable[str],
    *,
    symbol: SymbolMapping,
    contract_id: str,
    received_time: datetime | None = None,
    venue: str = "gate",
    funding_interval_hours: float = 8.0,
) -> tuple[FundingRateObservation, ...]:
    """Parse `funding_applies` CSV (`[0]=ts_s, [1]=funding_rate`) into `FundingRateObservation`."""
    del contract_id
    effective_received = (received_time or datetime.now(UTC)).astimezone(UTC)
    provider_symbol = f"GATE:{symbol.base}{symbol.quote}.P"
    scale_8h = 8.0 / funding_interval_hours

    by_time: dict[datetime, FundingRateObservation] = {}
    for row in csv.reader(lines):
        if not row or len(row) < 2:
            continue
        ts_val = int(float(row[0].strip()))
        rate = float(row[1].strip())
        obs_time = datetime.fromtimestamp(ts_val, tz=UTC)
        if obs_time > effective_received:
            continue
        rate_8h = rate * scale_8h
        by_time[obs_time] = FundingRateObservation(
            provider="gate_archive",
            venue=venue,
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=provider_symbol,
            market_type=MarketType.PERPETUAL,
            observed_time=obs_time,
            next_funding_time=None,
            received_time=effective_received,
            available_time=obs_time,
            funding_rate=rate,
            funding_interval_hours=funding_interval_hours,
            rate_per_8h=rate_8h,
            annualized_rate=rate_8h * 3.0 * 365.0,
            quality_flags=(QualityFlag.BACKFILL,),
            raw_payload={"timestamp": ts_val, "funding_rate": rate, "source": "gatedata"},
        )
    return tuple(by_time[ts] for ts in sorted(by_time))


def iter_gatedata_trades_csv(lines: Iterable[str]) -> Iterator[GateArchiveTrade]:
    """Stream `trades` CSV (`[0]=create_time_s, [1]=trade_id, [2]=price, [3]=signed_contracts`)."""
    for row in csv.reader(lines):
        if not row or len(row) < 4:
            continue
        ts = datetime.fromtimestamp(float(row[0].strip()), tz=UTC)
        yield GateArchiveTrade(
            timestamp=ts,
            trade_id=int(row[1].strip()),
            price=float(row[2].strip()),
            signed_contracts=float(row[3].strip()),
        )


@dataclass(frozen=True, slots=True)
class ContiguityValidationResult:
    """Validation result for time-series candle contiguity and backfill invariants."""

    is_contiguous: bool
    candle_count: int
    gaps: tuple[tuple[datetime, datetime], ...]
    available_eq_close_ok: bool
    backfill_flagged_ok: bool


def validate_candle_contiguity(
    candles: Sequence[MarketCandle],
    *,
    timeframe: Timeframe,
) -> ContiguityValidationResult:
    """Verify monotonic contiguity, `available_time == close_time`, and BACKFILL flag."""
    if not candles:
        return ContiguityValidationResult(
            is_contiguous=True,
            candle_count=0,
            gaps=(),
            available_eq_close_ok=True,
            backfill_flagged_ok=True,
        )

    step = timeframe.duration
    gaps: list[tuple[datetime, datetime]] = []
    available_ok = True
    backfill_ok = True

    for c in candles:
        if c.available_time != c.close_time:
            available_ok = False
        if QualityFlag.BACKFILL not in c.quality_flags:
            backfill_ok = False

    for prev, curr in itertools.pairwise(candles):
        if curr.open_time - prev.open_time != step:
            gaps.append((prev.open_time + step, curr.open_time))

    return ContiguityValidationResult(
        is_contiguous=len(gaps) == 0,
        candle_count=len(candles),
        gaps=tuple(gaps),
        available_eq_close_ok=available_ok,
        backfill_flagged_ok=backfill_ok,
    )


@dataclass(frozen=True, slots=True)
class FileIngestResult:
    """Per-file outcome of bulk archive download + PostgreSQL loading."""

    dataset: str
    contract: str
    yyyymm: str
    local_path: str
    sha256: str
    status: str  # "loaded" | "skipped_checksum" | "not_found" | "archived_raw"
    row_count: int
    inserted_rows: int
    contiguity_ok: bool | None = None


@dataclass(frozen=True, slots=True)
class BulkIngestSummary:
    """Aggregate summary for one `ingest-gate-history` execution."""

    files_processed: int
    downloaded_files: int
    skipped_files: int
    inserted_rows: int
    missing_files: int
    results: tuple[FileIngestResult, ...]


def _read_checksum_sidecar(path: Path) -> str | None:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        return None
    text = sidecar.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text.split()[0]


def _write_checksum_sidecar(path: Path, sha256: str) -> None:
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(f"{sha256}  {path.name}\n", encoding="utf-8")


class GateHistoryIngestor:
    """Download, checksum-verify, archive, and load `download.gatedata.org` histories."""

    def __init__(
        self,
        *,
        data_dir: Path = DEFAULT_GATE_RAW_DIR,
        base_url: str = DEFAULT_GATEDATA_BASE_URL,
        market_repository: MarketRepository | None = None,
        derivatives_repository: DerivativesRepository | None = None,
        ledger_repository: GateIngestRepository | None = None,
        multipliers: dict[str, Decimal] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._data_dir = data_dir
        self._base_url = base_url.rstrip("/")
        self._market_repo = market_repository
        self._deriv_repo = derivatives_repository
        self._ledger_repo = ledger_repository
        self._multipliers = dict(multipliers or {})
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _archive_path(self, dataset: str, contract: str, yyyymm: str) -> Path:
        return self._data_dir / dataset / yyyymm / f"{contract}-{yyyymm}.csv.gz"

    def _remote_url(self, dataset: str, contract: str, yyyymm: str) -> str:
        return f"{self._base_url}/{dataset}/{yyyymm}/{contract}-{yyyymm}.csv.gz"

    async def _ensure_file(
        self,
        dataset: str,
        contract: str,
        yyyymm: str,
    ) -> tuple[Path | None, str, bool]:
        """Ensure archive exists under `data/raw/gate/`. Returns `(path, sha256, downloaded)`."""
        target = self._archive_path(dataset, contract, yyyymm)
        if target.is_file():
            sidecar_sha = _read_checksum_sidecar(target)
            actual_sha = compute_file_sha256(target)
            if sidecar_sha is not None and sidecar_sha == actual_sha:
                return target, actual_sha, False
            if sidecar_sha is None:
                # Re-verify gzip stream cleanly before adopting existing local archive
                with gzip.open(target, "rt", encoding="utf-8") as gz:
                    gz.read(1024)
                _write_checksum_sidecar(target, actual_sha)
                return target, actual_sha, False

        url = self._remote_url(dataset, contract, yyyymm)
        response = await self._client.get(url)
        if response.status_code == 404:
            return None, "", False
        response.raise_for_status()

        raw_bytes = response.content
        # Validate gzip integrity before atomic rename so downloads remain resumable.
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as gz:
            gz.read(64)

        target.parent.mkdir(parents=True, exist_ok=True)
        part_path = Path(f"{target}.part")
        part_path.write_bytes(raw_bytes)
        sha256 = compute_file_sha256(part_path)
        part_path.replace(target)
        _write_checksum_sidecar(target, sha256)
        return target, sha256, True

    async def ingest_range(
        self,
        *,
        symbols: Sequence[str],
        from_ym: str,
        to_ym: str,
        datasets: Sequence[str],
        validate_contiguity: bool = True,
        as_of: datetime | None = None,
    ) -> BulkIngestSummary:
        months = iter_months(from_ym, to_ym)
        validated_datasets = [validate_dataset_type(d) for d in datasets]
        now = (as_of or datetime.now(UTC)).astimezone(UTC)

        results: list[FileIngestResult] = []
        downloaded = 0
        skipped = 0
        inserted_total = 0
        missing = 0

        for raw_sym in symbols:
            mapping, contract = to_gate_contract_symbol(raw_sym)
            for dataset in validated_datasets:
                for yyyymm in months:
                    path, sha256, was_downloaded = await self._ensure_file(
                        dataset, contract, yyyymm
                    )
                    if path is None:
                        missing += 1
                        results.append(
                            FileIngestResult(
                                dataset=dataset,
                                contract=contract,
                                yyyymm=yyyymm,
                                local_path=str(self._archive_path(dataset, contract, yyyymm)),
                                sha256="",
                                status="not_found",
                                row_count=0,
                                inserted_rows=0,
                            )
                        )
                        continue

                    if was_downloaded:
                        downloaded += 1

                    # Check ledger idempotency before re-parsing/inserting DB rows
                    if self._ledger_repo is not None:
                        recorded_sha = self._ledger_repo.get_ingested_sha256(
                            dataset=dataset,
                            contract=contract,
                            yyyymm=yyyymm,
                        )
                        if recorded_sha is not None and recorded_sha == sha256:
                            skipped += 1
                            results.append(
                                FileIngestResult(
                                    dataset=dataset,
                                    contract=contract,
                                    yyyymm=yyyymm,
                                    local_path=str(path),
                                    sha256=sha256,
                                    status="skipped_checksum",
                                    row_count=0,
                                    inserted_rows=0,
                                    contiguity_ok=True,
                                )
                            )
                            continue

                    res = self._load_file_into_db(
                        path=path,
                        sha256=sha256,
                        dataset=dataset,
                        contract=contract,
                        yyyymm=yyyymm,
                        symbol=mapping,
                        as_of=now,
                        validate_contiguity=validate_contiguity,
                    )
                    inserted_total += res.inserted_rows
                    if self._ledger_repo is not None:
                        self._ledger_repo.record_ingested_file(
                            dataset=dataset,
                            contract=contract,
                            yyyymm=yyyymm,
                            sha256=sha256,
                            byte_size=path.stat().st_size,
                            local_path=str(path),
                            row_count=res.row_count,
                            inserted_rows=res.inserted_rows,
                            ingested_at=now,
                        )
                    results.append(res)

        return BulkIngestSummary(
            files_processed=len(results),
            downloaded_files=downloaded,
            skipped_files=skipped,
            inserted_rows=inserted_total,
            missing_files=missing,
            results=tuple(results),
        )

    def _load_file_into_db(
        self,
        *,
        path: Path,
        sha256: str,
        dataset: str,
        contract: str,
        yyyymm: str,
        symbol: SymbolMapping,
        as_of: datetime,
        validate_contiguity: bool,
    ) -> FileIngestResult:
        if dataset in VERIFIED_GATEDATA_CANDLESTICK_TYPES:
            tf = VERIFIED_GATEDATA_CANDLESTICK_TYPES[dataset]
            if tf is None:
                line_count = _count_gzip_lines(path)
                return FileIngestResult(
                    dataset=dataset,
                    contract=contract,
                    yyyymm=yyyymm,
                    local_path=str(path),
                    sha256=sha256,
                    status="archived_raw",
                    row_count=line_count,
                    inserted_rows=0,
                )
            mult = self._multipliers.get(contract)
            if mult is None:
                raise ValueError(
                    f"quanto_multiplier required to normalize {contract} {dataset}; "
                    "pass multipliers map or fetch /futures/usdt/contracts"
                )
            with gzip.open(path, "rt", encoding="utf-8") as gz:
                candles = parse_gatedata_candlesticks_csv(
                    gz,
                    symbol=symbol,
                    contract_id=contract,
                    timeframe=tf,
                    quanto_multiplier=mult,
                    received_time=as_of,
                )
            contig_ok = True
            if validate_contiguity:
                contig = validate_candle_contiguity(candles, timeframe=tf)
                contig_ok = (
                    contig.is_contiguous
                    and contig.available_eq_close_ok
                    and contig.backfill_flagged_ok
                )
            inserted = self._market_repo.save_candles(candles) if self._market_repo else 0
            return FileIngestResult(
                dataset=dataset,
                contract=contract,
                yyyymm=yyyymm,
                local_path=str(path),
                sha256=sha256,
                status="loaded",
                row_count=len(candles),
                inserted_rows=inserted,
                contiguity_ok=contig_ok,
            )

        if dataset == "funding_applies":
            with gzip.open(path, "rt", encoding="utf-8") as gz:
                obs = parse_gatedata_funding_applies_csv(
                    gz,
                    symbol=symbol,
                    contract_id=contract,
                    received_time=as_of,
                )
            inserted = self._deriv_repo.save_funding_observations(obs) if self._deriv_repo else 0
            return FileIngestResult(
                dataset=dataset,
                contract=contract,
                yyyymm=yyyymm,
                local_path=str(path),
                sha256=sha256,
                status="loaded",
                row_count=len(obs),
                inserted_rows=inserted,
            )

        # `trades` and `mark_prices` archived cleanly under data/raw/gate/ for D-05 / research.
        line_count = _count_gzip_lines(path)
        return FileIngestResult(
            dataset=dataset,
            contract=contract,
            yyyymm=yyyymm,
            local_path=str(path),
            sha256=sha256,
            status="archived_raw",
            row_count=line_count,
            inserted_rows=0,
        )


def _count_gzip_lines(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as gz:
        return sum(1 for line in gz if line.strip())
