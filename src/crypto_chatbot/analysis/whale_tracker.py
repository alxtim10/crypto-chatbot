"""Deterministic whale tracking analysis, position aggregation, and Indonesian report formatters."""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol

from crypto_chatbot.market_data.whale_provider import (
    CURATED_HYPERLIQUID_WHALES,
    detect_network_type,
)
from crypto_chatbot.schemas.whale import (
    NetworkType,
    WhaleFlowType,
    WhaleMarketOverview,
    WhalePosition,
    WhalePositionSide,
    WhaleWalletProfile,
    WhaleWatchlistEntry,
)

MAX_WHALE_WATCHLIST_PER_USER = 20

HEDGING_DISCLAIMER = (
    "Catatan Manajemen Risiko: Posisi besar whale di pasar futures dapat berfungsi sebagai "
    "delta-neutral basis trade, hedging inventaris spot, atau penyedia likuiditas (LP), "
    "bukan semata-mata taruhan arah harga (directional bet)."
)


class WhaleWatchlistStore(Protocol):
    """Persistence contract for per-user whale wallet watchlists."""

    def get_user_entries(self, user_id: int) -> tuple[WhaleWatchlistEntry, ...] | None:
        """Return saved entries for user_id, or None if the user has never customized defaults."""

    def replace_user_entries(
        self,
        user_id: int,
        entries: Sequence[WhaleWatchlistEntry],
    ) -> None:
        """Persist the full ordered watchlist snapshot for user_id."""


def _b(text: str, *, html_mode: bool) -> str:
    safe = html.escape(text, quote=False) if html_mode else text
    return f"<b>{safe}</b>" if html_mode else safe


def _c(text: str, *, html_mode: bool) -> str:
    safe = html.escape(text, quote=False) if html_mode else text
    return f"<code>{safe}</code>" if html_mode else safe


def _i(text: str, *, html_mode: bool) -> str:
    safe = html.escape(text, quote=False) if html_mode else text
    return f"<i>{safe}</i>" if html_mode else safe


def _t(text: str, *, html_mode: bool) -> str:
    return html.escape(text, quote=False) if html_mode else text


def aggregate_whale_positions(
    symbol: str,
    positions: Sequence[WhalePosition],
    *,
    as_of: datetime | None = None,
) -> WhaleMarketOverview:
    """Deterministically aggregate long/short whale positions and calculate market bias."""
    clean_sym = symbol.strip().upper()
    now = as_of or datetime.now(UTC)

    sym_positions = [p for p in positions if p.symbol.upper() == clean_sym]
    total_long = sum(p.size_usd for p in sym_positions if p.side == WhalePositionSide.LONG)
    total_short = sum(p.size_usd for p in sym_positions if p.side == WhalePositionSide.SHORT)
    total_volume = total_long + total_short

    if total_volume > 0.0:
        long_pct = round((total_long / total_volume) * 100.0, 1)
        short_pct = round(100.0 - long_pct, 1)
    else:
        long_pct = 50.0
        short_pct = 50.0

    net_bias: Literal["HEAVY_LONG", "LEAN_LONG", "NEUTRAL", "LEAN_SHORT", "HEAVY_SHORT"]
    if long_pct >= 70.0:
        net_bias = "HEAVY_LONG"
    elif long_pct >= 55.0:
        net_bias = "LEAN_LONG"
    elif short_pct >= 70.0:
        net_bias = "HEAVY_SHORT"
    elif short_pct >= 55.0:
        net_bias = "LEAN_SHORT"
    else:
        net_bias = "NEUTRAL"

    unique_wallets = {p.wallet_address.lower() for p in sym_positions}
    sorted_positions = tuple(sorted(sym_positions, key=lambda p: p.size_usd, reverse=True)[:5])

    notes = (
        (
            f"Data dihimpun dari {len(unique_wallets)} dompet whale/institusi aktif "
            "di perpetual futures."
        ),
        HEDGING_DISCLAIMER,
    )

    return WhaleMarketOverview(
        symbol=clean_sym,
        as_of=now,
        total_long_usd=total_long,
        total_short_usd=total_short,
        long_percentage=long_pct,
        short_percentage=short_pct,
        net_bias=net_bias,
        active_whale_count=len(unique_wallets),
        top_positions=sorted_positions,
        notes=notes,
    )


def format_currency_usd(amount: float) -> str:
    """Format USD values with appropriate K/M/B suffixes, preserving sign cleanly."""
    sign = "-" if amount < 0 else ""
    mag = abs(amount)
    if mag >= 1_000_000_000:
        return f"{sign}${mag / 1_000_000_000:.2f}B"
    if mag >= 1_000_000:
        return f"{sign}${mag / 1_000_000:.2f}M"
    if mag >= 1_000:
        return f"{sign}${mag / 1_000:.1f}K"
    return f"{sign}${mag:.2f}"


def format_whale_overview_telegram(
    overview: WhaleMarketOverview,
    *,
    html_mode: bool = True,
) -> str:
    """Render a deterministic Telegram or CLI message summarizing whale perpetual bias."""
    bias_labels = {
        "HEAVY_LONG": "BULLISH KUAT (Heavy Long 🟢)",
        "LEAN_LONG": "CENDERUNG BULLISH (Lean Long 🟢)",
        "NEUTRAL": "NETRAL / SEIMBANG (Balanced ⚪)",
        "LEAN_SHORT": "CENDERUNG BEARISH (Lean Short 🔴)",
        "HEAVY_SHORT": "BEARISH KUAT (Heavy Short 🔴)",
    }
    bias_desc = bias_labels.get(overview.net_bias, overview.net_bias)

    lines = [
        f"🐋 {_b(f'WHALE PERP TRACKER — {overview.symbol}', html_mode=html_mode)}",
        f"⏰ Update: {_c(overview.as_of.strftime('%Y-%m-%d %H:%M UTC'), html_mode=html_mode)}",
        "",
        f"⚖️ {_b('Bias Posisi Paus:', html_mode=html_mode)} {_b(bias_desc, html_mode=html_mode)}",
        (
            "• Total Long: "
            f"{_c(format_currency_usd(overview.total_long_usd), html_mode=html_mode)} "
            f"({overview.long_percentage}%)"
        ),
        (
            f"• Total Short: "
            f"{_c(format_currency_usd(overview.total_short_usd), html_mode=html_mode)} "
            f"({overview.short_percentage}%)"
        ),
        (f"• Jumlah Dompet Terlacak: {_c(str(overview.active_whale_count), html_mode=html_mode)}"),
    ]

    if overview.top_positions:
        lines.append("")
        lines.append(f"🏆 {_b('Top Posisi Whale Terbuka:', html_mode=html_mode)}")
        for i, pos in enumerate(overview.top_positions, 1):
            side_badge = "🟢 LONG" if pos.side == WhalePositionSide.LONG else "🔴 SHORT"
            pnl_prefix = "+" if pos.unrealized_pnl_usd > 0 else ""
            pnl_str = f"{pnl_prefix}{format_currency_usd(pos.unrealized_pnl_usd)}"
            lev_str = f"{int(pos.leverage)}x" if pos.leverage else "Cross"
            addr_preview = f"{pos.wallet_address[:6]}...{pos.wallet_address[-4:]}"
            short_addr = _c(addr_preview, html_mode=html_mode)
            label_text = (
                f" ({_t(pos.wallet_label, html_mode=html_mode)})" if pos.wallet_label else ""
            )
            lines.append(
                f"{i}. {side_badge} {short_addr}{label_text}\n"
                f"   Size: {_c(format_currency_usd(pos.size_usd), html_mode=html_mode)} "
                f"({_t(lev_str, html_mode=html_mode)}) | "
                f"Entry: {_c(f'${pos.entry_price:,.2f}', html_mode=html_mode)}\n"
                f"   uPnL: {_c(pnl_str, html_mode=html_mode)}"
            )
    else:
        lines.append("")
        lines.append(
            "[Info] "
            + _i(
                "Belum ada posisi terbuka signifikan dari whale terpantau pada pair ini.",
                html_mode=html_mode,
            )
        )

    lines.append("")
    lines.append(f"💡 {_i(HEDGING_DISCLAIMER, html_mode=html_mode)}")
    return "\n".join(lines)


def format_whale_wallet_telegram(
    profile: WhaleWalletProfile,
    *,
    html_mode: bool = True,
) -> str:
    """Render a deterministic Telegram or CLI message for a tracked wallet profile and flows."""
    addr_preview = (
        f"{profile.address[:8]}...{profile.address[-6:]}"
        if len(profile.address) > 16
        else profile.address
    )
    addr_short = _c(addr_preview, html_mode=html_mode)
    label_str = f" {_b(f'({profile.label})', html_mode=html_mode)}" if profile.label else ""

    lines = [
        f"🔍 {_b('WHALE WALLET PROFILE', html_mode=html_mode)}",
        f"📍 Address: {addr_short}{label_str}",
        f"🌐 Network: {_c(profile.network.value.upper(), html_mode=html_mode)}",
    ]

    if profile.network == NetworkType.BITCOIN:
        lines.append(
            f"💰 Saldo On-Chain: {_c(f'{profile.balance_native:.4f} BTC', html_mode=html_mode)}"
        )
    elif profile.network == NetworkType.SOLANA:
        lines.append(
            "💰 Saldo On-Chain: "
            + _i(
                "Tidak tersedia (RPC on-chain Solana belum dikonfigurasi).",
                html_mode=html_mode,
            )
        )
    elif profile.balance_native > 0:
        lines.append(f"💰 Saldo: {_c(f'{profile.balance_native:.4f}', html_mode=html_mode)}")
    else:
        lines.append(
            "💰 Saldo On-Chain EVM: "
            + _i(
                "Tidak dimuat (pelacakan fokus pada posisi perpetual Hyperliquid).",
                html_mode=html_mode,
            )
        )

    if profile.positions:
        lines.append("")
        lines.append(
            "📊 " + _b(f"Posisi Futures Terbuka ({len(profile.positions)}):", html_mode=html_mode)
        )
        for pos in profile.positions:
            side_badge = "🟢 LONG" if pos.side == WhalePositionSide.LONG else "🔴 SHORT"
            pnl_prefix = "+" if pos.unrealized_pnl_usd > 0 else ""
            pnl_str = f"{pnl_prefix}{format_currency_usd(pos.unrealized_pnl_usd)}"
            lines.append(
                f"• {_b(pos.symbol, html_mode=html_mode)} {side_badge} | "
                f"Size: {_c(format_currency_usd(pos.size_usd), html_mode=html_mode)}\n"
                f"  Entry: {_c(f'${pos.entry_price:,.2f}', html_mode=html_mode)} | "
                f"uPnL: {_c(pnl_str, html_mode=html_mode)}"
            )

    if profile.recent_flows:
        lines.append("")
        lines.append(f"🌊 {_b('Aktivitas Aliran Transaksi Terbaru:', html_mode=html_mode)}")
        for flow in profile.recent_flows[:5]:
            if flow.flow_type == WhaleFlowType.CEX_INFLOW:
                badge = f"🚨 {_b('DEPOSIT KE CEX', html_mode=html_mode)} (Potensi Jual)"
            elif flow.flow_type == WhaleFlowType.CEX_OUTFLOW:
                badge = f"📥 {_b('WITHDRAW DARI CEX', html_mode=html_mode)} (Akumulasi)"
            else:
                badge = f"🔄 {_b('TRANSFER ON-CHAIN', html_mode=html_mode)}"

            cp_label = (
                f" -> {_t(flow.counterpart_label, html_mode=html_mode)}"
                if flow.counterpart_label
                else ""
            )
            qty_str = _c(f"{flow.amount:.4f} {flow.unit}", html_mode=html_mode)
            lines.append(
                f"• {badge}\n"
                f"  Jumlah: {qty_str}{cp_label}\n"
                f"  Tx: {_c(flow.tx_hash, html_mode=html_mode)} "
                f"({flow.timestamp.strftime('%d-%m %H:%M UTC')})"
            )

    if not profile.positions and not profile.recent_flows:
        lines.append("")
        lines.append(
            "[Info] "
            + _i(
                "Tidak ada posisi futures aktif atau aliran transaksi signifikan.",
                html_mode=html_mode,
            )
        )

    return "\n".join(lines)


def format_whale_watchlist_telegram(
    entries: Sequence[WhaleWatchlistEntry],
    *,
    html_mode: bool = True,
) -> str:
    """Render list of tracked whale addresses safely for Telegram HTML or CLI plain text."""
    add_cmd = _c("/whale add <address> [label]", html_mode=html_mode)
    if not entries:
        return (
            f"📋 {_b('DAFTAR WATCHLIST WHALE', html_mode=html_mode)}\n\n"
            "Belum ada dompet tersimpan di watchlist.\n"
            f"Gunakan {add_cmd} untuk mulai memantau dompet paus."
        )

    lines = [
        f"📋 {_b(f'DAFTAR WATCHLIST WHALE ({len(entries)})', html_mode=html_mode)}",
        "",
    ]
    for i, e in enumerate(entries, 1):
        preview = f"{e.address[:8]}...{e.address[-6:]}" if len(e.address) > 16 else e.address
        short = _c(preview, html_mode=html_mode)
        lines.append(
            f"{i}. {_b(e.label, html_mode=html_mode)} "
            f"({_t(e.network.value.upper(), html_mode=html_mode)})\n"
            f"   {short}"
        )

    lines.append("")
    lines.append(
        f"Gunakan {_c('/whale <address>', html_mode=html_mode)} untuk memeriksa detail atau saldo."
    )
    return "\n".join(lines)


class WhaleWatchlistManager:
    """Whale watchlist manager backed by optional persistent store or in-memory defaults."""

    def __init__(
        self,
        clock: Callable[[], datetime] | None = None,
        store: WhaleWatchlistStore | None = None,
    ) -> None:
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._store = store
        # Map user_id (or None for global default) to dict[address_lower, WhaleWatchlistEntry]
        self._user_entries: dict[int | None, dict[str, WhaleWatchlistEntry]] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        defaults: dict[str, WhaleWatchlistEntry] = {}
        for addr, label in CURATED_HYPERLIQUID_WHALES:
            net = detect_network_type(addr) or NetworkType.EVM
            defaults[addr.lower()] = WhaleWatchlistEntry(
                address=addr.lower(),
                network=net,
                label=label,
                added_at=self._clock(),
            )
        self._user_entries[None] = defaults

    def _load_mutable_user_dict(self, user_id: int | None) -> dict[str, WhaleWatchlistEntry]:
        if user_id is None:
            return self._user_entries[None]
        if self._store is not None:
            persisted = self._store.get_user_entries(user_id)
            if persisted is not None:
                return {e.address.lower(): e for e in persisted}
        cached = self._user_entries.get(user_id)
        if cached is not None:
            return dict(cached)
        return dict(self._user_entries[None])

    def list_entries(self, user_id: int | None = None) -> tuple[WhaleWatchlistEntry, ...]:
        """List all entries for a user, falling back to curated defaults only when unset."""
        if user_id is not None and self._store is not None:
            persisted = self._store.get_user_entries(user_id)
            if persisted is not None:
                return persisted
            return tuple(self._user_entries[None].values())

        user_dict = self._user_entries.get(user_id)
        if user_dict is not None:
            return tuple(user_dict.values())
        return tuple(self._user_entries[None].values())

    def add_entry(
        self,
        address: str,
        label: str,
        user_id: int | None = None,
    ) -> WhaleWatchlistEntry:
        """Add a new wallet address to user's watchlist."""
        clean_addr = address.strip()
        net = detect_network_type(clean_addr)
        if net is None:
            raise ValueError(f"Format alamat wallet tidak valid: {address}")

        user_dict = self._load_mutable_user_dict(user_id)
        normalized_addr = clean_addr.lower() if net == NetworkType.EVM else clean_addr
        key = normalized_addr.lower()
        if key not in user_dict and len(user_dict) >= MAX_WHALE_WATCHLIST_PER_USER:
            raise ValueError(
                f"Watchlist whale maksimal {MAX_WHALE_WATCHLIST_PER_USER} alamat per pengguna."
            )

        entry = WhaleWatchlistEntry(
            address=normalized_addr,
            network=net,
            label=label.strip() or f"Whale-{clean_addr[:6]}",
            added_at=self._clock(),
        )
        user_dict[key] = entry
        if user_id is not None:
            self._user_entries[user_id] = user_dict
            if self._store is not None:
                self._store.replace_user_entries(user_id, tuple(user_dict.values()))
        return entry

    def remove_entry(self, address: str, user_id: int | None = None) -> bool:
        """Remove a wallet address from user's watchlist."""
        clean_key = address.strip().lower()
        user_dict = self._load_mutable_user_dict(user_id)
        if clean_key not in user_dict:
            return False

        del user_dict[clean_key]
        if user_id is not None:
            self._user_entries[user_id] = user_dict
            if self._store is not None:
                self._store.replace_user_entries(user_id, tuple(user_dict.values()))
        return True
