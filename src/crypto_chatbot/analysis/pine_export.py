"""Deterministic TradingView Pine Script v6 generator from backend AnalysisSnapshot."""

from __future__ import annotations

from decimal import Decimal

from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    SetupBias,
    SetupStatus,
)


def _format_pine_price(value: float) -> str:
    """Format a positive float as a clean decimal literal with zero scientific notation."""
    abs_v = abs(value)
    if abs_v >= 100.0:
        precision = 2
    elif abs_v >= 1.0:
        precision = 4
    elif abs_v >= 0.01:
        precision = 6
    elif abs_v >= 0.0001:
        precision = 8
    else:
        precision = 10

    dec = Decimal(f"{value:.{precision}f}")
    text = format(dec, "f").rstrip("0")
    if text.endswith("."):
        text += "00"
    elif "." not in text:
        text += ".00"
    else:
        _, frac = text.split(".")
        if len(frac) == 1 and abs_v >= 1.0:
            text += "0"
    return text


def generate_pine_script(snapshot: AnalysisSnapshot) -> str:
    """Deterministically compile a valid Pine Script v6 overlay indicator from a snapshot."""
    symbol = snapshot.canonical_symbol
    tf = snapshot.timeframe.value
    as_of_str = snapshot.as_of.strftime("%Y-%m-%d %H:%M UTC")
    setup = snapshot.trade_setup
    setup_status = setup.status.value.upper() if setup is not None else "NONE"
    setup_bias = setup.bias.value.upper() if setup is not None else "NEUTRAL"

    lines: list[str] = [
        "//@version=6",
        (
            f'indicator("Crypto Research: {symbol} ({tf})", '
            "overlay=true, max_lines_count=50, max_boxes_count=20, max_labels_count=50)"
        ),
        f"// Source   : {snapshot.provider}:{snapshot.venue} ({snapshot.provider_symbol})",
        f"// Snapshot : {snapshot.snapshot_id}",
        f"// As Of    : {as_of_str}",
        f"// Quality  : {snapshot.quality_status.value.upper()}",
        f"// Setup    : {setup_status} | Bias: {setup_bias}",
        "",
        "f_level(price, txt, col, line_style, width_val) =>",
        (
            "    line.new(bar_index - 30, price, bar_index + 10, price, "
            "color=col, style=line_style, width=width_val, extend=extend.right)"
        ),
        (
            "    label.new(bar_index + 2, price, txt, style=label.style_label_left, "
            "color=color.new(col, 85), textcolor=col, size=size.small)"
        ),
        "",
        "f_zone(top_p, bot_p, txt, border_col, fill_col) =>",
        (
            "    box.new(bar_index - 30, top_p, bar_index + 12, bot_p, "
            "border_color=border_col, bgcolor=fill_col, extend=extend.right)"
        ),
        (
            "    label.new(bar_index + 2, (top_p + bot_p) / 2.0, txt, "
            "style=label.style_label_left, color=color.new(border_col, 85), "
            "textcolor=border_col, size=size.small)"
        ),
        "",
        "if barstate.islast",
    ]

    plot_lines: list[str] = []

    if setup is not None:
        if setup.status is SetupStatus.READY and (
            setup.entry_low is not None
            and setup.entry_high is not None
            and setup.stop_loss is not None
            and setup.tp1 is not None
            and setup.tp2 is not None
            and setup.tp3 is not None
        ):
            is_long = setup.bias is SetupBias.LONG
            entry_col = "color.teal" if is_long else "color.orange"
            entry_fill = "color.new(color.teal, 85)" if is_long else "color.new(color.orange, 85)"
            e_hi = _format_pine_price(setup.entry_high)
            e_lo = _format_pine_price(setup.entry_low)
            plot_lines.append(
                f'    f_zone({e_hi}, {e_lo}, "ENTRY {setup_bias} ({e_lo} - {e_hi})", '
                f"{entry_col}, {entry_fill})"
            )

            sl_p = _format_pine_price(setup.stop_loss)
            plot_lines.append(f'    f_level({sl_p}, "SL ({sl_p})", color.red, line.style_solid, 2)')

            rr1 = f"1:{setup.rr_tp1:.2f}" if setup.rr_tp1 is not None else "1:1"
            rr2 = f"1:{setup.rr_tp2:.2f}" if setup.rr_tp2 is not None else "1:2"
            rr3 = f"1:{setup.rr_tp3:.2f}" if setup.rr_tp3 is not None else "1:3"
            tp1_p = _format_pine_price(setup.tp1)
            tp2_p = _format_pine_price(setup.tp2)
            tp3_p = _format_pine_price(setup.tp3)
            plot_lines.append(
                f'    f_level({tp1_p}, "TP1 ({tp1_p} | {rr1})", color.lime, line.style_solid, 1)'
            )
            plot_lines.append(
                f'    f_level({tp2_p}, "TP2 ({tp2_p} | {rr2})", color.green, line.style_solid, 2)'
            )
            plot_lines.append(
                f'    f_level({tp3_p}, "TP3 ({tp3_p} | {rr3})", color.aqua, line.style_solid, 2)'
            )

        if setup.htf_support is not None:
            p = _format_pine_price(setup.htf_support)
            plot_lines.append(
                f'    f_level({p}, "HTF Sup ({p})", color.blue, line.style_dashed, 1)'
            )
        if setup.htf_resistance is not None:
            p = _format_pine_price(setup.htf_resistance)
            plot_lines.append(
                f'    f_level({p}, "HTF Res ({p})", color.fuchsia, line.style_dashed, 1)'
            )
        if setup.ltf_support is not None:
            p = _format_pine_price(setup.ltf_support)
            plot_lines.append(
                f'    f_level({p}, "LTF Sup 15m ({p})", color.aqua, line.style_dotted, 1)'
            )
        if setup.ltf_resistance is not None:
            p = _format_pine_price(setup.ltf_resistance)
            plot_lines.append(
                f'    f_level({p}, "LTF Res 15m ({p})", color.purple, line.style_dotted, 1)'
            )

    smc = snapshot.smc
    if smc is not None:
        if smc.dealing_range_high is not None:
            p = _format_pine_price(smc.dealing_range_high)
            plot_lines.append(
                f'    f_level({p}, "Range High ({p})", color.gray, line.style_dashed, 1)'
            )
        if smc.equilibrium is not None:
            p = _format_pine_price(smc.equilibrium)
            plot_lines.append(
                f'    f_level({p}, "Equilibrium ({p})", color.silver, line.style_dotted, 1)'
            )
        if smc.dealing_range_low is not None:
            p = _format_pine_price(smc.dealing_range_low)
            plot_lines.append(
                f'    f_level({p}, "Range Low ({p})", color.gray, line.style_dashed, 1)'
            )

        poi_specs = (
            (smc.active_bullish_fvg, "Bull FVG", "color.green", "color.new(color.green, 88)"),
            (smc.active_bearish_fvg, "Bear FVG", "color.red", "color.new(color.red, 88)"),
            (smc.active_bullish_ob, "Bull OB", "color.teal", "color.new(color.teal, 88)"),
            (smc.active_bearish_ob, "Bear OB", "color.maroon", "color.new(color.maroon, 88)"),
        )
        for zone, label, border_col, fill_col in poi_specs:
            if zone is not None:
                hi = _format_pine_price(zone.zone_high)
                lo = _format_pine_price(zone.zone_low)
                plot_lines.append(
                    f'    f_zone({hi}, {lo}, "{label} ({lo}-{hi})", {border_col}, {fill_col})'
                )

    if not plot_lines and snapshot.technical is not None:
        last_p = _format_pine_price(snapshot.technical.last_close)
        plot_lines.append(
            f'    f_level({last_p}, "Last Close ({last_p})", color.gray, line.style_dotted, 1)'
        )
    elif not plot_lines:
        plot_lines.append('    label.new(bar_index, high, "No Active Levels", size=size.small)')

    lines.extend(plot_lines)
    return "\n".join(lines) + "\n"
