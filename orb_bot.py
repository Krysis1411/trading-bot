"""
Live ORB (Opening Range Breakout) day trading bot — Alpaca paper trading.

Design
------
Runs every 5 minutes via GitHub Actions (market hours only).
Stateless: all state is re-derived from today's Alpaca bars + open positions.

Strategy
--------
Opening range : first 6 × 5-min bars (9:30–10:00 ET)
Entry         : close breaks above OR high AND bar volume >= avg OR vol × factor
Stop          : OR low - buffer
Target        : OR high + (OR range × per-symbol multiplier)
EOD close     : force-close any position at 3:45 PM ET
One trade/day : skip entry if we already have a filled order for this symbol today

Improvements (v2)
------------------
1. SPY trend filter   : only enter when SPY latest bar close > SPY session open
2. Min OR range filter : skip if OR range / OR high < ORB_MIN_OR_PCT (indecisive open)
3. Per-symbol multiplier : ORB_PROFIT_MULTIPLIERS dict, fallback to ORB_PROFIT_MULTIPLIER
"""
import logging
import os
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi

from config import (
    ORB_CLOSE_HOUR,
    ORB_CLOSE_MINUTE,
    ORB_MIN_OR_PCT,
    ORB_POSITION_SIZE,
    ORB_PROFIT_MULTIPLIER,
    ORB_PROFIT_MULTIPLIERS,
    ORB_RANGE_BARS,
    ORB_STOP_BUFFER,
    ORB_VOLUME_FACTOR,
    MAX_TOTAL_INVESTMENT,
)
from screener import get_active_symbols

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

api = tradeapi.REST(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"],
    "https://paper-api.alpaca.markets",
    api_version="v2",
)

ET = ZoneInfo("America/New_York")
CLOSE_TIME = time(ORB_CLOSE_HOUR, ORB_CLOSE_MINUTE)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_now_et() -> datetime:
    return datetime.now(ET)


def get_today_bars(symbol: str) -> pd.DataFrame | None:
    """Fetch all 5-min bars for today's regular session from Alpaca."""
    now_et = get_now_et()
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    try:
        bars = api.get_bars(
            symbol,
            "5Min",
            start=market_open.isoformat(),
            limit=100,
        ).df
        return bars if not bars.empty else None
    except Exception as e:
        log.error(f"{symbol}: failed to fetch bars — {e}")
        return None


def compute_opening_range(bars: pd.DataFrame) -> tuple[float, float, float] | None:
    """
    Return (or_high, or_low, avg_volume) from the first ORB_RANGE_BARS bars.
    Returns None if not enough bars have formed yet.
    """
    if len(bars) < ORB_RANGE_BARS:
        return None
    or_bars = bars.iloc[:ORB_RANGE_BARS]
    return (
        float(or_bars["high"].max()),
        float(or_bars["low"].min()),
        float(or_bars["volume"].mean()),
    )


def already_traded_today(symbol: str) -> bool:
    """Return True if a buy order was already filled for this symbol today."""
    today_start = get_now_et().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        orders = api.list_orders(
            status="filled",
            after=today_start.isoformat(),
            direction="asc",
        )
        return any(o.symbol == symbol and o.side == "buy" for o in orders)
    except Exception as e:
        log.warning(f"{symbol}: could not check today's orders — {e}")
        return False


def get_position(symbol: str):
    try:
        return api.get_position(symbol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Improvement 1: SPY trend filter
# ---------------------------------------------------------------------------

def get_spy_trend() -> bool | None:
    """
    Return True if SPY is trending up today (latest close > session open).
    Returns None if SPY data isn't available yet (before 10:00 ET).
    """
    bars = get_today_bars("SPY")
    if bars is None or len(bars) < 1:
        return None
    spy_open = float(bars.iloc[0]["open"])
    spy_last = float(bars.iloc[-1]["close"])
    trending_up = spy_last >= spy_open
    log.info(f"SPY trend: open={spy_open:.2f}  last={spy_last:.2f}  {'UP' if trending_up else 'DOWN'}")
    return trending_up


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def process_symbol(symbol: str, spy_bullish: bool | None, open_positions_count: int, max_open_positions: int) -> bool:
    """Returns True if a new position was opened, False otherwise."""
    bars = get_today_bars(symbol)
    if bars is None:
        log.info(f"{symbol}: no bars yet")
        return False

    or_result = compute_opening_range(bars)
    if or_result is None:
        log.info(f"{symbol}: opening range not ready yet ({len(bars)}/{ORB_RANGE_BARS} bars)")
        return False

    or_high, or_low, avg_or_volume = or_result
    or_range = or_high - or_low

    # Improvement 2: skip narrow opening ranges
    if ORB_MIN_OR_PCT > 0 and or_range / or_high < ORB_MIN_OR_PCT:
        log.info(f"{symbol}: OR too narrow ({or_range / or_high:.3%} < {ORB_MIN_OR_PCT:.1%}) — skipping")
        return False

    # Improvement 3: per-symbol profit multiplier
    profit_multiplier = ORB_PROFIT_MULTIPLIERS.get(symbol, ORB_PROFIT_MULTIPLIER)
    trade_qty = max(1, int(ORB_POSITION_SIZE / or_high))

    stop_price = or_low - ORB_STOP_BUFFER
    target_price = or_high + or_range * profit_multiplier

    current_bar = bars.iloc[-1]
    current_price = float(current_bar["close"])
    current_volume = float(current_bar["volume"])
    now_et = get_now_et()

    position = get_position(symbol)
    qty = int(position.qty) if position else 0

    log.info(
        f"{symbol} | Price: {current_price:.2f} | OR: {or_low:.2f}–{or_high:.2f}"
        f" | Qty: {trade_qty} | Pos: {qty} | Mult: {profit_multiplier:.1f}x | Time: {now_et.strftime('%H:%M')} ET"
    )

    # --- EOD forced close ---
    if now_et.time() >= CLOSE_TIME:
        if position:
            api.submit_order(symbol=symbol, qty=qty, side="sell",
                             type="market", time_in_force="day")
            log.info(f"EOD CLOSE — SELL {qty} {symbol} at ~{current_price:.2f}")
        return False

    # --- Manage existing position ---
    if position:
        pnl_pct = float(position.unrealized_plpc) * 100

        if current_price <= stop_price:
            api.submit_order(symbol=symbol, qty=qty, side="sell",
                             type="market", time_in_force="day")
            log.info(f"STOP LOSS — SELL {qty} {symbol} at ~{current_price:.2f} | P&L: {pnl_pct:.2f}%")
            return False

        if current_price >= target_price:
            api.submit_order(symbol=symbol, qty=qty, side="sell",
                             type="market", time_in_force="day")
            log.info(f"TAKE PROFIT — SELL {qty} {symbol} at ~{current_price:.2f} | P&L: {pnl_pct:.2f}%")
            return False

        log.info(
            f"{symbol} | Holding {qty} shares | P&L: {pnl_pct:.2f}%"
            f" | Stop: {stop_price:.2f} | Target: {target_price:.2f}"
        )
        return False

    # --- Entry: breakout above OR high ---
    if already_traded_today(symbol):
        log.info(f"{symbol} | Already traded today — skipping entry")
        return False

    # Improvement 1: SPY trend filter
    if spy_bullish is False:
        log.info(f"{symbol} | SPY trending DOWN — skipping entry")
        return False
    if spy_bullish is None:
        log.info(f"{symbol} | SPY data not ready — skipping entry")
        return False

    vol_ok = current_volume >= avg_or_volume * ORB_VOLUME_FACTOR

    # Prevent entering trades where the price has already shot up to the target
    if current_price >= target_price:
        log.info(f"{symbol} | Price {current_price:.2f} already hit/exceeded target {target_price:.2f} — skipping entry")
        return False

    if current_price > or_high and vol_ok:
        if open_positions_count >= max_open_positions:
            log.info(f"{symbol} | Budget limit reached ({open_positions_count}/{max_open_positions} trades open) — skipping entry to stay under ${MAX_TOTAL_INVESTMENT}")
            return False
            
        api.submit_order(
            symbol=symbol,
            qty=trade_qty,
            side="buy",
            type="market",
            time_in_force="day",
        )
        log.info(
            f"BUY BREAKOUT {trade_qty} {symbol}"
            f" | Price: ~{current_price:.2f}"
            f" | Cost: ~${trade_qty * current_price:.0f}"
            f" | Stop: {stop_price:.2f}"
            f" | Target: {target_price:.2f}"
            f" | Vol ratio: {current_volume / avg_or_volume:.2f}x"
            f" | Mult: {profit_multiplier:.1f}x"
        )
        return True
    elif current_price > or_high and not vol_ok:
        log.info(f"{symbol} | Price above OR high but volume too low ({current_volume / avg_or_volume:.2f}x < {ORB_VOLUME_FACTOR}x) — skipping")
        return False
    else:
        log.info(f"{symbol} | No breakout (price {current_price:.2f} vs OR high {or_high:.2f})")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_orb() -> None:
    if not api.get_clock().is_open:
        log.info("Market closed — skipping run")
        return

    now_et = get_now_et()
    log.info(f"--- ORB check at {now_et.strftime('%H:%M')} ET ---")

    # Before 10:00 AM: opening range not ready for any symbol
    if now_et.time() < time(10, 0):
        log.info("Before 10:00 ET — building opening range, no trades yet")
        return

    # Fetch SPY trend once and share across all symbols
    spy_bullish = get_spy_trend()

    # Fetch dynamic symbols using OpenBB screener
    active_symbols = get_active_symbols()
    if not active_symbols:
        log.warning("No active symbols found. Skipping run.")
        return

    # Budget setup
    try:
        open_positions_count = len(api.list_positions())
    except Exception as e:
        log.error(f"Failed to fetch existing positions: {e}")
        open_positions_count = 0
        
    max_open_positions = max(1, int(MAX_TOTAL_INVESTMENT / ORB_POSITION_SIZE))

    for symbol in active_symbols:
        try:
            opened_new = process_symbol(symbol, spy_bullish, open_positions_count, max_open_positions)
            if opened_new:
                open_positions_count += 1
        except Exception as e:
            log.error(f"{symbol}: unexpected error — {e}")


if __name__ == "__main__":
    log.info("ORB Bot starting (Dynamic Symbols via OpenBB)")
    log.info(f"OR: first {ORB_RANGE_BARS} bars | ${ORB_POSITION_SIZE}/trade | Default target: {ORB_PROFIT_MULTIPLIER}× range | EOD: {ORB_CLOSE_HOUR}:{ORB_CLOSE_MINUTE:02d} ET")
    log.info(f"Filters: SPY trend | min OR {ORB_MIN_OR_PCT:.1%} | per-symbol multipliers")
    run_orb()
    log.info("Run complete")
