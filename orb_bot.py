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

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, AssetClass, PositionSide
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import (
    DAILY_LOSS_LIMIT_PCT,
    MAX_TOTAL_INVESTMENT,
    ORB_CLOSE_HOUR,
    ORB_CLOSE_MINUTE,
    ORB_EQUITY_BLOCKLIST,
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

# ---------------------------------------------------------------------------
# Optional ML scorer — loaded at startup if model exists; bot runs without it
# ---------------------------------------------------------------------------
try:
    import joblib
    from ml.features import ML_CONFIDENCE_THRESHOLD, compute_features
    _SCORER = joblib.load("ml/models/breakout_scorer.pkl")
    _ML_ENABLED = True
except Exception:
    _SCORER = None
    _ML_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

_key = os.environ["ALPACA_API_KEY"]
_secret = os.environ["ALPACA_SECRET_KEY"]
trading_client = TradingClient(_key, _secret, paper=True)
data_client = StockHistoricalDataClient(_key, _secret)

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
        response = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=market_open,
        ))
        df = response.df
        if df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]
        return df if not df.empty else None
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
        orders = trading_client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.FILLED,
            after=today_start,
            symbols=[symbol],
        ))
        return any(o.side == OrderSide.BUY for o in orders)
    except Exception as e:
        log.warning(f"{symbol}: could not check today's orders — {e}")
        return False


def _was_entered_today(symbol: str) -> bool:
    """True if a BUY was filled for this symbol during today's session."""
    today_start = get_now_et().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        orders = trading_client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.FILLED,
            after=today_start,
            symbols=[symbol],
        ))
        return any(o.side == OrderSide.BUY for o in orders)
    except Exception:
        return True   # fail-safe: don't close if the check itself errors


def close_stale_positions() -> int:
    """
    Close every equity position that has no BUY order from today's session.

    This is the safety net for missed EOD closes (GitHub Actions delay/skip).
    Called unconditionally at the top of every run — even when the market is
    closed — so a position stranded overnight gets cleaned up the next morning.

    Returns the number of stale positions closed.
    """
    closed = 0
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        log.error(f"close_stale_positions: could not fetch positions — {e}")
        return 0

    for pos in positions:
        if pos.asset_class != AssetClass.US_EQUITY:
            continue
        sym = pos.symbol
        if _was_entered_today(sym):
            continue   # opened today — normal in-session position, leave it alone

        qty        = abs(int(float(pos.qty)))
        unreal_pl  = float(pos.unrealized_pl)
        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        log.warning(
            f"STALE POSITION {sym} × {qty}"
            f" | Unrealised P&L: ${unreal_pl:+.2f}"
            f" | Entered before today — force closing now"
        )
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=sym, qty=qty,
                side=close_side,
                time_in_force=TimeInForce.DAY,
            ))
            log.info(f"STALE CLOSE — {sym} × {qty}")
            closed += 1
        except Exception as e:
            log.error(f"Stale close failed for {sym}: {e}")

    return closed


def get_position(symbol: str):
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Improvement 1: SPY trend filter
# ---------------------------------------------------------------------------

def get_spy_trend() -> tuple[bool | None, float]:
    """
    Return (is_trending_up, trend_pct) for SPY today.
    trend_pct = (last - open) / open; 0.0 if data unavailable.
    """
    bars = get_today_bars("SPY")
    if bars is None or len(bars) < 1:
        return None, 0.0
    spy_open = float(bars.iloc[0]["open"])
    spy_last = float(bars.iloc[-1]["close"])
    trend_pct = (spy_last - spy_open) / spy_open if spy_open > 0 else 0.0
    trending_up = spy_last >= spy_open
    log.info(f"SPY trend: open={spy_open:.2f}  last={spy_last:.2f}  {'UP' if trending_up else 'DOWN'}  ({trend_pct:+.2%})")
    return trending_up, trend_pct


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def process_symbol(symbol: str, spy_bullish: bool | None, spy_trend_pct: float,
                   open_positions_count: int, max_open_positions: int) -> bool:
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
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            ))
            log.info(f"EOD CLOSE — SELL {qty} {symbol} at ~{current_price:.2f}")
        return False

    # --- Manage existing position ---
    if position:
        pnl_pct = float(position.unrealized_plpc) * 100

        if current_price <= stop_price:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            ))
            log.info(f"STOP LOSS — SELL {qty} {symbol} at ~{current_price:.2f} | P&L: {pnl_pct:.2f}%")
            return False

        if current_price >= target_price:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            ))
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

        # Optional ML gate — only active once breakout_scorer.pkl exists
        if _ML_ENABLED:
            feats = compute_features(
                or_high=or_high, or_low=or_low,
                breakout_price=current_price,
                volume=current_volume, avg_or_volume=avg_or_volume,
                bar_et=now_et,
                spy_trend_pct=spy_trend_pct,
            )
            confidence = _SCORER.predict_proba([feats])[0][1]
            log.info(f"{symbol} | ML confidence: {confidence:.0%}")
            if confidence < ML_CONFIDENCE_THRESHOLD:
                log.info(f"{symbol} | ML score below threshold ({ML_CONFIDENCE_THRESHOLD:.0%}) — skipping entry")
                return False

        trading_client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=trade_qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
        ))
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
    now_et = get_now_et()
    log.info(f"--- ORB check at {now_et.strftime('%H:%M')} ET ---")

    # ── Safety net: close stale positions from previous sessions ────────────
    # Runs unconditionally before any market-hours check.
    # If yesterday's EOD close was missed (GitHub Actions delay/skip), this
    # detects positions with no today BUY order and closes them immediately.
    stale_closed = close_stale_positions()
    if stale_closed:
        log.warning(f"Stale cleanup closed {stale_closed} position(s) from previous session(s)")

    if not trading_client.get_clock().is_open:
        log.info("Market closed — skipping run")
        return

    # Daily loss circuit-breaker (from QuantTrading risk_engine.py)
    # Blocks new entries; existing positions are still managed below.
    _block_new_entries = False
    try:
        account = trading_client.get_account()
        equity      = float(account.equity)
        last_equity = float(account.last_equity)
        if last_equity > 0:
            daily_pnl_pct = (equity - last_equity) / last_equity
            if daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
                log.warning(f"Daily loss limit hit ({daily_pnl_pct:.1%}) — managing existing positions only, no new entries")
                _block_new_entries = True
    except Exception:
        pass

    # Hard EOD sweep — runs directly from the positions list, no bar data needed.
    # Prevents overnight holds even if bar fetching fails at close time.
    if now_et.time() >= CLOSE_TIME:
        log.info("EOD — force-closing all equity positions")
        try:
            for pos in trading_client.get_all_positions():
                if pos.asset_class != AssetClass.US_EQUITY:
                    continue
                qty = abs(int(float(pos.qty)))
                if qty > 0:
                    try:
                        trading_client.submit_order(MarketOrderRequest(
                            symbol=pos.symbol, qty=qty,
                            side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        ))
                        log.info(f"EOD CLOSE — {pos.symbol} × {qty}")
                    except Exception as e:
                        log.error(f"EOD close failed for {pos.symbol}: {e}")
        except Exception as e:
            log.error(f"EOD sweep failed: {e}")
        return

    # Before 10:00 AM: opening range not ready for any symbol
    if now_et.time() < time(10, 0):
        log.info("Before 10:00 ET — building opening range, no trades yet")
        return

    # Fetch SPY trend once and share across all symbols
    spy_bullish, spy_trend_pct = get_spy_trend()

    # -----------------------------------------------------------------------
    # STEP 1: Always manage ALL currently held positions first.
    # This ensures stop-loss, take-profit, and EOD close fire even when a
    # held symbol is not in today's screener list.
    # -----------------------------------------------------------------------
    try:
        open_positions = trading_client.get_all_positions()
    except Exception as e:
        log.error(f"Failed to fetch existing positions: {e}")
        open_positions = []

    held_symbols = set()
    for pos in open_positions:
        held_symbols.add(pos.symbol)
        try:
            process_symbol(pos.symbol, spy_bullish, spy_trend_pct, len(open_positions), 0)
        except Exception as e:
            log.error(f"{pos.symbol}: unexpected error managing position — {e}")

    open_positions_count = len(open_positions)
    max_open_positions = max(1, int(MAX_TOTAL_INVESTMENT / ORB_POSITION_SIZE))

    # -----------------------------------------------------------------------
    # STEP 2: Screen for new entry opportunities (skip already-held symbols).
    # -----------------------------------------------------------------------
    if _block_new_entries:
        log.info("Skipping new entries — daily loss limit active")
        return

    raw_symbols    = get_active_symbols()
    active_symbols = [s for s in raw_symbols if s not in ORB_EQUITY_BLOCKLIST]
    blocked        = [s for s in raw_symbols if s in ORB_EQUITY_BLOCKLIST]
    if blocked:
        log.info(f"Screener blocklist removed: {', '.join(blocked)}")
    if not active_symbols:
        log.warning("No active symbols found from screener. Skipping new entries.")
    else:
        for symbol in active_symbols:
            if symbol in held_symbols:
                # Already managed above — don't double-process
                continue
            try:
                opened_new = process_symbol(symbol, spy_bullish, spy_trend_pct, open_positions_count, max_open_positions)
                if opened_new:
                    open_positions_count += 1
            except Exception as e:
                log.error(f"{symbol}: unexpected error — {e}")


if __name__ == "__main__":
    log.info("ORB Bot starting (Dynamic Symbols via OpenBB)")
    log.info(f"ML scorer: {'ENABLED (threshold {ML_CONFIDENCE_THRESHOLD:.0%})' if _ML_ENABLED else 'DISABLED (model not found — running rule-based only)'}")
    log.info(f"OR: first {ORB_RANGE_BARS} bars | ${ORB_POSITION_SIZE}/trade | Default target: {ORB_PROFIT_MULTIPLIER}× range | EOD: {ORB_CLOSE_HOUR}:{ORB_CLOSE_MINUTE:02d} ET")
    log.info(f"Filters: SPY trend | min OR {ORB_MIN_OR_PCT:.1%} | per-symbol multipliers")
    run_orb()
    log.info("Run complete")
