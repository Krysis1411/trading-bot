"""
Live ORB (Opening Range Breakout) day trading bot — Alpaca options paper trading.
Fetches options chains via yfinance, evaluates Implied Volatility (IV),
and dynamically submits multi-leg defined-risk orders (Spreads, Straddles, Iron Condors).
"""
import logging
import os
import re
from datetime import datetime, time, date, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest, GetOptionContractsRequest, OptionLegRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, PositionIntent, AssetClass, PositionSide, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import (
    ORB_CLOSE_HOUR,
    ORB_CLOSE_MINUTE,
    ORB_MIN_OR_PCT,
    ORB_OPTIONS_POSITION_SIZE,
    MAX_OPTIONS_INVESTMENT,
    ORB_PROFIT_MULTIPLIER,
    ORB_PROFIT_MULTIPLIERS,
    ORB_RANGE_BARS,
    ORB_STOP_BUFFER,
    ORB_VOLUME_FACTOR,
    ORB_OPTIONS_IV_THRESHOLD,
)
from screener import get_active_symbols

load_dotenv()

DRY_RUN = False  # Overridden to True by --dry-run CLI flag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

key = os.environ.get("ALPACA_API_KEY")
secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")

if not key or not secret:
    raise EnvironmentError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY/ALPACA_API_SECRET in environment")

trading_client = TradingClient(key, secret, paper=True)
data_client = StockHistoricalDataClient(key, secret)

ET = ZoneInfo("America/New_York")
CLOSE_TIME = time(ORB_CLOSE_HOUR, ORB_CLOSE_MINUTE)


# ---------------------------------------------------------------------------
# Account validation
# ---------------------------------------------------------------------------

def check_options_approval() -> bool:
    """Return True if the account has options Level 2+ (required for spreads and condors)."""
    try:
        account = trading_client.get_account()
        level = int(getattr(account, "options_approved_level", 0) or 0)
        if level < 2:
            log.warning(
                f"Options approval level {level} — Level 2 required for multi-leg strategies. "
                "Enable options trading in your Alpaca account settings."
            )
            return False
        log.info(f"Options approval: Level {level} confirmed")
        return True
    except Exception as e:
        log.warning(f"Could not verify options approval level: {e}")
        return False


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


def get_underlying_symbol(option_symbol: str) -> str:
    """Parse underlying symbol from options contract (e.g. NVDA260529C00212500 -> NVDA)."""
    match = re.match(r"^([A-Za-z]+)\d", option_symbol)
    return match.group(1) if match else option_symbol


def already_traded_today(symbol: str) -> bool:
    """Return True if an options order was already placed for this symbol today."""
    today_start = get_now_et().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        orders = trading_client.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            after=today_start,
        ))
        for o in orders:
            if get_underlying_symbol(o.symbol) == symbol:
                return True
        return False
    except Exception as e:
        log.warning(f"{symbol}: could not check today's orders — {e}")
        return False


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
# Options Strategy Classifier
# ---------------------------------------------------------------------------

def classify_holding(positions: list) -> str:
    """Determine the type of multi-leg option strategy being held based on positions."""
    calls = []
    puts = []
    for pos in positions:
        sym = pos.symbol
        match = re.search(r"\d{6}([CP])\d{8}", sym)
        if match:
            opt_type = 'call' if match.group(1) == 'C' else 'put'
        else:
            opt_type = 'call' if 'C' in sym else 'put'
        
        is_long = float(pos.qty) > 0
        if opt_type == 'call':
            calls.append((pos, is_long))
        else:
            puts.append((pos, is_long))
            
    if len(calls) > 0 and len(puts) > 0:
        any_short = any(not is_long for _, is_long in calls + puts)
        if any_short:
            return "iron_condor"
        else:
            return "straddle"
    elif len(calls) > 0:
        return "call_spread"
    elif len(puts) > 0:
        return "put_spread"
    return "unknown"


def close_all_legs(positions: list) -> None:
    """Submit market orders to close all positions in the list."""
    for pos in positions:
        if DRY_RUN:
            log.info(f"[DRY RUN] Would close {pos.symbol} (Qty: {pos.qty})")
            continue
        try:
            close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            req = MarketOrderRequest(
                symbol=pos.symbol,
                qty=abs(int(float(pos.qty))),
                side=close_side,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(req)
            log.info(f"Closed leg {pos.symbol} (Qty: {pos.qty})")
        except Exception as e:
            log.error(f"Failed to close leg {pos.symbol}: {e}")


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def process_symbol_options(symbol: str, spy_bullish: bool | None, open_positions_count: int, max_open_positions: int) -> bool:
    """
    Evaluate ORB and volatility conditions for a symbol. If setup exists,
    selects options strategy via OpenBB and submits an Alpaca MLEG order.
    Returns True if a new position was opened, False otherwise.
    """
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

    if ORB_MIN_OR_PCT > 0 and or_range / or_high < ORB_MIN_OR_PCT:
        log.info(f"{symbol}: OR too narrow ({or_range / or_high:.3%} < {ORB_MIN_OR_PCT:.1%}) — skipping")
        return False

    current_bar = bars.iloc[-1]
    current_price = float(current_bar["close"])
    current_volume = float(current_bar["volume"])
    now_et = get_now_et()

    # Budget Check
    if open_positions_count >= max_open_positions:
        log.info(f"{symbol} | Budget limit reached ({open_positions_count}/{max_open_positions} trades open) — skipping entry to stay under ${MAX_OPTIONS_INVESTMENT}")
        return False

    if already_traded_today(symbol):
        log.info(f"{symbol} | Already traded today — skipping entry")
        return False

    # Evaluate breakout and range conditions
    is_breakout_above = current_price > or_high
    is_breakout_below = current_price < or_low
    vol_ok = current_volume >= avg_or_volume * ORB_VOLUME_FACTOR
    
    is_after_1030 = now_et.time() >= time(10, 30)
    is_range_bound = (
        is_after_1030 and 
        (or_low + 0.2 * or_range) <= current_price <= (or_high - 0.2 * or_range)
    )

    if not ((is_breakout_above and vol_ok) or (is_breakout_below and vol_ok) or is_range_bound):
        log.info(f"{symbol} | No option setup found (Price: {current_price:.2f} | OR: {or_low:.2f}–{or_high:.2f})")
        return False

    # Fetch option chain via yfinance
    try:
        log.info(f"{symbol} | Fetching option chain via yfinance...")
        ticker = yf.Ticker(symbol)
        available = ticker.options  # tuple of expiry strings e.g. ('2026-05-30', ...)
        if not available:
            log.warning(f"{symbol} | No options expirations available")
            return False

        today_str = date.today().isoformat()
        valid_exps = [e for e in available if e >= today_str]
        if not valid_exps:
            log.warning(f"{symbol} | No valid expirations found")
            return False

        nearest_expiry = valid_exps[0]
        chain = ticker.option_chain(nearest_expiry)
        calls = chain.calls
        puts = chain.puts
    except Exception as e:
        log.error(f"{symbol} | Failed to fetch option chain: {e}")
        return False

    if calls.empty or puts.empty:
        log.warning(f"{symbol} | Missing Call or Put contracts for expiry {nearest_expiry}")
        return False

    # Sort strikes by closeness to current price to find ATM contracts
    calls_sorted = calls.iloc[(calls['strike'] - current_price).abs().argsort()]
    puts_sorted = puts.iloc[(puts['strike'] - current_price).abs().argsort()]

    atm_call = calls_sorted.iloc[0]
    atm_put = puts_sorted.iloc[0]

    avg_atm_iv = (float(atm_call['impliedVolatility']) + float(atm_put['impliedVolatility'])) / 2
    log.info(f"{symbol} | ATM Call: {atm_call['contractSymbol']} (Strike: {atm_call['strike']} | Ask: {atm_call['ask']})")
    log.info(f"{symbol} | ATM Put: {atm_put['contractSymbol']} (Strike: {atm_put['strike']} | Ask: {atm_put['ask']})")
    log.info(f"{symbol} | Average ATM IV: {avg_atm_iv:.2%}")

    strategy = None
    legs = []
    qty = 0
    limit_price = 0.05  # set per strategy below; positive = debit, negative = credit

    if avg_atm_iv > ORB_OPTIONS_IV_THRESHOLD and is_range_bound:
        # Strategy: Iron Condor (High IV & Range-bound)
        strategy = "Iron Condor"
        
        # Strike spacing based on price
        if current_price > 100:
            width = 5.0
        elif current_price > 50:
            width = 2.0
        else:
            width = 1.0

        # Short Call: strike closest to or_high (must be > current_price)
        sc_candidates = calls[calls['strike'] > current_price]
        if sc_candidates.empty:
            log.warning(f"{symbol} | No short call candidates")
            return False
        short_call = sc_candidates.iloc[(sc_candidates['strike'] - or_high).abs().argsort()].iloc[0]
        
        # Long Call: strike closest to short_call strike + width
        lc_candidates = calls[calls['strike'] > short_call['strike']]
        if lc_candidates.empty:
            log.warning(f"{symbol} | No long call candidates")
            return False
        long_call = lc_candidates.iloc[(lc_candidates['strike'] - (short_call['strike'] + width)).abs().argsort()].iloc[0]

        # Short Put: strike closest to or_low (must be < current_price)
        sp_candidates = puts[puts['strike'] < current_price]
        if sp_candidates.empty:
            log.warning(f"{symbol} | No short put candidates")
            return False
        short_put = sp_candidates.iloc[(sp_candidates['strike'] - or_low).abs().argsort()].iloc[0]

        # Long Put: strike closest to short_put strike - width
        lp_candidates = puts[puts['strike'] < short_put['strike']]
        if lp_candidates.empty:
            log.warning(f"{symbol} | No long put candidates")
            return False
        long_put = lp_candidates.iloc[(lp_candidates['strike'] - (short_put['strike'] - width)).abs().argsort()].iloc[0]

        # Compute net credit received
        net_credit = (
            (float(short_call['bid']) - float(long_call['ask'])) +
            (float(short_put['bid']) - float(long_put['ask']))
        )
        if net_credit <= 0:
            log.warning(f"{symbol} | Iron Condor net credit is negative ({net_credit:.2f}) — skipping")
            return False

        risk = width - net_credit
        if risk <= 0:
            risk = 0.05
        
        qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (risk * 100)))
        legs = [
            OptionLegRequest(symbol=short_call['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_call['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=short_put['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_put['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
        ]
        limit_price = -round(net_credit, 2)  # negative = credit received
        log.info(f"{symbol} | Iron Condor | Credit: ${net_credit:.2f} | Risk: ${risk:.2f} | Qty: {qty} | Limit: ${limit_price:.2f}")

    elif avg_atm_iv <= ORB_OPTIONS_IV_THRESHOLD:
        # Low IV: we BUY premium
        if is_breakout_above:
            if spy_bullish is True:
                # Bull Call Spread (breakout aligned with SPY trend)
                strategy = "Bull Call Spread"
                long_call = atm_call
                
                oc_candidates = calls[calls['strike'] > long_call['strike']]
                if oc_candidates.empty:
                    log.warning(f"{symbol} | No short call candidates for spread")
                    return False
                short_call = oc_candidates.iloc[(oc_candidates['strike'] - (long_call['strike'] + or_range)).abs().argsort()].iloc[0]

                net_debit = float(long_call['ask']) - float(short_call['bid'])
                if net_debit <= 0:
                    net_debit = 0.05
                
                qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
                legs = [
                    OptionLegRequest(symbol=long_call['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                    OptionLegRequest(symbol=short_call['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
                ]
                limit_price = round(net_debit, 2)  # positive = debit paid
                log.info(f"{symbol} | Bull Call Spread | Debit: ${net_debit:.2f} | Qty: {qty} | Limit: ${limit_price:.2f}")
            else:
                # Straddle (uncorrelated breakout)
                strategy = "Straddle"
                net_debit = float(atm_call['ask']) + float(atm_put['ask'])
                qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
                legs = [
                    OptionLegRequest(symbol=atm_call['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                    OptionLegRequest(symbol=atm_put['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                ]
                limit_price = round(net_debit, 2)  # positive = debit paid
                log.info(f"{symbol} | Straddle | Debit: ${net_debit:.2f} | Qty: {qty} | Limit: ${limit_price:.2f}")

        elif is_breakout_below:
            if spy_bullish is False:
                # Bear Put Spread (breakout aligned with SPY trend)
                strategy = "Bear Put Spread"
                long_put = atm_put
                
                op_candidates = puts[puts['strike'] < long_put['strike']]
                if op_candidates.empty:
                    log.warning(f"{symbol} | No short put candidates for spread")
                    return False
                short_put = op_candidates.iloc[(op_candidates['strike'] - (long_put['strike'] - or_range)).abs().argsort()].iloc[0]

                net_debit = float(long_put['ask']) - float(short_put['bid'])
                if net_debit <= 0:
                    net_debit = 0.05

                qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
                legs = [
                    OptionLegRequest(symbol=long_put['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                    OptionLegRequest(symbol=short_put['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
                ]
                limit_price = round(net_debit, 2)  # positive = debit paid
                log.info(f"{symbol} | Bear Put Spread | Debit: ${net_debit:.2f} | Qty: {qty} | Limit: ${limit_price:.2f}")
            else:
                # Straddle (uncorrelated breakdown)
                strategy = "Straddle"
                net_debit = float(atm_call['ask']) + float(atm_put['ask'])
                qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
                legs = [
                    OptionLegRequest(symbol=atm_call['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                    OptionLegRequest(symbol=atm_put['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                ]
                limit_price = round(net_debit, 2)  # positive = debit paid
                log.info(f"{symbol} | Straddle | Debit: ${net_debit:.2f} | Qty: {qty} | Limit: ${limit_price:.2f}")

    if strategy is None:
        log.info(f"{symbol} | No options setup matched the criteria (IV: {avg_atm_iv:.2%})")
        return False

    # Place MLEG order
    # limit_price: positive = net debit (we pay), negative = net credit (we receive)
    if DRY_RUN:
        log.info(
            f"[DRY RUN] Would submit {strategy} for {symbol}"
            f" | Legs: {[l.symbol for l in legs]}"
            f" | Qty: {qty} | Limit: ${limit_price:.2f}"
        )
        return True

    try:
        req = LimitOrderRequest(
            qty=qty,
            limit_price=limit_price,
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            legs=legs,
        )
        order = trading_client.submit_order(req)
        log.info(f"SUBMITTED {strategy} for {symbol} | ID: {order.id} | Qty: {qty} | Limit: ${limit_price:.2f}")
        return True
    except Exception as e:
        log.error(f"{symbol} | Failed to submit MLEG order: {e}")
        return False


def manage_existing_options_positions(underlying_symbol: str, positions: list, spy_bullish: bool | None) -> None:
    """Monitor underlying stock price levels to exit options strategies dynamically."""
    strategy = classify_holding(positions)
    
    bars = get_today_bars(underlying_symbol)
    if bars is None:
        return
    or_result = compute_opening_range(bars)
    if or_result is None:
        return
    or_high, or_low, _ = or_result
    or_range = or_high - or_low
    
    current_price = float(bars.iloc[-1]["close"])
    now_et = get_now_et()
    
    # Forced close at EOD
    if now_et.time() >= CLOSE_TIME:
        log.info(f"EOD CLOSE — closing all options legs for {underlying_symbol}")
        close_all_legs(positions)
        return
        
    profit_multiplier = ORB_PROFIT_MULTIPLIERS.get(underlying_symbol, ORB_PROFIT_MULTIPLIER)
    
    if strategy == "call_spread":
        stop_price = or_low - ORB_STOP_BUFFER
        target_price = or_high + or_range * profit_multiplier
        if current_price <= stop_price:
            log.info(f"STOP LOSS — Closing Call Spread for {underlying_symbol} (price {current_price:.2f} <= {stop_price:.2f})")
            close_all_legs(positions)
        elif current_price >= target_price:
            log.info(f"TAKE PROFIT — Closing Call Spread for {underlying_symbol} (price {current_price:.2f} >= {target_price:.2f})")
            close_all_legs(positions)
            
    elif strategy == "put_spread":
        stop_price = or_high + ORB_STOP_BUFFER
        target_price = or_low - or_range * profit_multiplier
        if current_price >= stop_price:
            log.info(f"STOP LOSS — Closing Put Spread for {underlying_symbol} (price {current_price:.2f} >= {stop_price:.2f})")
            close_all_legs(positions)
        elif current_price <= target_price:
            log.info(f"TAKE PROFIT — Closing Put Spread for {underlying_symbol} (price {current_price:.2f} <= {target_price:.2f})")
            close_all_legs(positions)
            
    elif strategy == "straddle":
        target_high = or_high + or_range * profit_multiplier
        target_low = or_low - or_range * profit_multiplier
        # Stop loss if total P&L is down -50%
        try:
            total_pnl_pct = sum(float(pos.unrealized_plpc) for pos in positions) * 100
        except Exception:
            total_pnl_pct = 0.0
            
        if current_price >= target_high or current_price <= target_low:
            log.info(f"TAKE PROFIT — Closing Straddle for {underlying_symbol} (price {current_price:.2f})")
            close_all_legs(positions)
        elif total_pnl_pct <= -50.0:
            log.info(f"STOP LOSS — Closing Straddle for {underlying_symbol} (P&L: {total_pnl_pct:.2f}%)")
            close_all_legs(positions)
            
    elif strategy == "iron_condor":
        # Stop loss if the stock breaches the short strikes (or_low / or_high)
        if current_price >= or_high or current_price <= or_low:
            log.info(f"STOP LOSS — Closing Iron Condor for {underlying_symbol} (price {current_price:.2f} breached {or_low:.2f}-{or_high:.2f})")
            close_all_legs(positions)
        else:
            log.info(f"{underlying_symbol} | Iron Condor holding inside range (Price: {current_price:.2f} vs {or_low:.2f}-{or_high:.2f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_orb_options() -> None:
    if not DRY_RUN and not trading_client.get_clock().is_open:
        log.info("Market closed — skipping run")
        return

    if not DRY_RUN and not check_options_approval():
        log.error("Options Level 2 not approved — cannot submit multi-leg orders. Exiting.")
        return

    now_et = get_now_et()
    log.info(f"--- ORB Options check at {now_et.strftime('%H:%M')} ET ---")

    # Before 10:00 AM: opening range not ready
    if now_et.time() < time(10, 0):
        log.info("Before 10:00 ET — building opening range, no options trades yet")
        return

    # Fetch SPY trend once and share across all symbols
    spy_bullish = get_spy_trend()

    # 1. Fetch existing positions and group them by underlying symbol
    try:
        raw_positions = trading_client.get_all_positions()
        # Filter for options positions
        options_positions = [pos for pos in raw_positions if pos.asset_class == AssetClass.US_OPTION]
    except Exception as e:
        log.error(f"Failed to fetch existing positions: {e}")
        options_positions = []

    # Group by underlying
    grouped_holdings = {}
    for pos in options_positions:
        underlying = get_underlying_symbol(pos.symbol)
        if underlying not in grouped_holdings:
            grouped_holdings[underlying] = []
        grouped_holdings[underlying].append(pos)

    # Manage existing options positions first
    held_symbols = set(grouped_holdings.keys())
    for underlying, pos_list in grouped_holdings.items():
        try:
            manage_existing_options_positions(underlying, pos_list, spy_bullish)
        except Exception as e:
            log.error(f"{underlying}: unexpected error managing options positions — {e}")

    # Set up budgets
    # Number of active options strategies (grouped by underlying symbol)
    open_positions_count = len(held_symbols)
    max_open_positions = max(1, int(MAX_OPTIONS_INVESTMENT / ORB_OPTIONS_POSITION_SIZE))

    # 2. Screen for new options entry opportunities
    active_symbols = get_active_symbols()
    if not active_symbols:
        log.warning("No active symbols found from screener. Skipping new entries.")
    else:
        for symbol in active_symbols:
            if symbol in held_symbols:
                continue
            try:
                opened_new = process_symbol_options(symbol, spy_bullish, open_positions_count, max_open_positions)
                if opened_new:
                    open_positions_count += 1
            except Exception as e:
                log.error(f"{symbol}: unexpected error — {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ORB Options Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log strategy selection and leg structure without submitting any orders",
    )
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if DRY_RUN:
        log.info("*** DRY RUN MODE — no orders will be submitted ***")

    log.info("ORB Options Bot starting (Dynamic Symbols via OpenBB)")
    log.info(f"OR: first {ORB_RANGE_BARS} bars | ${ORB_OPTIONS_POSITION_SIZE}/trade | Default target: {ORB_PROFIT_MULTIPLIER}× range | EOD: {ORB_CLOSE_HOUR}:{ORB_CLOSE_MINUTE:02d} ET")
    log.info(f"Filters: SPY trend | min OR {ORB_MIN_OR_PCT:.1%} | IV boundary {ORB_OPTIONS_IV_THRESHOLD:.0%}")
    run_orb_options()
    log.info("Run complete")
