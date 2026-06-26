"""
Live ORB (Opening Range Breakout) day trading bot — Alpaca options paper trading.
Fetches options chains via yfinance, evaluates Implied Volatility (IV),
and dynamically submits multi-leg defined-risk orders (Spreads, Straddles, Iron Condors).
"""
import logging
import math
import os
import re
import time as _time
from datetime import datetime, time, date, timezone, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

import yfinance as yf
from ml.iv_calculator import compute_iv_rank, compute_iv_skew, enrich_chain
from ml.regime_detector import RegimeResult, daily_trend, detect_regime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest, OptionLegRequest
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
    DAILY_LOSS_LIMIT_PCT,
    MAX_DRAWDOWN_PCT,
    IC_MAX_DTE,
    MIN_UNDERLYING_PRICE,
    IC_MIN_CREDIT_RATIO,
    IC_SIGMA_MULTIPLE,
    IC_PROFIT_TARGET_PCT,
    IC_PNL_STOP_MULTIPLE,
    ORB_OPTIONS_BLOCKLIST,
    SKIP_MONDAY_ENTRIES,
    IC_MAX_ENTRY_HOUR,
    IC_MAX_ENTRY_MINUTE,
    MIN_BREAKOUT_STRENGTH_PCT,
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


def check_risk_limits() -> bool:
    """
    Adapted from trading00money/QuantTrading risk_engine.py.
    Returns False (block new entries) when:
      - Today's P&L < -DAILY_LOSS_LIMIT_PCT of yesterday's equity, OR
      - Total drawdown from peak > MAX_DRAWDOWN_PCT
    """
    try:
        account = trading_client.get_account()
        equity      = float(account.equity)
        last_equity = float(account.last_equity)   # previous session close

        # Daily loss circuit-breaker
        if last_equity > 0:
            daily_pnl_pct = (equity - last_equity) / last_equity
            if daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
                log.warning(
                    f"Daily loss limit hit ({daily_pnl_pct:.1%} vs limit -{DAILY_LOSS_LIMIT_PCT:.0%}) "
                    "— no new entries today"
                )
                return False

        # Drawdown kill-switch vs all-time peak in this session
        # Use buying_power + portfolio_value as a rough peak proxy
        portfolio = float(account.portfolio_value)
        peak = max(equity, portfolio)
        if peak > 0:
            drawdown = (peak - equity) / peak
            if drawdown >= MAX_DRAWDOWN_PCT:
                log.warning(
                    f"Max drawdown hit ({drawdown:.1%} vs limit {MAX_DRAWDOWN_PCT:.0%}) "
                    "— stopping all new entries"
                )
                return False

        log.info(f"Risk OK — equity ${equity:,.0f} | daily P&L {(equity-last_equity)/last_equity:+.1%}")
        return True
    except Exception as e:
        log.warning(f"Could not check risk limits: {e} — proceeding")
        return True


def get_daily_bars(symbol: str, lookback_days: int = 60) -> pd.DataFrame | None:
    """Fetch daily bars for regime detection and daily trend filter."""
    start = get_now_et() - timedelta(days=lookback_days)
    try:
        response = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
        ))
        df = response.df
        if df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]
        return df if not df.empty else None
    except Exception as e:
        log.warning(f"{symbol}: failed to fetch daily bars — {e}")
        return None


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


def _close_stale_options() -> None:
    """
    Close options legs that have no matching equity BUY order from today.
    Safety net for missed EOD closes — runs on every bot execution before
    the market-hours check so overnight holds are caught the next morning.
    """
    today_start = get_now_et().replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        all_pos = trading_client.get_all_positions()
        options_legs = [p for p in all_pos if p.asset_class == AssetClass.US_OPTION]
    except Exception as e:
        log.error(f"_close_stale_options: could not fetch positions — {e}")
        return

    if not options_legs:
        return

    # Extract the underlying symbol from an option symbol like "AAPL260627C00200000"
    underlying_re = re.compile(r"^([A-Z]+)\d")

    for leg in options_legs:
        m = underlying_re.match(leg.symbol)
        if not m:
            continue
        underlying = m.group(1)
        try:
            orders = trading_client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.FILLED,
                after=today_start,
                symbols=[underlying],
            ))
            entered_today = any(o.side == OrderSide.BUY for o in orders)
        except Exception:
            entered_today = True   # fail-safe: don't close if check errors

        if not entered_today:
            log.warning(
                f"STALE OPTIONS LEG {leg.symbol} (underlying {underlying})"
                f" | Unrealised P&L: ${float(leg.unrealized_pl):+.2f}"
                f" | Entered before today — force closing"
            )
            close_all_legs([leg])


def close_all_legs(positions: list) -> None:
    """
    Close all legs with two fixes:
    1. Long legs (sell-to-close) before short legs (buy-to-close).
       Prevents Alpaca's 'uncovered option' rejection that fires when the
       short protective leg is bought back first, leaving the long exposed.
    2. Near-worthless options ($0.05 or less) use a $0.01 limit order
       instead of market, avoiding the 'no available quote' rejection.
    """
    # Sort: long positions first (0), then short positions (1)
    ordered = sorted(positions, key=lambda p: 0 if float(p.qty) > 0 else 1)

    for pos in ordered:
        if DRY_RUN:
            log.info(f"[DRY RUN] Would close {pos.symbol} (Qty: {pos.qty})")
            continue
        try:
            qty       = abs(int(float(pos.qty)))
            close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            price     = float(pos.current_price) if hasattr(pos, "current_price") else 1.0

            if price <= 0.05:
                # No market bid — use a $0.01 limit so the order is accepted
                req = LimitOrderRequest(
                    symbol=pos.symbol, qty=qty,
                    side=close_side, time_in_force=TimeInForce.DAY,
                    limit_price=0.01,
                )
            else:
                req = MarketOrderRequest(
                    symbol=pos.symbol, qty=qty,
                    side=close_side, time_in_force=TimeInForce.DAY,
                )
            trading_client.submit_order(req)
            log.info(f"Closed leg {pos.symbol} (Qty: {pos.qty})")
        except Exception as e:
            log.error(f"Failed to close leg {pos.symbol}: {e}")


# ---------------------------------------------------------------------------
# Market situation classifier
# ---------------------------------------------------------------------------

from enum import Enum


class MarketSituation(Enum):
    PREMIUM_SELL      = "premium_sell"       # Range-bound + high IV → Iron Condor
    DIRECTIONAL_FULL  = "directional_full"   # Breakout + low IV + all signals → debit spread
    STRADDLE_PLAY     = "straddle_play"      # Breakout + low IV + partial signals → Straddle
    VOLATILITY_SPIKE  = "volatility_spike"   # Breakout + high IV → skip (overpriced + risky)
    LOW_IV_RANGEBOUND = "low_iv_rangebound"  # Range-bound + low IV → skip (no edge)
    NO_SETUP          = "no_setup"           # No clean signal


def _classify_situation(
    is_breakout_above: bool,
    is_breakout_below: bool,
    is_range_bound: bool,
    iv_rank: float,
    iv_rank_threshold: float,
    spy_bullish: bool | None,
    daily_trend: str,
    iv_skew: float,
) -> tuple[MarketSituation, int]:
    """
    Map all market signals to a single MarketSituation and a directional
    confirmation score (0–3). The score counts how many of the three
    independent confirmation signals (SPY trend, daily trend, IV skew) agree
    with the breakout direction.

    Score 3 → DIRECTIONAL_FULL (debit spread)
    Score 1–2 → STRADDLE_PLAY (expect a move, unsure of direction)
    Score 0 → NO_SETUP
    """
    high_iv = iv_rank > iv_rank_threshold
    is_breakout = is_breakout_above or is_breakout_below

    # ── Range-bound scenarios ────────────────────────────────────────────────
    if is_range_bound and not is_breakout:
        if high_iv:
            return MarketSituation.PREMIUM_SELL, 0
        return MarketSituation.LOW_IV_RANGEBOUND, 0

    # ── Breakout scenarios ───────────────────────────────────────────────────
    if is_breakout:
        if high_iv:
            # Breakout into elevated IV: IC is dangerous (one side will breach),
            # debit spread is overpriced. Best to stay flat.
            return MarketSituation.VOLATILITY_SPIKE, 0

        # Low IV + breakout: score directional confirmation signals
        confirmed = 0
        if is_breakout_above:
            if spy_bullish is True:       confirmed += 1
            if daily_trend == "BULLISH":  confirmed += 1
            if iv_skew >= -0.05:          confirmed += 1   # calls not overpriced vs puts
        else:  # breakout_below
            if spy_bullish is False:      confirmed += 1
            if daily_trend == "BEARISH":  confirmed += 1
            if iv_skew <= 0.05:           confirmed += 1   # puts not wildly expensive

        if confirmed == 3:
            return MarketSituation.DIRECTIONAL_FULL, confirmed
        if confirmed >= 1:
            return MarketSituation.STRADDLE_PLAY, confirmed

    return MarketSituation.NO_SETUP, 0


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

def process_symbol_options(symbol: str, spy_bullish: bool | None,
                           open_positions_count: int, max_open_positions: int,
                           regime: RegimeResult | None = None) -> bool:
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

    # Minimum underlying price — cheap stocks have illiquid options and wide spreads
    if current_price < MIN_UNDERLYING_PRICE:
        log.info(f"{symbol} | Price ${current_price:.2f} < ${MIN_UNDERLYING_PRICE:.0f} minimum — skipping (illiquid options)")
        return False

    # Entry time cutoff — backtest win rate after 12:30 PM drops to 7.5% (3/40 trades)
    cutoff = time(IC_MAX_ENTRY_HOUR, IC_MAX_ENTRY_MINUTE)
    if now_et.time() >= cutoff:
        log.info(f"{symbol} | After {cutoff.strftime('%H:%M')} ET entry cutoff — skipping (backtest win rate 7.5% after this time)")
        return False

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

    # Minimum breakout strength — weak breakouts (<0.5% through OR) have 15% win rate in
    # backtest vs 52% for breakouts >1%. Only applies to directional strategies; ICs are
    # range-bound entries and are unaffected.
    if is_breakout_above:
        strength = (current_price - or_high) / or_high
        if strength < MIN_BREAKOUT_STRENGTH_PCT:
            log.info(
                f"{symbol} | Breakout too weak ({strength:.2%} < {MIN_BREAKOUT_STRENGTH_PCT:.1%}) "
                "— skipping directional entry"
            )
            is_breakout_above = False
    if is_breakout_below:
        strength = (or_low - current_price) / or_low
        if strength < MIN_BREAKOUT_STRENGTH_PCT:
            log.info(
                f"{symbol} | Breakdown too weak ({strength:.2%} < {MIN_BREAKOUT_STRENGTH_PCT:.1%}) "
                "— skipping directional entry"
            )
            is_breakout_below = False

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
        is_0dte = nearest_expiry == today_str
        chain = ticker.option_chain(nearest_expiry)
        calls = chain.calls
        puts = chain.puts
    except Exception as e:
        log.error(f"{symbol} | Failed to fetch option chain: {e}")
        return False

    if is_0dte:
        log.info(f"{symbol} | 0DTE detected (expiry today: {nearest_expiry})")

    if calls.empty or puts.empty:
        log.warning(f"{symbol} | Missing Call or Put contracts for expiry {nearest_expiry}")
        return False

    # Enrich chain with Newton-Raphson IV (more accurate than yfinance's stale column)
    T_years = (date.fromisoformat(nearest_expiry) - date.today()).days / 365.0
    T_years = max(T_years, 1 / (252 * 6.5))   # floor at one 5-min bar
    calls, puts = enrich_chain(calls, puts, current_price, T_years)

    # Sort strikes by closeness to current price to find ATM contracts
    calls_sorted = calls.iloc[(calls['strike'] - current_price).abs().argsort()]
    puts_sorted  = puts.iloc[(puts['strike']  - current_price).abs().argsort()]

    atm_call = calls_sorted.iloc[0]
    atm_put  = puts_sorted.iloc[0]

    # Use NR-computed IV; fall back to yfinance if NR column missing
    avg_atm_iv = (float(atm_call.get('nr_iv', atm_call['impliedVolatility'])) +
                  float(atm_put.get('nr_iv',  atm_put['impliedVolatility']))) / 2

    # IV Rank (0–100): where current IV sits in the symbol's 1-year range
    iv_rank = compute_iv_rank(symbol, avg_atm_iv)

    # IV Skew: OTM put IV - OTM call IV
    # Positive = bearish skew (puts expensive) | Negative = bullish skew (calls expensive)
    iv_skew = compute_iv_skew(calls, puts, current_price)

    # Regime-aware adjustments (from QuantTrading regime_detector.py)
    # Fetch intraday regime from today's 5-min bars
    intraday_bars = get_today_bars(symbol)
    intraday_regime = detect_regime(intraday_bars) if intraday_bars is not None and len(intraday_bars) >= 30 else None

    # Fetch daily regime and trend filter
    daily_bars = get_daily_bars(symbol)
    stock_daily_trend = daily_trend(daily_bars) if daily_bars is not None else "NEUTRAL"

    regime_label = intraday_regime.regime if intraday_regime else (regime.regime if regime else "NORMAL")

    log.info(f"{symbol} | ATM Call: {atm_call['contractSymbol']} (Strike: {atm_call['strike']} | Ask: {atm_call['ask']})")
    log.info(f"{symbol} | ATM Put: {atm_put['contractSymbol']} (Strike: {atm_put['strike']} | Ask: {atm_put['ask']})")
    log.info(f"{symbol} | ATM IV: {avg_atm_iv:.2%} | IV Rank: {iv_rank:.0f}/100 | Skew: {iv_skew:+.3f}")
    log.info(f"{symbol} | Regime: {regime_label} | Daily trend: {stock_daily_trend}")

    # CRISIS regime: skip new entries (QuantTrading: "crisis always wins")
    if regime_label == "CRISIS":
        log.warning(f"{symbol} | CRISIS regime detected — skipping new options entry")
        return False

    strategy = None
    legs = []
    qty = 0
    limit_price = 0.05  # positive = debit paid, negative = credit received

    # IV rank threshold: adjusted for regime (RANGING→aggressive, TRENDING→conservative)
    # then compressed 25% on 0DTE because IV is structurally elevated on expiry day.
    if regime_label == "RANGING":
        iv_rank_threshold = 50.0
    elif regime_label == "TRENDING":
        iv_rank_threshold = 80.0
    else:
        iv_rank_threshold = 70.0
    if is_0dte:
        iv_rank_threshold *= 0.75
        log.info(f"{symbol} | 0DTE IV rank threshold → {iv_rank_threshold:.1f}")

    # ── Classify the market situation ────────────────────────────────────────
    situation, confirmation_score = _classify_situation(
        is_breakout_above, is_breakout_below, is_range_bound,
        iv_rank, iv_rank_threshold, spy_bullish, stock_daily_trend, iv_skew,
    )
    direction_arrow = "↑" if is_breakout_above else ("↓" if is_breakout_below else "–")
    log.info(
        f"{symbol} | Situation: {situation.value.upper()} "
        f"| IV Rank: {iv_rank:.0f}/{iv_rank_threshold:.0f} "
        f"| Breakout: {direction_arrow} "
        f"| Confirmation: {confirmation_score}/3"
    )

    # ── Skip situations ───────────────────────────────────────────────────────
    if situation == MarketSituation.VOLATILITY_SPIKE:
        log.info(
            f"{symbol} | VOLATILITY_SPIKE — price breaking out with IV rank {iv_rank:.0f} "
            "already elevated; IC risks breach, debit spread overpriced — skipping"
        )
        return False

    if situation == MarketSituation.LOW_IV_RANGEBOUND:
        log.info(
            f"{symbol} | LOW_IV_RANGEBOUND — range-bound but IV rank {iv_rank:.0f} too low "
            f"for profitable IC credit (threshold {iv_rank_threshold:.0f}) — skipping"
        )
        return False

    if situation == MarketSituation.NO_SETUP:
        log.info(f"{symbol} | NO_SETUP — no clean signal, no trade")
        return False

    # ── Iron Condor ───────────────────────────────────────────────────────────
    if situation == MarketSituation.PREMIUM_SELL:
        # Gate 1: DTE limit
        dte = (date.fromisoformat(nearest_expiry) - date.today()).days
        if dte > IC_MAX_DTE:
            log.info(
                f"{symbol} | IC rejected: DTE={dte} > IC_MAX_DTE={IC_MAX_DTE} "
                "(multi-day condors breach almost always)"
            )
            return False

        strategy = "Iron Condor"
        width = 5.0 if current_price > 100 else (2.0 if current_price > 50 else 1.0)

        # Gate 2: sigma-based strike placement
        holding_days = max(1, dte + 1)
        expected_move = current_price * avg_atm_iv * math.sqrt(holding_days / 252)
        min_call_strike = current_price + IC_SIGMA_MULTIPLE * expected_move
        max_put_strike  = current_price - IC_SIGMA_MULTIPLE * expected_move
        log.info(
            f"{symbol} | IC expected move: ${expected_move:.2f} over {holding_days}d "
            f"| min call strike: ${min_call_strike:.2f} | max put strike: ${max_put_strike:.2f}"
        )

        sc_target = max(or_high, min_call_strike)
        sc_candidates = calls[calls['strike'] >= sc_target]
        if sc_candidates.empty:
            sc_candidates = calls[calls['strike'] > current_price]
        if sc_candidates.empty:
            log.warning(f"{symbol} | IC rejected: no short call candidates above ${sc_target:.2f}")
            return False
        short_call = sc_candidates.iloc[(sc_candidates['strike'] - sc_target).abs().argsort()].iloc[0]

        lc_candidates = calls[calls['strike'] > short_call['strike']]
        if lc_candidates.empty:
            log.warning(f"{symbol} | IC rejected: no long call candidates")
            return False
        long_call = lc_candidates.iloc[(lc_candidates['strike'] - (short_call['strike'] + width)).abs().argsort()].iloc[0]

        sp_target = min(or_low, max_put_strike)
        sp_candidates = puts[puts['strike'] <= sp_target]
        if sp_candidates.empty:
            sp_candidates = puts[puts['strike'] < current_price]
        if sp_candidates.empty:
            log.warning(f"{symbol} | IC rejected: no short put candidates below ${sp_target:.2f}")
            return False
        short_put = sp_candidates.iloc[(sp_candidates['strike'] - sp_target).abs().argsort()].iloc[0]

        lp_candidates = puts[puts['strike'] < short_put['strike']]
        if lp_candidates.empty:
            log.warning(f"{symbol} | IC rejected: no long put candidates")
            return False
        long_put = lp_candidates.iloc[(lp_candidates['strike'] - (short_put['strike'] - width)).abs().argsort()].iloc[0]

        net_credit = (
            (float(short_call['bid']) - float(long_call['ask'])) +
            (float(short_put['bid'])  - float(long_put['ask']))
        )
        credit_ratio = net_credit / width

        # Gate 3: minimum credit-to-width ratio
        if net_credit <= 0 or credit_ratio < IC_MIN_CREDIT_RATIO:
            log.info(
                f"{symbol} | IC rejected: credit ${net_credit:.2f} = "
                f"{credit_ratio:.1%} of width ${width:.0f} "
                f"(need ≥{IC_MIN_CREDIT_RATIO:.0%}) — negative EV"
            )
            return False

        risk = max(0.05, width - net_credit)
        qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (risk * 100)))
        legs = [
            OptionLegRequest(symbol=short_call['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_call['contractSymbol'],  side=OrderSide.BUY,  ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=short_put['contractSymbol'],  side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=long_put['contractSymbol'],   side=OrderSide.BUY,  ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
        ]
        limit_price = -round(net_credit, 2)
        log.info(
            f"{symbol} | Iron Condor | "
            f"Strikes: put ${short_put['strike']:.2f}/${long_put['strike']:.2f} "
            f"call ${short_call['strike']:.2f}/${long_call['strike']:.2f} | "
            f"Credit: ${net_credit:.2f} ({credit_ratio:.0%} of width) | "
            f"Risk: ${risk:.2f} | Qty: {qty}"
        )

    # ── Bull Call Spread / Bear Put Spread (all 3 signals confirmed) ──────────
    elif situation == MarketSituation.DIRECTIONAL_FULL:
        if is_breakout_above:
            strategy = "Bull Call Spread"
            long_call = atm_call
            oc_candidates = calls[calls['strike'] > long_call['strike']]
            if oc_candidates.empty:
                log.warning(f"{symbol} | Bull Call Spread rejected: no short call candidates")
                return False
            short_call = oc_candidates.iloc[(oc_candidates['strike'] - (long_call['strike'] + or_range)).abs().argsort()].iloc[0]
            net_debit = max(0.05, float(long_call['ask']) - float(short_call['bid']))
            qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
            legs = [
                OptionLegRequest(symbol=long_call['contractSymbol'],  side=OrderSide.BUY,  ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                OptionLegRequest(symbol=short_call['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            ]
            limit_price = round(net_debit, 2)
            log.info(
                f"{symbol} | Bull Call Spread ↑ | Confirmation: {confirmation_score}/3 "
                f"(SPY={'✓' if spy_bullish else '✗'} Daily={stock_daily_trend} Skew={iv_skew:+.3f}) "
                f"| Debit: ${net_debit:.2f} | Qty: {qty}"
            )
        else:
            strategy = "Bear Put Spread"
            long_put = atm_put
            op_candidates = puts[puts['strike'] < long_put['strike']]
            if op_candidates.empty:
                log.warning(f"{symbol} | Bear Put Spread rejected: no short put candidates")
                return False
            short_put = op_candidates.iloc[(op_candidates['strike'] - (long_put['strike'] - or_range)).abs().argsort()].iloc[0]
            net_debit = max(0.05, float(long_put['ask']) - float(short_put['bid']))
            qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
            legs = [
                OptionLegRequest(symbol=long_put['contractSymbol'],  side=OrderSide.BUY,  ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
                OptionLegRequest(symbol=short_put['contractSymbol'], side=OrderSide.SELL, ratio_qty=1, position_intent=PositionIntent.SELL_TO_OPEN),
            ]
            limit_price = round(net_debit, 2)
            log.info(
                f"{symbol} | Bear Put Spread ↓ | Confirmation: {confirmation_score}/3 "
                f"(SPY={'✓' if spy_bullish is False else '✗'} Daily={stock_daily_trend} Skew={iv_skew:+.3f}) "
                f"| Debit: ${net_debit:.2f} | Qty: {qty}"
            )

    # ── Straddle (breakout with mixed directional signals) ────────────────────
    elif situation == MarketSituation.STRADDLE_PLAY:
        strategy = "Straddle"
        net_debit = float(atm_call['ask']) + float(atm_put['ask'])
        qty = max(1, int(ORB_OPTIONS_POSITION_SIZE / (net_debit * 100)))
        legs = [
            OptionLegRequest(symbol=atm_call['contractSymbol'], side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=atm_put['contractSymbol'],  side=OrderSide.BUY, ratio_qty=1, position_intent=PositionIntent.BUY_TO_OPEN),
        ]
        limit_price = round(net_debit, 2)
        log.info(
            f"{symbol} | Straddle {direction_arrow} | Confirmation: {confirmation_score}/3 "
            f"(move expected, direction mixed) | Debit: ${net_debit:.2f} | Qty: {qty}"
        )

    if strategy is None:
        log.info(f"{symbol} | No strategy built for situation={situation.value} — check chain availability")
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


def _parse_strike(option_symbol: str) -> float:
    """Extract strike price from OCC symbol e.g. INTC260605C00111000 → 111.0"""
    match = re.search(r"\d{6}[CP](\d{8})", option_symbol)
    return int(match.group(1)) / 1000.0 if match else 0.0


def _get_short_strikes(positions: list) -> tuple[float | None, float | None]:
    """
    Return (short_call_strike, short_put_strike) by parsing OCC symbols.
    Used by manage_existing_options_positions to set the correct stop levels
    for Iron Condors regardless of which day they were entered.
    """
    short_call = None
    short_put  = None
    for pos in positions:
        sym = pos.symbol
        match = re.search(r"\d{6}([CP])\d{8}", sym)
        if not match:
            continue
        opt_type = match.group(1)
        is_short = float(pos.qty) < 0
        strike   = _parse_strike(sym)
        if is_short and opt_type == "C":
            short_call = strike
        elif is_short and opt_type == "P":
            short_put = strike
    return short_call, short_put


def manage_existing_options_positions(underlying_symbol: str, positions: list, spy_bullish: bool | None) -> None:
    """Monitor underlying stock price levels to exit options strategies dynamically."""
    strategy = classify_holding(positions)

    bars = get_today_bars(underlying_symbol)
    if bars is None:
        return

    current_price = float(bars.iloc[-1]["close"])
    now_et = get_now_et()

    # Forced close at EOD
    if now_et.time() >= CLOSE_TIME:
        log.info(f"EOD CLOSE — closing all options legs for {underlying_symbol}")
        close_all_legs(positions)
        return

    # OR levels needed only for debit-spread and straddle stops/targets
    or_result = compute_opening_range(bars)
    or_high = or_low = or_range = None
    if or_result is not None:
        or_high, or_low, _ = or_result
        or_range = or_high - or_low

    profit_multiplier = ORB_PROFIT_MULTIPLIERS.get(underlying_symbol, ORB_PROFIT_MULTIPLIER)

    if strategy == "call_spread":
        if or_high is None:
            return
        stop_price   = or_low - ORB_STOP_BUFFER
        target_price = or_high + or_range * profit_multiplier
        if current_price <= stop_price:
            log.info(f"STOP LOSS — Closing Call Spread for {underlying_symbol} (price {current_price:.2f} <= {stop_price:.2f})")
            close_all_legs(positions)
        elif current_price >= target_price:
            log.info(f"TAKE PROFIT — Closing Call Spread for {underlying_symbol} (price {current_price:.2f} >= {target_price:.2f})")
            close_all_legs(positions)

    elif strategy == "put_spread":
        if or_high is None:
            return
        stop_price   = or_high + ORB_STOP_BUFFER
        target_price = or_low - or_range * profit_multiplier
        if current_price >= stop_price:
            log.info(f"STOP LOSS — Closing Put Spread for {underlying_symbol} (price {current_price:.2f} >= {stop_price:.2f})")
            close_all_legs(positions)
        elif current_price <= target_price:
            log.info(f"TAKE PROFIT — Closing Put Spread for {underlying_symbol} (price {current_price:.2f} <= {target_price:.2f})")
            close_all_legs(positions)

    elif strategy == "straddle":
        if or_high is None:
            return
        target_high = or_high + or_range * profit_multiplier
        target_low  = or_low  - or_range * profit_multiplier
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
        # P&L-based exits: profit target + credit-multiple stop
        try:
            net_credit_received = (
                sum(abs(float(p.cost_basis)) for p in positions if float(p.qty) < 0) -
                sum(abs(float(p.cost_basis)) for p in positions if float(p.qty) > 0)
            )
            total_pnl = sum(float(p.unrealized_pl) for p in positions)
        except Exception:
            net_credit_received = 0.0
            total_pnl = 0.0

        if net_credit_received > 0:
            if total_pnl >= IC_PROFIT_TARGET_PCT * net_credit_received:
                log.info(
                    f"TAKE PROFIT — Closing Iron Condor for {underlying_symbol} "
                    f"(P&L ${total_pnl:.2f} >= {IC_PROFIT_TARGET_PCT:.0%} of credit ${net_credit_received:.2f})"
                )
                close_all_legs(positions)
                return
            if total_pnl <= -(IC_PNL_STOP_MULTIPLE * net_credit_received):
                log.info(
                    f"P&L STOP — Closing Iron Condor for {underlying_symbol} "
                    f"(P&L ${total_pnl:.2f} <= -{IC_PNL_STOP_MULTIPLE:.0f}× credit ${net_credit_received:.2f})"
                )
                close_all_legs(positions)
                return

        # Stop at the SHORT STRIKES — parsed directly from the OCC contract symbols.
        # Previously this used today's OR high/low which is wrong for multi-day condors.
        sc_strike, sp_strike = _get_short_strikes(positions)

        if sc_strike and sp_strike:
            log.info(
                f"{underlying_symbol} | Iron Condor stops: put ${sp_strike:.2f} / call ${sc_strike:.2f}"
                f" | Current: ${current_price:.2f}"
            )
            if current_price >= sc_strike:
                log.info(f"STOP LOSS — Closing Iron Condor for {underlying_symbol} "
                         f"(price {current_price:.2f} >= short call {sc_strike:.2f})")
                close_all_legs(positions)
            elif current_price <= sp_strike:
                log.info(f"STOP LOSS — Closing Iron Condor for {underlying_symbol} "
                         f"(price {current_price:.2f} <= short put {sp_strike:.2f})")
                close_all_legs(positions)
            else:
                log.info(f"{underlying_symbol} | Iron Condor holding "
                         f"(${current_price:.2f} inside ${sp_strike:.2f}–${sc_strike:.2f})")
        else:
            # Fallback: can't parse strikes — use OR if available
            if or_high and (current_price >= or_high or current_price <= or_low):
                log.info(f"STOP LOSS (fallback) — Closing Iron Condor for {underlying_symbol}")
                close_all_legs(positions)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_orb_options() -> None:
    now_et = get_now_et()
    log.info(f"--- ORB Options check at {now_et.strftime('%H:%M')} ET ---")

    # Safety net: close any stale options legs from a previous session.
    # Runs before the market-closed check so a missed EOD close is caught
    # the next morning even when the market is not yet open.
    if not DRY_RUN:
        _close_stale_options()

    if not DRY_RUN and not trading_client.get_clock().is_open:
        log.info("Market closed — skipping run")
        return

    if not DRY_RUN and not check_options_approval():
        log.error("Options Level 2 not approved — cannot submit multi-leg orders. Exiting.")
        return

    _block_new_entries = not DRY_RUN and not check_risk_limits()

    # Hard EOD sweep — close all options legs directly from positions list,
    # no bar data needed. Prevents overnight holds even if bar fetch fails.
    if not DRY_RUN and now_et.time() >= CLOSE_TIME:
        log.info("EOD — force-closing all options positions")
        try:
            all_pos = trading_client.get_all_positions()
            options_legs = [p for p in all_pos if p.asset_class == AssetClass.US_OPTION]
            if options_legs:
                close_all_legs(options_legs)
                log.info(f"EOD closed {len(options_legs)} options legs")
            else:
                log.info("EOD: no open options positions")
        except Exception as e:
            log.error(f"EOD options sweep failed: {e}")
        return

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

    # Re-fetch positions after management so stops that fired this run
    # don't inflate the budget count and block all new entries.
    try:
        refreshed = trading_client.get_all_positions()
        refreshed_opts = [p for p in refreshed if p.asset_class == AssetClass.US_OPTION]
        open_positions_count = len({get_underlying_symbol(p.symbol) for p in refreshed_opts})
        log.info(f"Budget: {open_positions_count} open positions after management")
    except Exception:
        open_positions_count = len(held_symbols)   # fall back to pre-management count

    max_open_positions = max(1, int(MAX_OPTIONS_INVESTMENT / ORB_OPTIONS_POSITION_SIZE))

    # 2. Screen for new options entry opportunities
    if _block_new_entries:
        log.warning("Risk limits active — skipping new entries, existing positions still managed")
        return

    # Monday skip — 14.3% backtest win rate vs 26–31% Tue–Thu
    if SKIP_MONDAY_ENTRIES and now_et.weekday() == 0:
        log.info("Monday — skipping all new entries (backtest win rate 14.3%)")
        return

    active_symbols = get_active_symbols()
    if not active_symbols:
        log.warning("No active symbols found from screener. Skipping new entries.")
    else:
        blocklist = set(ORB_OPTIONS_BLOCKLIST)
        for symbol in active_symbols:
            if symbol in held_symbols:
                continue
            if symbol in blocklist:
                log.info(f"{symbol} | In options blocklist — skipping (backtest net loser)")
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

    log.info("ORB Options Bot starting")
    log.info(
        f"OR: first {ORB_RANGE_BARS} bars | ${ORB_OPTIONS_POSITION_SIZE}/trade"
        f" | Entry window: 10:00–{IC_MAX_ENTRY_HOUR}:{IC_MAX_ENTRY_MINUTE:02d} ET"
        f" | EOD close: {ORB_CLOSE_HOUR}:{ORB_CLOSE_MINUTE:02d} ET"
    )
    log.info(
        f"Filters: price≥${MIN_UNDERLYING_PRICE:.0f} | credit≥{IC_MIN_CREDIT_RATIO:.0%}"
        f" | {IC_SIGMA_MULTIPLE}σ strikes | min breakout {MIN_BREAKOUT_STRENGTH_PCT:.1%}"
        f" | daily loss limit {DAILY_LOSS_LIMIT_PCT:.0%}"
    )

    run_orb_options()
    log.info("Run complete")
