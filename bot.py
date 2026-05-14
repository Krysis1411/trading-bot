"""
Live trading bot — RSI + 200-day MA strategy via Alpaca paper trading.

Run directly:  python bot.py
Scheduled via: .github/workflows/run-bot.yml  (every 5 min during market hours)
"""
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi

from config import (
    MA_TREND_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    STOP_LOSS_PCT,
    SYMBOLS,
    TRADE_QUANTITY,
)

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


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> float:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss
    return round(float(100 - (100 / (1 + rs.iloc[-1]))), 2)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_market_data(symbol: str) -> tuple[float, float, float] | tuple[None, None, None]:
    """Return (price, rsi, ma_200) or (None, None, None) on failure."""
    try:
        now = datetime.now(timezone.utc)
        hourly_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        daily_start = (now - timedelta(days=300)).strftime("%Y-%m-%d")

        hourly = api.get_bars(symbol, "1Hour", start=hourly_start, limit=RSI_PERIOD + 20).df
        daily = api.get_bars(symbol, "1Day", start=daily_start, limit=MA_TREND_PERIOD).df

        if len(hourly) < RSI_PERIOD:
            log.warning(f"{symbol}: not enough hourly bars ({len(hourly)})")
            return None, None, None
        if len(daily) < MA_TREND_PERIOD:
            log.warning(f"{symbol}: not enough daily bars ({len(daily)})")
            return None, None, None

        price = round(float(hourly["close"].iloc[-1]), 2)
        rsi = calculate_rsi(hourly["close"])
        ma_200 = round(float(daily["close"].mean()), 2)
        return price, rsi, ma_200

    except Exception as e:
        log.error(f"{symbol}: data fetch failed — {e}")
        return None, None, None


def get_position(symbol: str):
    try:
        return api.get_position(symbol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def run_strategy() -> None:
    if not api.get_clock().is_open:
        log.info("Market closed — skipping run")
        return

    log.info("--- Strategy check ---")

    for symbol in SYMBOLS:
        try:
            price, rsi, ma_200 = get_market_data(symbol)
            if price is None:
                continue

            in_uptrend = price > ma_200
            position = get_position(symbol)
            qty = int(position.qty) if position else 0

            log.info(
                f"{symbol} | Price: {price} | RSI: {rsi} | "
                f"200MA: {ma_200} | Trend: {'UP' if in_uptrend else 'DOWN'} | Pos: {qty}"
            )

            # --- Exit ---
            if position:
                pnl_pct = float(position.unrealized_plpc) * 100

                if pnl_pct <= -STOP_LOSS_PCT:
                    api.submit_order(symbol=symbol, qty=qty, side="sell",
                                     type="market", time_in_force="day")
                    log.info(f"STOP LOSS — SELL {qty} {symbol} | P&L: {pnl_pct:.2f}%")
                    continue

                if rsi > RSI_OVERBOUGHT:
                    api.submit_order(symbol=symbol, qty=qty, side="sell",
                                     type="market", time_in_force="day")
                    log.info(f"TAKE PROFIT — SELL {qty} {symbol} | RSI: {rsi}")
                    continue

                log.info(f"{symbol} | Holding | P&L: {pnl_pct:.2f}%")
                continue

            # --- Entry ---
            if rsi < RSI_OVERSOLD and in_uptrend:
                api.submit_order(symbol=symbol, qty=TRADE_QUANTITY, side="buy",
                                 type="market", time_in_force="day")
                log.info(f"BUY {TRADE_QUANTITY} {symbol} | RSI: {rsi} | Above 200MA")
            elif rsi < RSI_OVERSOLD:
                log.info(f"{symbol} | RSI oversold but downtrend — skipping")
            else:
                log.info(f"{symbol} | No signal")

        except Exception as e:
            log.error(f"{symbol}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info(f"Symbols  : {', '.join(SYMBOLS)}")
    log.info(f"Strategy : RSI({RSI_PERIOD}) + {MA_TREND_PERIOD}-day MA trend filter")
    log.info(f"Buy      : RSI < {RSI_OVERSOLD} AND price above {MA_TREND_PERIOD}-day MA")
    log.info(f"Sell     : RSI > {RSI_OVERBOUGHT} OR P&L < -{STOP_LOSS_PCT}%")
    run_strategy()
    log.info("Run complete")
