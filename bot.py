import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from config import (
    SYMBOLS,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    MA_TREND_PERIOD,
    STOP_LOSS_PCT,
    TRADE_QUANTITY,
)

load_dotenv()

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def is_market_open():
    return api.get_clock().is_open


def calculate_rsi(prices, period=RSI_PERIOD):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_market_data(symbol):
    """Returns (current_price, rsi, ma_200) or (None, None, None) on failure."""
    try:
        hourly_start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
        daily_start = (datetime.utcnow() - timedelta(days=300)).strftime("%Y-%m-%d")
        hourly_bars = api.get_bars(symbol, "1Hour", start=hourly_start, limit=RSI_PERIOD + 20).df
        daily_bars = api.get_bars(symbol, "1Day", start=daily_start, limit=MA_TREND_PERIOD).df

        if len(hourly_bars) < RSI_PERIOD:
            logger.warning(f"{symbol}: not enough hourly bars ({len(hourly_bars)})")
            return None, None, None

        if len(daily_bars) < MA_TREND_PERIOD:
            logger.warning(f"{symbol}: not enough daily bars ({len(daily_bars)})")
            return None, None, None

        rsi = round(calculate_rsi(hourly_bars["close"]).iloc[-1], 2)
        ma_200 = round(daily_bars["close"].mean(), 2)
        current_price = round(hourly_bars["close"].iloc[-1], 2)

        return current_price, rsi, ma_200

    except Exception as e:
        logger.error(f"{symbol}: failed to get market data — {e}")
        return None, None, None


def get_position(symbol):
    try:
        return api.get_position(symbol)
    except Exception:
        return None


def run_strategy():
    if not is_market_open():
        logger.info("Market closed — skipping run")
        return

    logger.info("--- Strategy check ---")

    for symbol in SYMBOLS:
        try:
            current_price, rsi, ma_200 = get_market_data(symbol)
            if rsi is None:
                continue

            in_uptrend = current_price > ma_200
            position = get_position(symbol)
            qty = int(position.qty) if position else 0

            logger.info(
                f"{symbol} | Price: {current_price} | RSI: {rsi} | "
                f"200MA: {ma_200} | Trend: {'UP' if in_uptrend else 'DOWN'} | Position: {qty} shares"
            )

            # --- Exit logic ---
            if position:
                unrealized_pct = float(position.unrealized_plpc) * 100

                if unrealized_pct <= -STOP_LOSS_PCT:
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day",
                    )
                    logger.info(f"STOP LOSS — SELL {qty} {symbol} | P&L: {unrealized_pct:.2f}%")
                    continue

                if rsi > RSI_OVERBOUGHT:
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day",
                    )
                    logger.info(f"TAKE PROFIT — SELL {qty} {symbol} | RSI: {rsi}")
                    continue

                logger.info(f"{symbol} | Holding | P&L: {unrealized_pct:.2f}%")

            # --- Entry logic ---
            elif rsi < RSI_OVERSOLD and in_uptrend:
                api.submit_order(
                    symbol=symbol,
                    qty=TRADE_QUANTITY,
                    side="buy",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"BUY {TRADE_QUANTITY} {symbol} | RSI: {rsi} | Above 200MA: {in_uptrend}")

            elif rsi < RSI_OVERSOLD and not in_uptrend:
                logger.info(f"{symbol} | RSI oversold but DOWNTREND — skipping entry")

            else:
                logger.info(f"{symbol} | No signal")

        except Exception as e:
            logger.error(f"{symbol}: {e}")


if __name__ == "__main__":
    logger.info(f"Symbols  : {', '.join(SYMBOLS)}")
    logger.info(f"Strategy : RSI({RSI_PERIOD}) + 200-day MA trend filter")
    logger.info(f"Buy      : RSI < {RSI_OVERSOLD} AND price above 200MA")
    logger.info(f"Sell     : RSI > {RSI_OVERBOUGHT} OR P&L < -{STOP_LOSS_PCT}%")
    run_strategy()
    logger.info("Run complete")
