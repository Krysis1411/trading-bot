import os
import logging
import pandas as pd
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from config import (
    SYMBOLS,
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
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


def get_rsi(symbol):
    bars = api.get_bars(symbol, "1Hour", limit=RSI_PERIOD + 10).df
    if len(bars) < RSI_PERIOD:
        logger.warning(f"{symbol}: not enough bar data ({len(bars)} bars)")
        return None
    rsi = calculate_rsi(bars["close"])
    return round(rsi.iloc[-1], 2)


def get_position_qty(symbol):
    try:
        return int(api.get_position(symbol).qty)
    except Exception:
        return 0


def run_strategy():
    if not is_market_open():
        logger.info("Market closed — skipping run")
        return

    logger.info("--- RSI Strategy check ---")

    for symbol in SYMBOLS:
        try:
            rsi = get_rsi(symbol)
            if rsi is None:
                continue

            qty = get_position_qty(symbol)
            logger.info(f"{symbol} | RSI: {rsi} | Position: {qty} shares")

            if rsi < RSI_OVERSOLD and qty == 0:
                api.submit_order(
                    symbol=symbol,
                    qty=TRADE_QUANTITY,
                    side="buy",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"BUY  {TRADE_QUANTITY} share(s) of {symbol} — RSI oversold at {rsi}")

            elif rsi > RSI_OVERBOUGHT and qty > 0:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"SELL {qty} share(s) of {symbol} — RSI overbought at {rsi}")

            else:
                logger.info(f"{symbol} | No signal (RSI: {rsi})")

        except Exception as e:
            logger.error(f"{symbol}: {e}")


if __name__ == "__main__":
    logger.info(f"Symbols  : {', '.join(SYMBOLS)}")
    logger.info(f"Strategy : RSI({RSI_PERIOD}) — Buy < {RSI_OVERSOLD} | Sell > {RSI_OVERBOUGHT}")
    run_strategy()
    logger.info("Run complete")
