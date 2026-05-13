import os
import logging
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

from config import SYMBOLS, SHORT_WINDOW, LONG_WINDOW, TRADE_QUANTITY

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


def get_moving_averages(symbol):
    bars = api.get_bars(symbol, "1Day", limit=LONG_WINDOW + 1).df
    if len(bars) < LONG_WINDOW:
        logger.warning(f"{symbol}: not enough bar data ({len(bars)} bars)")
        return None, None
    prices = bars["close"]
    return prices.tail(SHORT_WINDOW).mean(), prices.tail(LONG_WINDOW).mean()


def get_position_qty(symbol):
    try:
        return int(api.get_position(symbol).qty)
    except Exception:
        return 0


def run_strategy():
    if not is_market_open():
        logger.info("Market closed — skipping run")
        return

    logger.info("--- Strategy check ---")

    for symbol in SYMBOLS:
        try:
            short_ma, long_ma = get_moving_averages(symbol)
            if short_ma is None:
                continue

            qty = get_position_qty(symbol)
            logger.info(
                f"{symbol} | Short MA: {short_ma:.2f} | Long MA: {long_ma:.2f} | Position: {qty} shares"
            )

            if short_ma > long_ma and qty == 0:
                api.submit_order(
                    symbol=symbol,
                    qty=TRADE_QUANTITY,
                    side="buy",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"BUY  {TRADE_QUANTITY} share(s) of {symbol}")

            elif short_ma < long_ma and qty > 0:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="day",
                )
                logger.info(f"SELL {qty} share(s) of {symbol}")

        except Exception as e:
            logger.error(f"{symbol}: {e}")


if __name__ == "__main__":
    logger.info(f"Symbols  : {', '.join(SYMBOLS)}")
    logger.info(f"Strategy : {SHORT_WINDOW}-day / {LONG_WINDOW}-day MA crossover")
    run_strategy()
    logger.info("Run complete")
