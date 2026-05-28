"""
close_all.py — Immediately close ALL open Alpaca positions at market price.
Run once to flatten everything, then start fresh with orb_bot.py.
"""
import logging
import os

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")
if not secret:
    raise EnvironmentError("Neither ALPACA_SECRET_KEY nor ALPACA_API_SECRET found in environment.")

api = tradeapi.REST(
    os.environ["ALPACA_API_KEY"],
    secret,
    "https://paper-api.alpaca.markets",
    api_version="v2",
)

positions = api.list_positions()

if not positions:
    log.info("No open positions found — nothing to close.")
else:
    log.info(f"Found {len(positions)} open position(s). Closing all at market price...")
    for pos in positions:
        symbol = pos.symbol
        qty = int(pos.qty)
        unrealized_pl = float(pos.unrealized_pl)
        current_price = float(pos.current_price)
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="market",
                time_in_force="day",
            )
            log.info(
                f"CLOSED {qty} × {symbol} @ ~${current_price:.2f}"
                f" | Unrealized P&L: ${unrealized_pl:+.2f}"
            )
        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")

    log.info("Done. All positions submitted for closure.")
