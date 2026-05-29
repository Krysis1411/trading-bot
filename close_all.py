"""
close_all.py — Immediately close ALL open Alpaca positions at market price.
Run once to flatten everything, then start fresh with orb_bot.py.
"""
import logging
import os

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, PositionSide

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

key    = os.environ.get("ALPACA_API_KEY")
secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")

if not key or not secret:
    raise EnvironmentError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in environment")

client = TradingClient(key, secret, paper=True)
positions = client.get_all_positions()

if not positions:
    log.info("No open positions found — nothing to close.")
else:
    log.info(f"Found {len(positions)} open position(s). Closing all at market price...")
    for pos in positions:
        symbol       = pos.symbol
        qty          = abs(int(float(pos.qty)))
        unrealized   = float(pos.unrealized_pl)
        current      = float(pos.current_price)
        close_side   = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        try:
            client.submit_order(MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=close_side,
                time_in_force=TimeInForce.DAY,
            ))
            log.info(
                f"CLOSED {'LONG' if close_side == OrderSide.SELL else 'SHORT'}"
                f" {qty} × {symbol} @ ~${current:.2f}"
                f" | Unrealized P&L: ${unrealized:+.2f}"
            )
        except Exception as e:
            log.error(f"Failed to close {symbol}: {e}")

    log.info("Done. All positions submitted for closure.")
