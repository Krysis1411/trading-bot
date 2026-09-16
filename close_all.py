"""
close_all.py — Immediately close ALL open Alpaca positions at market price.
Run once to flatten everything, then start fresh with orb_bot.py.

Also the script eod-close.yml runs daily as the last-resort EOD safety net
for both US bots. That's exactly why every step here must be defensive: a
CIFR options position was found to have survived a full missed EOD close
AND this safety-close workflow, sitting open ~27 hours overnight (Aug 25
11:25 AM ET -> Aug 26 2:54 PM ET) -- confirmed via Alpaca's fill-activity
log. The workflow run for that day shows as "success" in GitHub Actions,
so whatever went wrong was swallowed rather than surfaced.

The most likely culprit, found on inspection: `float(pos.current_price)` /
`float(pos.unrealized_pl)` were read OUTSIDE the try/except, before the
actual close attempt. Alpaca can return None for either field on a thinly
traded options contract with no recent quote -- CIFR's contracts were
priced under $1, plausibly illiquid enough to hit this. `float(None)`
raises TypeError there, which is NOT caught (the try block starts after),
so it would abort the whole per-position loop -- leaving every position
after the crash point unclosed, on the one script whose entire job is to
guarantee nothing is left open. Fixed by moving all field access inside
the try/except (one bad position no longer blocks the rest) and by
verifying afterward that everything actually closed, with one retry pass.
"""
import logging
import os
import time

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


def _close_one(pos) -> bool:
    """Attempt to close a single position. Returns True if the close order
    was accepted. All field access is inside the try -- a bad/missing quote
    on this one position must never stop the rest of the batch."""
    try:
        symbol = pos.symbol
        qty = abs(int(float(pos.qty)))
        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        unrealized = pos.unrealized_pl
        current = pos.current_price
        client.submit_order(MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=close_side,
            time_in_force=TimeInForce.DAY,
        ))
        pl_str = f"${float(unrealized):+.2f}" if unrealized is not None else "unknown (no quote)"
        px_str = f"${float(current):.2f}" if current is not None else "unknown"
        log.info(
            f"CLOSED {'LONG' if close_side == OrderSide.SELL else 'SHORT'}"
            f" {qty} × {symbol} @ ~{px_str} | Unrealized P&L: {pl_str}"
        )
        return True
    except Exception as e:
        log.error(f"Failed to close {getattr(pos, 'symbol', '?')}: {e}")
        return False


def close_all_with_verification(max_attempts: int = 3, wait_seconds: float = 5.0) -> bool:
    """Close every open position, then re-check and retry until nothing is
    left open (or attempts run out). Returns True iff the account ends flat.
    This is the safety-net script -- silently returning after one pass
    without confirming the account is actually flat is exactly how a
    position slips through undetected."""
    for attempt in range(1, max_attempts + 1):
        positions = client.get_all_positions()
        if not positions:
            if attempt == 1:
                log.info("No open positions found — nothing to close.")
            else:
                log.info(f"Verified flat after {attempt - 1} attempt(s).")
            return True

        log.info(f"Attempt {attempt}/{max_attempts}: {len(positions)} open position(s). Closing at market...")
        for pos in positions:
            _close_one(pos)

        if attempt < max_attempts:
            time.sleep(wait_seconds)  # let orders fill before re-checking

    remaining = client.get_all_positions()
    if remaining:
        symbols = ", ".join(p.symbol for p in remaining)
        log.error(
            f"STILL OPEN after {max_attempts} attempts: {symbols} — "
            "manual intervention needed, this is the last-resort safety net"
        )
        return False
    return True


if __name__ == "__main__":
    ok = close_all_with_verification()
    log.info("Done — account confirmed flat." if ok else "Done — NOT flat, see errors above.")
    raise SystemExit(0 if ok else 1)
