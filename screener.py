import logging
import yfinance as yf

from config import ORB_SCREENER_LIMIT, ORB_SYMBOLS

log = logging.getLogger(__name__)


def get_active_symbols() -> list[str]:
    """
    Fetch the most active stocks for the day via yfinance screener.
    Falls back to the static ORB_SYMBOLS list if the screener fails.
    """
    try:
        log.info(f"Fetching top {ORB_SCREENER_LIMIT} active stocks via yfinance screener...")
        screener = yf.Screener()
        screener.set_predefined_body("most_actives")
        result = screener.response
        quotes = result.get("quotes", [])
        if not quotes:
            raise ValueError("yfinance screener returned empty quotes")
        symbols = [q["symbol"] for q in quotes[:ORB_SCREENER_LIMIT]]
        log.info(f"Screener found: {', '.join(symbols)}")
        return symbols
    except Exception as e:
        log.warning(f"yfinance screener failed ({e}) — using static ORB_SYMBOLS fallback")
        log.info(f"Fallback symbols: {', '.join(ORB_SYMBOLS)}")
        return list(ORB_SYMBOLS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Symbols:", get_active_symbols())
