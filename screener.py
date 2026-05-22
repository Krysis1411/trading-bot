import logging
import requests

from config import ORB_SCREENER_LIMIT, ORB_SYMBOLS

log = logging.getLogger(__name__)

_SCREENER_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"


def get_active_symbols() -> list[str]:
    """
    Fetch the most active stocks via Yahoo Finance screener.
    Falls back to the static ORB_SYMBOLS list if the request fails.
    """
    try:
        log.info(f"Fetching top {ORB_SCREENER_LIMIT} active stocks via Yahoo Finance screener...")
        resp = requests.get(
            _SCREENER_URL,
            params={"formatted": "false", "scrIds": "most_actives", "count": ORB_SCREENER_LIMIT},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        quotes = resp.json()["finance"]["result"][0]["quotes"]
        symbols = [q["symbol"] for q in quotes[:ORB_SCREENER_LIMIT]]
        log.info(f"Screener found: {', '.join(symbols)}")
        return symbols
    except Exception as e:
        log.warning(f"Screener request failed ({e}) — using static ORB_SYMBOLS fallback")
        log.info(f"Fallback symbols: {', '.join(ORB_SYMBOLS)}")
        return list(ORB_SYMBOLS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Symbols:", get_active_symbols())
