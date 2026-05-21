import logging
from openbb import obb

from config import ORB_SCREENER_LIMIT, ORB_SCREENER_PROVIDER

log = logging.getLogger(__name__)

def get_active_symbols() -> list[str]:
    """
    Fetch the most active stocks for the day using OpenBB.
    Returns a list of ticker symbols.
    """
    try:
        log.info(f"Fetching top {ORB_SCREENER_LIMIT} active stocks via OpenBB ({ORB_SCREENER_PROVIDER})...")
        res = obb.equity.discovery.active(provider=ORB_SCREENER_PROVIDER)
        df = res.to_df()
        
        if df.empty:
            log.warning("OpenBB returned empty dataframe for active stocks.")
            return []
            
        # Get top N symbols
        symbols = df['symbol'].head(ORB_SCREENER_LIMIT).tolist()
        log.info(f"OpenBB Screener found: {', '.join(symbols)}")
        return symbols
    except Exception as e:
        log.error(f"Failed to fetch active stocks from OpenBB: {e}")
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = get_active_symbols()
    print("Symbols:", symbols)
