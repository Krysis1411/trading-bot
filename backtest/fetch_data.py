"""
Download OHLCV data and save as CSV files ready for NautilusTrader's BarDataWrangler.

Data source priority
--------------------
1. Alpaca Market Data API  — uses the same credentials as the live bot, no rate
                             limits for paper/live accounts, reliable intraday data
2. yfinance                — free fallback, but Yahoo Finance rate-limits heavily;
                             if you hit "Too Many Requests" wait 30+ minutes

Set ALPACA_API_KEY and ALPACA_SECRET_KEY in a local .env file (same keys you
added to GitHub Actions secrets) to use Alpaca as the primary source.
"""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Alpaca source
# ---------------------------------------------------------------------------

_ALPACA_TIMEFRAME_MAP = {
    "5m":  "5Min",
    "1h":  "1Hour",
    "1d":  "1Day",
    "5Min": "5Min",
    "1Hour": "1Hour",
    "1Day": "1Day",
}

_ALPACA_LOOKBACK_DAYS = {
    "5m":  60,
    "5Min": 60,
    "1h":  730,
    "1Hour": 730,
    "1d":  3650,
    "1Day": 3650,
}


def _fetch_alpaca(symbol: str, interval: str, output_dir: str) -> str:
    try:
        import alpaca_trade_api as tradeapi
    except ImportError:
        raise EnvironmentError("alpaca-trade-api not installed — falling back to yfinance")

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")

    api = tradeapi.REST(api_key, secret_key, "https://paper-api.alpaca.markets", api_version="v2")

    tf = _ALPACA_TIMEFRAME_MAP.get(interval, interval)
    days = _ALPACA_LOOKBACK_DAYS.get(interval, 60)
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    bars = api.get_bars(symbol, tf, start=start, limit=10000, adjustment="raw").df

    if bars.empty:
        raise ValueError(f"Alpaca returned no data for {symbol} ({tf})")

    bars = bars.rename(columns={"open": "open", "high": "high", "low": "low",
                                "close": "close", "volume": "volume"})
    bars = bars[["open", "high", "low", "close", "volume"]]

    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")

    bars.index.name = "timestamp"
    bars = bars.sort_index()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_{interval}.csv")
    bars.to_csv(path)
    print(f"  [Alpaca] {symbol} {interval}: {len(bars)} bars → {path}")
    return path


# ---------------------------------------------------------------------------
# yfinance fallback
# ---------------------------------------------------------------------------

_YF_MAX_PERIOD = {
    "5m": "60d",
    "1h": "2y",
    "1d": "10y",
}


def _fetch_yfinance(symbol: str, interval: str, output_dir: str) -> str:
    import yfinance as yf

    period = _YF_MAX_PERIOD.get(interval, "5y")
    df: pd.DataFrame = yf.Ticker(symbol).history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol} ({interval})")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df = df.sort_index()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_{interval}.csv")
    df.to_csv(path)
    print(f"  [yfinance] {symbol} {interval}: {len(df)} bars → {path}")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_bars(
    symbol: str,
    interval: str = "1d",
    period: str | None = None,   # kept for backwards compat, ignored (source controls lookback)
    output_dir: str = "backtest/data",
) -> str:
    """
    Download OHLCV bars for *symbol* and save a UTC CSV for BarDataWrangler.

    Tries Alpaca first (requires .env with ALPACA_API_KEY / ALPACA_SECRET_KEY),
    falls back to yfinance if credentials are missing.
    """
    try:
        return _fetch_alpaca(symbol, interval, output_dir)
    except EnvironmentError:
        print(f"  Alpaca credentials not found — falling back to yfinance for {symbol}")
        return _fetch_yfinance(symbol, interval, output_dir)
    except Exception as e:
        print(f"  Alpaca failed ({e}) — falling back to yfinance for {symbol}")
        return _fetch_yfinance(symbol, interval, output_dir)


def load_bars_df(csv_path: str) -> pd.DataFrame:
    """Read a CSV written by fetch_bars() back into a DataFrame."""
    df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()


if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] or ["AAPL", "MSFT", "SPY"]
    for sym in symbols:
        fetch_bars(sym, "5m", output_dir="backtest/data")
