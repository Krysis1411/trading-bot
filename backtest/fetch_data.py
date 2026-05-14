"""Download OHLCV data via yfinance and save as CSV files ready for BarDataWrangler."""
import os

import pandas as pd
import yfinance as yf


# yfinance interval → how far back we can go
_MAX_PERIOD = {
    "1h": "2y",   # yfinance hourly limit
    "1d": "10y",
}


def fetch_bars(
    symbol: str,
    interval: str = "1d",
    period: str | None = None,
    output_dir: str = "backtest/data",
) -> str:
    """
    Download OHLCV bars for *symbol* and write a CSV the BarDataWrangler can read.

    Expected output columns: open, high, low, close, volume
    Expected index name    : timestamp  (UTC, tz-aware)

    Returns
    -------
    str
        Path to the written CSV file.
    """
    if period is None:
        period = _MAX_PERIOD.get(interval, "5y")

    ticker = yf.Ticker(symbol)
    df: pd.DataFrame = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} ({interval}, {period})")

    # Normalise column names
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df[["open", "high", "low", "close", "volume"]]

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df = df.sort_index()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_{interval}.csv")
    df.to_csv(path)
    print(f"  Saved {len(df)} {interval} bars → {path}")
    return path


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
        fetch_bars(sym, "1h", output_dir="backtest/data")
        fetch_bars(sym, "1d", output_dir="backtest/data")
