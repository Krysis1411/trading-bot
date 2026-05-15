"""
Run a NautilusTrader backtest of the Opening Range Breakout (ORB) strategy.

NOTE: yfinance only provides 5-min data for the last 60 days, so each backtest
covers roughly 2 months (~40 trading days). Use a paid data source (e.g.
Databento, Polygon) for longer history.

Usage
-----
    python -m backtest.run_orb_backtest            # default: AAPL
    python -m backtest.run_orb_backtest MSFT
    python -m backtest.run_orb_backtest AAPL MSFT SPY QQQ NVDA
"""
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from backtest.fetch_data import fetch_bars, load_bars_df
from config import (
    BACKTEST_STARTING_BALANCE,
    ORB_CLOSE_HOUR,
    ORB_CLOSE_MINUTE,
    ORB_MIN_OR_PCT,
    ORB_PROFIT_MULTIPLIER,
    ORB_PROFIT_MULTIPLIERS,
    ORB_RANGE_BARS,
    ORB_STOP_BUFFER,
    ORB_TRADE_QUANTITY,
    ORB_VOLUME_FACTOR,
)
from strategies.orb import ORBConfig, ORBStrategy

DATA_DIR = Path(__file__).parent / "data"


def run_orb_backtest(symbol: str = "AAPL") -> None:
    print(f"\n{'='*60}")
    print(f"  ORB Backtest: {symbol}")
    print(f"  Range: first {ORB_RANGE_BARS} bars (30 min)  |  "
          f"Target: {ORB_PROFIT_MULTIPLIER}× range  |  "
          f"EOD close: {ORB_CLOSE_HOUR}:{ORB_CLOSE_MINUTE:02d} ET")
    print(f"{'='*60}\n")

    instrument = TestInstrumentProvider.equity(symbol=symbol, venue="XNAS")

    # yfinance 5-min data is limited to 60 days
    print("Downloading 5-min data (60-day yfinance limit)...")
    csv_path = fetch_bars(symbol, "5m", period="60d", output_dir=str(DATA_DIR))
    df = load_bars_df(csv_path)

    # Drop pre-market / after-hours rows so the OR is always the true 9:30 open
    # yfinance 5m data includes extended hours — filter to regular session only
    import zoneinfo
    ET = zoneinfo.ZoneInfo("America/New_York")
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert(ET)
    df_et = df_et.between_time("09:30", "16:00")
    df_et.index = df_et.index.tz_convert("UTC")

    bar_type = BarType.from_str(f"{instrument.id}-5-MINUTE-LAST-EXTERNAL")
    bars = BarDataWrangler(bar_type, instrument).process(df_et)
    print(f"  {len(bars)} bars loaded ({df_et.index[0].date()} → {df_et.index[-1].date()})")

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),
            risk_engine=RiskEngineConfig(bypass=True),
        )
    )

    XNAS = Venue("XNAS")
    engine.add_venue(
        venue=XNAS,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=USD,
        starting_balances=[Money(BACKTEST_STARTING_BALANCE, USD)],
    )
    engine.add_instrument(instrument)
    engine.add_data(bars)

    profit_multiplier = ORB_PROFIT_MULTIPLIERS.get(symbol, ORB_PROFIT_MULTIPLIER)
    strategy = ORBStrategy(
        config=ORBConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            trade_size=Decimal(str(ORB_TRADE_QUANTITY)),
            orb_range_bars=ORB_RANGE_BARS,
            profit_multiplier=profit_multiplier,
            volume_factor=ORB_VOLUME_FACTOR,
            stop_buffer=ORB_STOP_BUFFER,
            close_hour=ORB_CLOSE_HOUR,
            close_minute=ORB_CLOSE_MINUTE,
            min_or_pct=ORB_MIN_OR_PCT,
        )
    )
    engine.add_strategy(strategy)

    print("Running backtest...")
    engine.run()

    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 120):
        print("\n=== ACCOUNT ===")
        print(engine.trader.generate_account_report(XNAS))
        print("\n=== ORDER FILLS ===")
        print(engine.trader.generate_order_fills_report())
        print("\n=== POSITIONS ===")
        print(engine.trader.generate_positions_report())

    engine.reset()
    engine.dispose()


if __name__ == "__main__":
    symbols = sys.argv[1:] or ["AAPL"]
    for sym in symbols:
        run_orb_backtest(sym)
