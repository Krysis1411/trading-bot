"""
Run a NautilusTrader backtest of the RSI + 200-day MA strategy.

Usage
-----
    # single symbol (default AAPL)
    python -m backtest.run_backtest

    # specific symbol
    python -m backtest.run_backtest MSFT

    # multiple symbols (sequential)
    python -m backtest.run_backtest AAPL MSFT SPY
"""
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Allow running from the repo root
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
    MA_TREND_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    STOP_LOSS_PCT,
    TRADE_QUANTITY,
    BACKTEST_STARTING_BALANCE,
)
from strategies.rsi_momentum import RSIMomentumConfig, RSIMomentumStrategy

DATA_DIR = Path(__file__).parent / "data"


def run_backtest(symbol: str = "AAPL") -> None:
    print(f"\n{'='*60}")
    print(f"  Backtest: {symbol}  |  RSI({RSI_PERIOD}) + {MA_TREND_PERIOD}-day MA")
    print(f"  Buy: RSI < {RSI_OVERSOLD} above MA  |  Sell: RSI > {RSI_OVERBOUGHT} or -{STOP_LOSS_PCT}%")
    print(f"{'='*60}\n")

    # --- Instrument ---
    instrument = TestInstrumentProvider.equity(symbol=symbol, venue="XNAS")

    # --- Data ---
    print("Downloading historical data...")
    hourly_csv = fetch_bars(symbol, "1h", output_dir=str(DATA_DIR))
    daily_csv = fetch_bars(symbol, "1d", output_dir=str(DATA_DIR))

    hourly_df = load_bars_df(hourly_csv)
    daily_df = load_bars_df(daily_csv)

    # --- Bar types ---
    hourly_bar_type = BarType.from_str(f"{instrument.id}-1-HOUR-LAST-EXTERNAL")
    daily_bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")

    # --- Convert to NautilusTrader Bar objects ---
    hourly_bars = BarDataWrangler(hourly_bar_type, instrument).process(hourly_df)
    daily_bars = BarDataWrangler(daily_bar_type, instrument).process(daily_df)
    print(f"  Loaded {len(hourly_bars)} hourly bars, {len(daily_bars)} daily bars")

    # --- Engine ---
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="WARNING"),  # set INFO for verbose output
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

    # Daily bars must be added first so the MA warms up before hourly signals
    engine.add_data(daily_bars)
    engine.add_data(hourly_bars)

    # --- Strategy ---
    strategy = RSIMomentumStrategy(
        config=RSIMomentumConfig(
            instrument_id=instrument.id,
            hourly_bar_type=hourly_bar_type,
            daily_bar_type=daily_bar_type,
            trade_size=Decimal(str(TRADE_QUANTITY)),
            rsi_period=RSI_PERIOD,
            rsi_oversold=float(RSI_OVERSOLD),
            rsi_overbought=float(RSI_OVERBOUGHT),
            ma_period=MA_TREND_PERIOD,
            stop_loss_pct=float(STOP_LOSS_PCT),
        )
    )
    engine.add_strategy(strategy)

    # --- Run ---
    print("Running backtest...")
    engine.run()

    # --- Reports ---
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
        run_backtest(sym)
