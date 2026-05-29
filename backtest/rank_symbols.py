"""
Rank all ORB_SYMBOLS by backtest performance and save results to CSV.

Usage
-----
    python -m backtest.rank_symbols               # all ORB_SYMBOLS
    python -m backtest.rank_symbols TSLA COIN MARA  # specific symbols
"""
import sys
from pathlib import Path

import pandas as pd
import zoneinfo

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
    ORB_POSITION_SIZE,
    ORB_PROFIT_MULTIPLIER,
    ORB_PROFIT_MULTIPLIERS,
    ORB_RANGE_BARS,
    ORB_STOP_BUFFER,
    ORB_SYMBOLS,
    ORB_VOLUME_FACTOR,
)
from strategies.orb import ORBConfig, ORBStrategy

ET = zoneinfo.ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SPY_INSTRUMENT = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
SPY_BAR_TYPE = BarType.from_str("SPY.XNAS-5-MINUTE-LAST-EXTERNAL")


def _parse_pnl(value) -> float:
    """Parse a NautilusTrader Money value like '-13.70 USD' to float."""
    try:
        return float(str(value).split()[0])
    except Exception:
        return 0.0


def _load_spy_bars():
    """Fetch/load SPY 5-min bars, returning processed Bar list or None."""
    try:
        csv_path = fetch_bars("SPY", "5m", output_dir=str(DATA_DIR))
        df = load_bars_df(csv_path)
        df.index = df.index.tz_convert(ET)
        df = df.between_time("09:30", "16:00")
        df.index = df.index.tz_convert("UTC")
        if len(df) < 10:
            return None
        return BarDataWrangler(SPY_BAR_TYPE, SPY_INSTRUMENT).process(df)
    except Exception as e:
        print(f"  [SPY] failed to load — {e}")
        return None


def backtest_symbol(symbol: str, spy_bars) -> tuple[dict, list[dict]] | None:
    try:
        instrument = TestInstrumentProvider.equity(symbol=symbol, venue="XNAS")

        csv_path = fetch_bars(symbol, "5m", output_dir=str(DATA_DIR))
        df = load_bars_df(csv_path)

        # Keep regular session only (9:30–16:00 ET)
        df.index = df.index.tz_convert(ET)
        df = df.between_time("09:30", "16:00")
        df.index = df.index.tz_convert("UTC")

        if len(df) < 100:
            print(f"  skipped (only {len(df)} bars)")
            return None

        bar_type = BarType.from_str(f"{instrument.id}-5-MINUTE-LAST-EXTERNAL")
        bars = BarDataWrangler(bar_type, instrument).process(df)

        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id=TraderId("BACKTESTER-001"),
                logging=LoggingConfig(log_level="ERROR"),
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

        # Add SPY data for the trend filter (only if symbol is not SPY itself)
        spy_bar_type_arg = None
        if spy_bars is not None and symbol != "SPY":
            engine.add_instrument(SPY_INSTRUMENT)
            engine.add_data(spy_bars)
            spy_bar_type_arg = SPY_BAR_TYPE

        profit_multiplier = ORB_PROFIT_MULTIPLIERS.get(symbol, ORB_PROFIT_MULTIPLIER)

        engine.add_strategy(
            ORBStrategy(
                config=ORBConfig(
                    instrument_id=instrument.id,
                    bar_type=bar_type,
                    position_size_usd=float(ORB_POSITION_SIZE),
                    orb_range_bars=ORB_RANGE_BARS,
                    profit_multiplier=profit_multiplier,
                    volume_factor=ORB_VOLUME_FACTOR,
                    stop_buffer=ORB_STOP_BUFFER,
                    close_hour=ORB_CLOSE_HOUR,
                    close_minute=ORB_CLOSE_MINUTE,
                    spy_bar_type=spy_bar_type_arg,
                    min_or_pct=ORB_MIN_OR_PCT,
                )
            )
        )
        engine.run()

        positions = engine.trader.generate_positions_report()
        account   = engine.trader.generate_account_report(XNAS)

        # Collect ML training data from the strategy's trade log
        strategy_instance = engine.trader.strategies()[0]
        trade_log = list(strategy_instance.trade_log)

        if positions.empty:
            result = dict(symbol=symbol, multiplier=profit_multiplier, trades=0, wins=0, losses=0,
                          win_rate_pct=0.0, total_pnl=0.0,
                          avg_pnl_per_trade=0.0, best_trade=0.0,
                          worst_trade=0.0, final_balance=BACKTEST_STARTING_BALANCE)
        else:
            pnls = positions["realized_pnl"].apply(_parse_pnl)
            trades = len(pnls)
            wins = int((pnls > 0).sum())
            losses = int((pnls <= 0).sum())
            total_pnl = round(float(pnls.sum()), 2)
            final_balance = (
                round(float(str(account["total"].iloc[-1]).split()[0]), 2)
                if not account.empty else BACKTEST_STARTING_BALANCE
            )
            result = dict(
                symbol=symbol,
                multiplier=profit_multiplier,
                trades=trades,
                wins=wins,
                losses=losses,
                win_rate_pct=round(wins / trades * 100, 1) if trades else 0.0,
                total_pnl=total_pnl,
                avg_pnl_per_trade=round(total_pnl / trades, 2) if trades else 0.0,
                best_trade=round(float(pnls.max()), 2),
                worst_trade=round(float(pnls.min()), 2),
                final_balance=final_balance,
            )

        engine.reset()
        engine.dispose()
        return result, trade_log

    except Exception as e:
        print(f"  ERROR — {e}")
        return None


def run_ranking(symbols: list[str]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nRunning ORB ranking for {len(symbols)} symbols")
    print(f"Settings: {ORB_RANGE_BARS}-bar OR | default {ORB_PROFIT_MULTIPLIER}x target | "
          f"{ORB_VOLUME_FACTOR}x vol | ${ORB_POSITION_SIZE}/trade | "
          f"${BACKTEST_STARTING_BALANCE:,.0f} capital")
    print(f"Filters: min OR {ORB_MIN_OR_PCT:.1%} | SPY trend filter ON\n")

    print("Loading SPY bars for trend filter...", end="  ", flush=True)
    spy_bars = _load_spy_bars()
    print("OK" if spy_bars is not None else "FAILED (trend filter disabled)")

    results    = []
    all_trades = []   # ML training data collected across all symbols

    for i, sym in enumerate(symbols, 1):
        print(f"[{i:>2}/{len(symbols)}] {sym:<6}", end="  ", flush=True)
        ret = backtest_symbol(sym, spy_bars)
        if ret:
            result, trade_log = ret
            results.append(result)
            all_trades.extend(trade_log)
            pnl_str = f"${result['total_pnl']:>+8.2f}"
            print(f"trades={result['trades']:>2}  win%={result['win_rate_pct']:>5.1f}%  "
                  f"pnl={pnl_str}  avg=${result['avg_pnl_per_trade']:>+7.2f}  "
                  f"mult={result['multiplier']:.1f}x")
        else:
            print("skipped")

    if not results:
        print("\nNo results to rank.")
        return

    df = pd.DataFrame(results).sort_values("total_pnl", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank

    csv_path = RESULTS_DIR / "orb_ranking.csv"
    df.to_csv(csv_path)

    # Save ML training data
    if all_trades:
        features_df   = pd.DataFrame(all_trades)
        features_path = RESULTS_DIR / "breakout_features.csv"
        features_df.to_csv(features_path, index=False)
        wins = int(features_df["hit_target"].sum()) if "hit_target" in features_df.columns else 0
        print(f"\n  ML training data → {features_path}"
              f"  ({len(all_trades)} trades | {wins/len(all_trades):.1%} win rate)")

    # Print ranked table
    w = 88
    print(f"\n{'='*w}")
    print(f"  ORB SYMBOL RANKING  |  ${ORB_POSITION_SIZE}/trade  |  60 days  |  ${BACKTEST_STARTING_BALANCE:,.0f} capital")
    print(f"  SPY trend filter ON  |  min OR {ORB_MIN_OR_PCT:.1%}  |  per-symbol multipliers")
    print(f"{'='*w}")
    print(f"  {'Rank':<5}{'Symbol':<8}{'Mult':>5}{'Trades':>6}{'Wins':>6}{'Win%':>7}{'Total P&L':>12}{'Avg/Trade':>11}{'Best':>10}{'Worst':>10}")
    print(f"  {'-'*w}")
    for rank, row in df.iterrows():
        flag = "  ★" if row["total_pnl"] > 0 and row["trades"] >= 3 else ""
        print(f"  {rank:<5}{row['symbol']:<8}{row['multiplier']:>4.1f}x"
              f"{row['trades']:>6}{row['wins']:>6}"
              f"{row['win_rate_pct']:>6.0f}%"
              f"  ${row['total_pnl']:>+9.2f}"
              f"  ${row['avg_pnl_per_trade']:>+8.2f}"
              f"  ${row['best_trade']:>+7.2f}"
              f"  ${row['worst_trade']:>+7.2f}{flag}")
    print(f"  {'='*w}")

    winners = df[df["total_pnl"] > 0]
    losers = df[df["total_pnl"] < 0]
    no_signal = df[df["trades"] == 0]
    print(f"\n  Profitable: {len(winners)}  |  Losing: {len(losers)}  |  No signals: {len(no_signal)}")
    print(f"  Best:  {df.iloc[0]['symbol']} ${df.iloc[0]['total_pnl']:+.2f}")
    print(f"  Worst: {df.iloc[-1]['symbol']} ${df.iloc[-1]['total_pnl']:+.2f}")
    print(f"\n  Results saved → {csv_path}\n")


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ORB_SYMBOLS
    run_ranking(symbols)
