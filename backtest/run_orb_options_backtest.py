"""
Run a NautilusTrader backtest of the ORB Options strategy.

Options are NOT fetched from a live chain — instead they are priced at entry
and exit using Black-Scholes with realized historical volatility as the IV proxy.
This makes the backtest fully self-contained with no paid data requirement.

Usage
-----
    # Single symbol (detailed trade log)
    python -m backtest.run_orb_options_backtest AAPL

    # Multiple symbols
    python -m backtest.run_orb_options_backtest AAPL TSLA COIN

    # Full ranking over all ORB_SYMBOLS
    python -m backtest.run_orb_options_backtest --rank
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
    ORB_OPTIONS_IV_THRESHOLD,
    ORB_OPTIONS_POSITION_SIZE,
    ORB_PROFIT_MULTIPLIER,
    ORB_PROFIT_MULTIPLIERS,
    ORB_RANGE_BARS,
    ORB_STOP_BUFFER,
    ORB_SYMBOLS,
    ORB_VOLUME_FACTOR,
)
from strategies.orb_options import ORBOptionsConfig, ORBOptionsStrategy

ET = zoneinfo.ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SPY_INSTRUMENT = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
SPY_BAR_TYPE = BarType.from_str("SPY.XNAS-5-MINUTE-LAST-EXTERNAL")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_spy_bars():
    try:
        path = fetch_bars("SPY", "5m", output_dir=str(DATA_DIR))
        df = load_bars_df(path)
        df.index = df.index.tz_convert(ET)
        df = df.between_time("09:30", "16:00")
        df.index = df.index.tz_convert("UTC")
        return BarDataWrangler(SPY_BAR_TYPE, SPY_INSTRUMENT).process(df) if len(df) >= 10 else None
    except Exception as e:
        print(f"  [SPY] failed — {e}")
        return None


def _run_engine(symbol: str, spy_bars) -> tuple[list[dict], str | None]:
    try:
        instrument = TestInstrumentProvider.equity(symbol=symbol, venue="XNAS")
        path = fetch_bars(symbol, "5m", output_dir=str(DATA_DIR))
        df = load_bars_df(path)
        df.index = df.index.tz_convert(ET)
        df = df.between_time("09:30", "16:00")
        df.index = df.index.tz_convert("UTC")
        if len(df) < 100:
            return [], f"only {len(df)} bars"

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

        spy_bar_type_arg = None
        if spy_bars is not None and symbol != "SPY":
            engine.add_instrument(SPY_INSTRUMENT)
            engine.add_data(spy_bars)
            spy_bar_type_arg = SPY_BAR_TYPE

        mult = ORB_PROFIT_MULTIPLIERS.get(symbol, ORB_PROFIT_MULTIPLIER)
        strategy = ORBOptionsStrategy(
            config=ORBOptionsConfig(
                instrument_id=instrument.id,
                bar_type=bar_type,
                position_size_usd=float(ORB_OPTIONS_POSITION_SIZE),
                orb_range_bars=ORB_RANGE_BARS,
                profit_multiplier=mult,
                volume_factor=ORB_VOLUME_FACTOR,
                stop_buffer=ORB_STOP_BUFFER,
                close_hour=ORB_CLOSE_HOUR,
                close_minute=ORB_CLOSE_MINUTE,
                min_or_pct=ORB_MIN_OR_PCT,
                iv_threshold=ORB_OPTIONS_IV_THRESHOLD,
                spy_bar_type=spy_bar_type_arg,
            )
        )
        engine.add_strategy(strategy)
        engine.run()
        trades = list(strategy.trades)
        engine.reset()
        engine.dispose()
        return trades, None

    except Exception as e:
        return [], str(e)


def _summarize(symbol: str, trades: list[dict]) -> dict:
    if not trades:
        return dict(symbol=symbol, trades=0, wins=0, losses=0, win_rate_pct=0.0,
                    total_pnl=0.0, avg_pnl=0.0, best=0.0, worst=0.0,
                    iron_condor=0, bull_call=0, bear_put=0, straddle=0)
    pnls = [t["pnl"] for t in trades]
    strats = {}
    for t in trades:
        strats[t["strategy"]] = strats.get(t["strategy"], 0) + 1
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    return dict(
        symbol=symbol,
        trades=n,
        wins=wins,
        losses=n - wins,
        win_rate_pct=round(wins / n * 100, 1),
        total_pnl=round(sum(pnls), 2),
        avg_pnl=round(sum(pnls) / n, 2),
        best=round(max(pnls), 2),
        worst=round(min(pnls), 2),
        iron_condor=strats.get("Iron Condor", 0),
        bull_call=strats.get("Bull Call Spread", 0),
        bear_put=strats.get("Bear Put Spread", 0),
        straddle=strats.get("Straddle", 0),
    )


# ---------------------------------------------------------------------------
# Single-symbol detailed output
# ---------------------------------------------------------------------------

def run_single(symbol: str, spy_bars) -> None:
    print(f"\n{'='*72}")
    print(f"  ORB Options Backtest — {symbol}")
    print(f"  ${ORB_OPTIONS_POSITION_SIZE}/trade | IV threshold {ORB_OPTIONS_IV_THRESHOLD:.0%} | Black-Scholes pricing")
    print(f"{'='*72}\n")

    trades, err = _run_engine(symbol, spy_bars)
    if err:
        print(f"  ERROR: {err}")
        return
    if not trades:
        print("  No trades triggered.")
        return

    print(f"  {'Strategy':<22} {'Entry$':>7} {'Exit$':>7} {'Qty':>4} {'P&L':>10}  Reason")
    print(f"  {'-'*62}")
    for t in trades:
        print(
            f"  {t['strategy']:<22}"
            f"  {t['entry_cost']:>5.2f}"
            f"  {t.get('exit_value', 0):>5.2f}"
            f"  {t.get('qty',1):>4}"
            f"  ${t['pnl']:>+8.2f}"
            f"  {t.get('exit_reason','?')}"
        )

    s = _summarize(symbol, trades)
    print(f"\n  Trades: {s['trades']}  Wins: {s['wins']}  Win%: {s['win_rate_pct']:.0f}%"
          f"  Total P&L: ${s['total_pnl']:+.2f}  Avg: ${s['avg_pnl']:+.2f}")
    print(f"  Strategies: IC={s['iron_condor']} BCS={s['bull_call']}"
          f" BPS={s['bear_put']} STR={s['straddle']}")


# ---------------------------------------------------------------------------
# Full ranking
# ---------------------------------------------------------------------------

def run_ranking(symbols: list[str]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nRunning ORB Options ranking for {len(symbols)} symbols")
    print(f"Settings: ${ORB_OPTIONS_POSITION_SIZE}/trade | IV threshold {ORB_OPTIONS_IV_THRESHOLD:.0%}"
          f" | Black-Scholes | SPY trend filter\n")

    print("Loading SPY bars...", end="  ", flush=True)
    spy_bars = _load_spy_bars()
    print("OK" if spy_bars else "FAILED (trend filter disabled)")

    results = []
    for i, sym in enumerate(symbols, 1):
        print(f"[{i:>2}/{len(symbols)}] {sym:<6}", end="  ", flush=True)
        trades, err = _run_engine(sym, spy_bars)
        if err:
            print(f"ERROR: {err}")
            continue
        r = _summarize(sym, trades)
        results.append(r)
        if r["trades"] == 0:
            print("no trades")
        else:
            print(
                f"trades={r['trades']:>2}  win%={r['win_rate_pct']:>5.1f}%"
                f"  pnl=${r['total_pnl']:>+8.2f}  avg=${r['avg_pnl']:>+7.2f}"
                f"  IC={r['iron_condor']} BCS={r['bull_call']}"
                f" BPS={r['bear_put']} STR={r['straddle']}"
            )

    if not results:
        print("\nNo results.")
        return

    df = (pd.DataFrame(results)
          .sort_values("total_pnl", ascending=False)
          .reset_index(drop=True))
    df.index += 1

    csv_path = RESULTS_DIR / "orb_options_ranking.csv"
    df.to_csv(csv_path)

    w = 100
    print(f"\n{'='*w}")
    print(f"  ORB OPTIONS RANKING | ${ORB_OPTIONS_POSITION_SIZE}/trade | IV {ORB_OPTIONS_IV_THRESHOLD:.0%} | Black-Scholes pricing")
    print(f"{'='*w}")
    print(f"  {'Rank':<5}{'Symbol':<8}{'Trades':>7}{'Win%':>7}{'Total P&L':>12}"
          f"{'Avg':>10}{'Best':>10}{'Worst':>10}  Strategies")
    print(f"  {'-'*w}")
    for rank, row in df.iterrows():
        flag = "  ★" if row["total_pnl"] > 0 and row["trades"] >= 3 else ""
        strats = (f"IC={int(row['iron_condor'])} BCS={int(row['bull_call'])}"
                  f" BPS={int(row['bear_put'])} STR={int(row['straddle'])}")
        print(
            f"  {rank:<5}{row['symbol']:<8}{int(row['trades']):>7}"
            f"{row['win_rate_pct']:>6.0f}%"
            f"  ${row['total_pnl']:>+9.2f}"
            f"  ${row['avg_pnl']:>+7.2f}"
            f"  ${row['best']:>+7.2f}"
            f"  ${row['worst']:>+7.2f}  {strats}{flag}"
        )
    print(f"  {'='*w}")

    winners = df[df["total_pnl"] > 0]
    losers = df[df["total_pnl"] < 0]
    print(f"\n  Profitable: {len(winners)}  |  Losing: {len(losers)}")
    if len(df) > 0:
        print(f"  Best:  {df.iloc[0]['symbol']} ${df.iloc[0]['total_pnl']:+.2f}")
        print(f"  Worst: {df.iloc[-1]['symbol']} ${df.iloc[-1]['total_pnl']:+.2f}")
    print(f"\n  Results saved → {csv_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--rank" in args:
        syms = [a for a in args if a != "--rank"] or ORB_SYMBOLS
        run_ranking(syms)
    else:
        syms = args or ["AAPL"]
        print("Loading SPY bars...", end="  ", flush=True)
        spy = _load_spy_bars()
        print("OK" if spy else "FAILED")
        for sym in syms:
            run_single(sym, spy)
