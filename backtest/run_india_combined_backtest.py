"""
India Combined Strategy Backtest Runner.

Runs ORB + VWAP + EMA Confluence strategy against all 61 NSE Parquet files
and compares results against the plain ORB baseline side-by-side.

Usage
-----
    # Full ranking (all watchlist symbols, combined vs ORB baseline)
    python -m backtest.run_india_combined_backtest --rank

    # Single symbol with full trade log
    python -m backtest.run_india_combined_backtest JSWSTEEL

    # Full 61-symbol universe
    python -m backtest.run_india_combined_backtest --universe

    # Walk-forward validation (67/33 in-sample/out-of-sample split)
    python -m backtest.run_india_combined_backtest --walkforward

    # Refresh data first, then rank
    python -m backtest.run_india_combined_backtest --fetch --rank
"""
from __future__ import annotations

import sys
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
from zoneinfo import ZoneInfo

from backtest.fetch_nse_data import fetch_nse_bars, load_nse_bars_df
from config import (
    INDIA_ORB_BREAKOUT_STRENGTH_PCT,
    INDIA_ORB_MIN_OR_PCT,
    INDIA_ORB_PROFIT_MULTIPLIER,
    INDIA_ORB_RANGE_BARS,
    INDIA_ORB_STOP_BUFFER_PCT,
    INDIA_ORB_VOLUME_FACTOR,
    INDIA_POSITION_SIZE_INR,
    INDIA_SYMBOLS,
)
from strategies.india_combined import IndiaCombinedConfig, IndiaCombinedStrategy
from strategies.india_orb import IndiaORBConfig, IndiaORBStrategy

IST      = ZoneInfo("Asia/Kolkata")
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
XNSE = Venue("XNSE")

_NIFTY_INSTRUMENT = TestInstrumentProvider.equity(symbol="NIFTY50", venue="XNSE")
_NIFTY_BAR_TYPE   = BarType.from_str("NIFTY50.XNSE-5-MINUTE-LAST-EXTERNAL")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_nifty_bars():
    import yfinance as yf
    try:
        raw = yf.download("^NSEI", period="60d", interval="5m",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(1, axis=1)
        raw.columns = [c.lower() for c in raw.columns]
        raw.index.name = "timestamp"
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")
        ist_df = raw.copy()
        ist_df.index = ist_df.index.tz_convert(IST)
        ist_df = ist_df.between_time("09:15", "15:30")
        ist_df.index = ist_df.index.tz_convert("UTC")
        if ist_df.empty:
            return None
        return BarDataWrangler(_NIFTY_BAR_TYPE, _NIFTY_INSTRUMENT).process(ist_df)
    except Exception as e:
        print(f"  [Nifty] failed ({e}) — trend filter disabled")
        return None


def _load_symbol_df(symbol: str, date_from: str | None, date_to: str | None):
    path = DATA_DIR / f"{symbol}_NSE_5m.parquet"
    if not path.exists():
        path = fetch_nse_bars(symbol, DATA_DIR)
    df = load_nse_bars_df(path)
    if date_from:
        df = df[df.index >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        df = df[df.index < pd.Timestamp(date_to, tz="UTC")]
    return df.between_time("03:45", "10:00")


# ---------------------------------------------------------------------------
# Engine runners
# ---------------------------------------------------------------------------

def _run_combined(
    symbol: str,
    nifty_bars,
    date_from: str | None = None,
    date_to:   str | None = None,
    profit_multiplier: float = INDIA_ORB_PROFIT_MULTIPLIER,
    stop_buffer_pct:   float = INDIA_ORB_STOP_BUFFER_PCT,
    volume_factor:     float = INDIA_ORB_VOLUME_FACTOR,
    vwap_band_sigma:   float = 1.0,
    min_signals:       int   = 2,
    position_boost:    float = 1.5,
) -> tuple[list[dict], str | None]:
    try:
        instrument = TestInstrumentProvider.equity(symbol=symbol, venue="XNSE")
        df = _load_symbol_df(symbol, date_from, date_to)
        if len(df) < 50:
            return [], f"only {len(df)} bars"

        bar_type = BarType.from_str(f"{instrument.id}-5-MINUTE-LAST-EXTERNAL")
        bars     = BarDataWrangler(bar_type, instrument).process(df)

        engine = BacktestEngine(config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="ERROR"),
            risk_engine=RiskEngineConfig(bypass=True),
        ))
        engine.add_venue(
            venue=XNSE, oms_type=OmsType.NETTING,
            account_type=AccountType.CASH, base_currency=USD,
            starting_balances=[Money(500_000, USD)],
        )
        engine.add_instrument(instrument)
        engine.add_data(bars)

        nifty_bar_type_arg = None
        if nifty_bars is not None:
            engine.add_instrument(_NIFTY_INSTRUMENT)
            engine.add_data(nifty_bars)
            nifty_bar_type_arg = _NIFTY_BAR_TYPE

        strategy = IndiaCombinedStrategy(IndiaCombinedConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            position_size_inr=float(INDIA_POSITION_SIZE_INR),
            orb_range_bars=INDIA_ORB_RANGE_BARS,
            profit_multiplier=profit_multiplier,
            volume_factor=volume_factor,
            stop_buffer_pct=stop_buffer_pct,
            min_or_pct=INDIA_ORB_MIN_OR_PCT,
            nifty_bar_type=nifty_bar_type_arg,
            vwap_band_sigma=vwap_band_sigma,
            min_signals=min_signals,
            position_size_boost=position_boost,
        ))
        engine.add_strategy(strategy)
        engine.run()
        trades = list(strategy.trades)
        engine.reset()
        engine.dispose()
        return trades, None
    except Exception as e:
        return [], str(e)


def _run_orb_baseline(
    symbol: str,
    nifty_bars,
    date_from: str | None = None,
    date_to:   str | None = None,
) -> tuple[list[dict], str | None]:
    """Plain ORB (current live strategy) for comparison."""
    try:
        instrument = TestInstrumentProvider.equity(symbol=symbol, venue="XNSE")
        df = _load_symbol_df(symbol, date_from, date_to)
        if len(df) < 50:
            return [], f"only {len(df)} bars"

        bar_type = BarType.from_str(f"{instrument.id}-5-MINUTE-LAST-EXTERNAL")
        bars     = BarDataWrangler(bar_type, instrument).process(df)

        engine = BacktestEngine(config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggingConfig(log_level="ERROR"),
            risk_engine=RiskEngineConfig(bypass=True),
        ))
        engine.add_venue(
            venue=XNSE, oms_type=OmsType.NETTING,
            account_type=AccountType.CASH, base_currency=USD,
            starting_balances=[Money(500_000, USD)],
        )
        engine.add_instrument(instrument)
        engine.add_data(bars)

        nifty_bar_type_arg = None
        if nifty_bars is not None:
            engine.add_instrument(_NIFTY_INSTRUMENT)
            engine.add_data(nifty_bars)
            nifty_bar_type_arg = _NIFTY_BAR_TYPE

        strategy = IndiaORBStrategy(IndiaORBConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            position_size_inr=float(INDIA_POSITION_SIZE_INR),
            orb_range_bars=INDIA_ORB_RANGE_BARS,
            profit_multiplier=INDIA_ORB_PROFIT_MULTIPLIER,
            volume_factor=INDIA_ORB_VOLUME_FACTOR,
            stop_buffer_pct=INDIA_ORB_STOP_BUFFER_PCT,
            min_or_pct=INDIA_ORB_MIN_OR_PCT,
            nifty_bar_type=nifty_bar_type_arg,
            trailing_stop=True,
            breakout_strength_pct=INDIA_ORB_BREAKOUT_STRENGTH_PCT,
        ))
        engine.add_strategy(strategy)
        engine.run()
        trades = list(strategy.trades)
        engine.reset()
        engine.dispose()
        return trades, None
    except Exception as e:
        return [], str(e)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _stats(trades: list[dict]) -> dict:
    import math, statistics as _stats
    from collections import defaultdict

    pnls = [t["pnl"] for t in trades]
    n    = len(pnls)
    if n == 0:
        return dict(n=0, win_rate=0.0, pf=0.0, total=0.0, avg=0.0, max_dd=0.0, sharpe=0.0)

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / (abs(sum(losses)) or 0.001) if wins else 0.0
    wr     = len(wins) / n * 100

    cumsum, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        cumsum += p
        peak    = max(peak, cumsum)
        max_dd  = max(max_dd, peak - cumsum)

    daily: dict = defaultdict(float)
    for t in trades:
        day = pd.Timestamp(t["entry_ts"], unit="ns", tz="UTC").tz_convert(IST).date()
        daily[day] += t["pnl"]
    dpnls = list(daily.values())
    if len(dpnls) > 1:
        m, s = _stats.mean(dpnls), _stats.stdev(dpnls)
        sharpe = (m / s * math.sqrt(252)) if s > 0 else 0.0
    else:
        sharpe = 0.0

    return dict(n=n, win_rate=round(wr, 1), pf=round(pf, 2),
                total=round(sum(pnls), 2), avg=round(sum(pnls)/n, 2),
                max_dd=round(max_dd, 2), sharpe=round(sharpe, 2))


def _entry_type_breakdown(trades: list[dict]) -> None:
    from collections import Counter
    types = Counter(t.get("entry_type", "unknown") for t in trades)
    pnl_by_type: dict = {}
    for t in trades:
        et = t.get("entry_type", "unknown")
        pnl_by_type.setdefault(et, []).append(t["pnl"])

    print(f"\n  Entry type breakdown:")
    print(f"  {'Type':<22} {'Trades':>7} {'Wins':>6} {'Win%':>6} {'Avg P&L':>10} {'Total P&L':>12}")
    print(f"  {'-'*68}")
    for etype, pnls in sorted(pnl_by_type.items()):
        n = len(pnls)
        w = sum(1 for p in pnls if p > 0)
        print(f"  {etype:<22} {n:>7} {w:>6} {w/n*100:>5.0f}%  "
              f"₹{sum(pnls)/n:>+8,.0f}  ₹{sum(pnls):>+10,.0f}")


def _signal_count_breakdown(trades: list[dict]) -> None:
    pnl_by_sig: dict = {}
    for t in trades:
        s = t.get("signals", 0)
        pnl_by_sig.setdefault(s, []).append(t["pnl"])

    print(f"\n  Signal count breakdown:")
    print(f"  {'Signals':>8} {'Trades':>7} {'Wins':>6} {'Win%':>6} {'Avg P&L':>10} {'Total P&L':>12}")
    print(f"  {'-'*56}")
    for sig in sorted(pnl_by_sig):
        pnls = pnl_by_sig[sig]
        n = len(pnls)
        w = sum(1 for p in pnls if p > 0)
        print(f"  {sig:>8} {n:>7} {w:>6} {w/n*100:>5.0f}%  "
              f"₹{sum(pnls)/n:>+8,.0f}  ₹{sum(pnls):>+10,.0f}")


# ---------------------------------------------------------------------------
# Single-symbol detailed run
# ---------------------------------------------------------------------------

def run_single(symbol: str, nifty_bars) -> None:
    w = 80
    print(f"\n{'='*w}")
    print(f"  India Combined Backtest — {symbol}  (ORB + VWAP + EMA Confluence)")
    print(f"  ₹{INDIA_POSITION_SIZE_INR:,}/trade (×1.5 when 3 signals) | "
          f"mult={INDIA_ORB_PROFIT_MULTIPLIER}× | stop={INDIA_ORB_STOP_BUFFER_PCT:.1%} | "
          f"VWAP±1σ | EMA 9/21")
    print(f"{'='*w}\n")

    print("  Running Combined strategy...")
    c_trades, c_err = _run_combined(symbol, nifty_bars)
    print("  Running ORB baseline...")
    o_trades, o_err = _run_orb_baseline(symbol, nifty_bars)

    if c_err:
        print(f"  Combined ERROR: {c_err}")
    if o_err:
        print(f"  ORB ERROR: {o_err}")

    # --- Combined trade log ---
    if c_trades:
        print(f"\n  ── Combined Trades ──────────────────────────────────────────────")
        print(f"  {'Entry ₹':>9} {'Exit ₹':>9} {'Qty':>5} {'P&L ₹':>10} {'P&L%':>7} "
              f"{'Day':<10} {'Entry':>6} {'Sig':>4}  Type / Reason")
        print(f"  {'-'*88}")
        for t in c_trades:
            print(
                f"  {t['entry_price']:>9.2f}"
                f"  {t['exit_price']:>9.2f}"
                f"  {t['qty']:>5}"
                f"  {t['pnl']:>+10.2f}"
                f"  {t['pnl_pct']:>+6.2f}%"
                f"  {t['entry_weekday']:<10}"
                f"  {t['entry_time_ist']:>6}"
                f"  {t.get('signals', '?'):>3}"
                f"  {t.get('entry_type','')}/{t['exit_reason']}"
            )

    # --- Side-by-side summary ---
    cs = _stats(c_trades)
    os = _stats(o_trades)
    print(f"\n  ── Summary ──────────────────────────────────────────────────────")
    print(f"  {'Metric':<24} {'Combined':>12} {'ORB Baseline':>14}")
    print(f"  {'-'*52}")
    for label, ck, ok in [
        ("Trades",        "n",        "n"),
        ("Win rate",      "win_rate", "win_rate"),
        ("Profit factor", "pf",       "pf"),
        ("Total P&L (₹)", "total",    "total"),
        ("Avg P&L / trade","avg",     "avg"),
        ("Max drawdown",  "max_dd",   "max_dd"),
        ("Sharpe",        "sharpe",   "sharpe"),
    ]:
        cv = cs.get(ck, "-")
        ov = os.get(ok, "-")
        suffix = "%" if ck == "win_rate" else ""
        print(f"  {label:<24} {str(cv)+suffix:>12} {str(ov)+suffix:>14}")

    if c_trades:
        _entry_type_breakdown(c_trades)
        _signal_count_breakdown(c_trades)


# ---------------------------------------------------------------------------
# Full ranking
# ---------------------------------------------------------------------------

def run_ranking(symbols: list[str], nifty_bars, label: str = "Watchlist") -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"\nIndia Combined Strategy Ranking — {len(symbols)} symbols ({label})")
    print(f"Parameters: mult={INDIA_ORB_PROFIT_MULTIPLIER}× | stop={INDIA_ORB_STOP_BUFFER_PCT:.1%} | "
          f"vol={INDIA_ORB_VOLUME_FACTOR}× | VWAP±1σ | EMA 9/21\n")

    c_results, o_results = [], []
    all_c, all_o = [], []

    for i, sym in enumerate(symbols, 1):
        print(f"  [{i:>2}/{len(symbols)}] {sym:<16}", end="", flush=True)
        c_trades, c_err = _run_combined(sym, nifty_bars)
        o_trades, o_err = _run_orb_baseline(sym, nifty_bars)

        if c_err:
            print(f" ✗ combined: {c_err}")
            continue
        if o_err:
            print(f" ✗ orb: {o_err}")

        cs = _stats(c_trades)
        os = _stats(o_trades)
        all_c.extend(c_trades)
        all_o.extend(o_trades)

        delta_pf    = cs['pf']    - os['pf']
        delta_total = cs['total'] - os['total']
        print(
            f" Combined: {cs['n']:>3}tr  {cs['win_rate']:>5.1f}%wr  "
            f"PF{cs['pf']:.2f}  ₹{cs['total']:>+7,.0f}"
            f"  │  ORB: PF{os['pf']:.2f}  ₹{os['total']:>+7,.0f}"
            f"  │  ΔPF{delta_pf:>+.2f}  Δ₹{delta_total:>+7,.0f}"
        )
        c_results.append({**cs, "symbol": sym})
        o_results.append({**os, "symbol": sym})

    if not c_results:
        print("\n  No results.")
        return

    # Portfolio totals
    cs_port = _stats(all_c)
    os_port = _stats(all_o)
    print(f"\n  {'─'*80}")
    print(f"  PORTFOLIO  Combined: {cs_port['n']}tr  "
          f"{cs_port['win_rate']:.1f}%wr  PF{cs_port['pf']:.2f}  "
          f"₹{cs_port['total']:+,.0f}  Sharpe {cs_port['sharpe']:.2f}"
          f"  MaxDD ₹{cs_port['max_dd']:,.0f}")
    print(f"             ORB base: {os_port['n']}tr  "
          f"{os_port['win_rate']:.1f}%wr  PF{os_port['pf']:.2f}  "
          f"₹{os_port['total']:+,.0f}  Sharpe {os_port['sharpe']:.2f}"
          f"  MaxDD ₹{os_port['max_dd']:,.0f}")

    # Delta
    dpf = cs_port['pf'] - os_port['pf']
    dtot = cs_port['total'] - os_port['total']
    dsh  = cs_port['sharpe'] - os_port['sharpe']
    print(f"\n  IMPROVEMENT vs ORB: ΔPF {dpf:>+.2f}  "
          f"ΔP&L ₹{dtot:>+,.0f}  ΔSharpe {dsh:>+.2f}")

    if all_c:
        _entry_type_breakdown(all_c)
        _signal_count_breakdown(all_c)

    # Save CSV
    pd.DataFrame(c_results).sort_values("pf", ascending=False).to_csv(
        RESULTS_DIR / "india_combined_ranking.csv", index=False
    )
    print(f"\n  Results saved → {RESULTS_DIR / 'india_combined_ranking.csv'}")


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def run_walkforward(symbols: list[str], nifty_bars) -> None:
    all_dates = []
    for sym in symbols[:3]:
        path = DATA_DIR / f"{sym}_NSE_5m.parquet"
        if not path.exists():
            continue
        df = load_nse_bars_df(path)
        all_dates.extend(df.index.tolist())
    if not all_dates:
        print("No data for walk-forward.")
        return

    all_dates.sort()
    split_idx = int(len(all_dates) * 0.67)
    split_ts  = all_dates[split_idx]
    date_from = str(all_dates[0].date())
    date_split = str(pd.Timestamp(split_ts).date())
    date_to    = str(all_dates[-1].date())

    print(f"\nWalk-forward: in-sample {date_from}→{date_split} | "
          f"out-of-sample {date_split}→{date_to}")

    for label, dfrom, dto in [
        ("IN-SAMPLE  (train)", date_from, date_split),
        ("OUT-OF-SAMPLE (test)", date_split, date_to),
    ]:
        all_c, all_o = [], []
        for sym in symbols:
            c_t, _ = _run_combined(sym, nifty_bars, dfrom, dto)
            o_t, _ = _run_orb_baseline(sym, nifty_bars, dfrom, dto)
            all_c.extend(c_t)
            all_o.extend(o_t)
        cs = _stats(all_c)
        os = _stats(all_o)
        print(f"\n  {label}")
        print(f"    Combined  {cs['n']:>3}tr  {cs['win_rate']:.1f}%wr  "
              f"PF{cs['pf']:.2f}  ₹{cs['total']:+,.0f}  Sharpe {cs['sharpe']:.2f}")
        print(f"    ORB base  {os['n']:>3}tr  {os['win_rate']:.1f}%wr  "
              f"PF{os['pf']:.2f}  ₹{os['total']:+,.0f}  Sharpe {os['sharpe']:.2f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from backtest.fetch_nse_data import fetch_nse_bars

    NSE_UNIVERSE = [
        "TORNTPHARM", "BHARTIARTL", "JSWSTEEL", "BAJFINANCE", "GODREJCP",
        "HCLTECH", "INFY", "VEDL", "DABUR", "DRREDDY", "HINDUNILVR", "ONGC",
        # Extended universe for --universe flag
        "SUNPHARMA", "ADANIENT", "POWERGRID", "RELIANCE", "IDFCFIRSTB",
        "BPCL", "AXISBANK", "INDUSINDBK", "BAJAJFINSV", "BANKBARODA",
        "FEDERALBNK", "CHOLAFIN", "MUTHOOTFIN", "HEROMOTOCO", "EICHERMOT",
        "TVSMOTOR", "TATASTEEL", "HINDALCO", "GRASIM", "ULTRACEMCO",
        "CIPLA", "DIVISLAB", "APOLLOHOSP", "TECHM", "COALINDIA",
        "BRITANNIA", "MARICO", "TATACONSUM", "HAL", "BEL",
        "HAVELLS", "SIEMENS", "ABB", "TRENT", "PIDILITIND",
        "MARUTI", "DMART", "TITAN", "NTPC", "TCS",
        "SBIN", "ICICIBANK", "KOTAKBANK", "WIPRO", "HDFCLIFE",
        "ITC", "IRCTC", "LT", "HDFCBANK",
    ]

    parser = argparse.ArgumentParser(description="India Combined Strategy Backtest")
    parser.add_argument("symbol", nargs="?", help="Single symbol to analyse")
    parser.add_argument("--rank",        action="store_true", help="Rank all watchlist symbols")
    parser.add_argument("--universe",    action="store_true", help="Run full NSE universe (~61 symbols)")
    parser.add_argument("--walkforward", action="store_true", help="67/33 walk-forward validation")
    parser.add_argument("--fetch",       action="store_true", help="Download/refresh NSE data first")
    args = parser.parse_args()

    symbols = INDIA_SYMBOLS if (args.rank or args.walkforward) else \
              [s for s in NSE_UNIVERSE if (DATA_DIR / f"{s}_NSE_5m.parquet").exists()] \
              if args.universe else []

    if args.fetch:
        all_syms = list(set(symbols) | set(NSE_UNIVERSE[:20]))
        print(f"Fetching {len(all_syms)} symbols...")
        for sym in all_syms:
            try:
                fetch_nse_bars(sym, DATA_DIR)
                print(f"  ✓ {sym}")
            except Exception as e:
                print(f"  ✗ {sym}: {e}")

    print("Loading Nifty bars for trend filter...")
    nifty_bars = _load_nifty_bars()
    print(f"  Nifty: {'loaded' if nifty_bars is not None else 'unavailable — trend filter disabled'}")

    if args.symbol:
        run_single(args.symbol.upper(), nifty_bars)
    elif args.rank:
        run_ranking(INDIA_SYMBOLS, nifty_bars, label="12-symbol watchlist")
    elif args.universe:
        run_ranking(symbols, nifty_bars, label="full universe")
    elif args.walkforward:
        run_walkforward(INDIA_SYMBOLS, nifty_bars)
    else:
        parser.print_help()
