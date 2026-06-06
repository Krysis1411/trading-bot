"""
ORB Options Strategy — NautilusTrader backtesting implementation.

Replays equity 5-min bars through the same ORB signal logic as the live bot
(orb_options_bot.py) and simulates options P&L using Black-Scholes with
realized historical volatility (20-bar annualised HV) as the IV proxy.

Pricing
-------
Options are priced at entry and exit using Black-Scholes with realised
historical volatility (20-bar annualised HV) as the IV proxy.

Strategy selection (mirrors orb_options_bot.py)
-----------------------------------------------
  IV rank > threshold + range-bound         → Iron Condor
  IV rank ≤ threshold + breakout up + SPY ↑ → Bull Call Spread
  IV rank ≤ threshold + breakout up + SPY ? → Straddle
  IV rank ≤ threshold + breakout dn + SPY ↓ → Bear Put Spread
  IV rank ≤ threshold + breakout dn + SPY ? → Straddle

Quality filters (must all pass before entry)
--------------------------------------------
  MIN_UNDERLYING_PRICE   — skip illiquid options chains
  SKIP_MONDAY_ENTRIES    — Mondays have 8.7% win rate on IC symbols
  IC_MAX_ENTRY_HOUR/MIN  — entries after 12:30 PM have 7.5% win rate
  MIN_BREAKOUT_STRENGTH_PCT — directional only; weak breakouts fail
  IC_SIGMA_MULTIPLE      — short strikes ≥ 1.5× expected move
  IC_MIN_CREDIT_RATIO    — credit ≥ 20% of spread width

Management exits
----------------
  IC profit target  — close when unrealised P&L ≥ IC_PROFIT_TARGET_PCT × credit
  IC P&L stop       — close when loss ≥ IC_PNL_STOP_MULTIPLE × credit
  Price-based stop  — close when price touches a short strike
  EOD               — force-close at close_hour:close_minute

SYNC NOTE
---------
After any change to orb_options_bot.py or config.py run:
    python check_backtest_parity.py
and update this file if it reports drift.
"""
import math
from collections import deque
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from nautilus_trader.config import PositiveFloat, PositiveInt, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from config import (
    IC_MAX_ENTRY_HOUR,
    IC_MAX_ENTRY_MINUTE,
    IC_MIN_CREDIT_RATIO,
    IC_PNL_STOP_MULTIPLE,
    IC_PROFIT_TARGET_PCT,
    IC_SIGMA_MULTIPLE,
    MIN_BREAKOUT_STRENGTH_PCT,
    MIN_UNDERLYING_PRICE,
    SKIP_MONDAY_ENTRIES,
)

ET = ZoneInfo("America/New_York")
_RISK_FREE = 0.05
_BARS_PER_DAY = 78         # 5-min bars in a regular session
_TRADING_DAYS = 252

# IV rank threshold — mirrors live bot: NORMAL regime (70.0) × 0.75 0DTE compression.
# The backtest always runs intraday (T = hours left today), so is always effectively 0DTE.
_IV_RANK_THRESHOLD = 52.5

# Rolling window for IV rank: ~1 trading year of every-bar HV samples
_IV_RANK_WINDOW = _TRADING_DAYS * _BARS_PER_DAY


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _bs(S: float, K: float, T: float, sigma: float, kind: str) -> float:
    """Black-Scholes price for a European call or put. T in years."""
    intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    if T <= 1e-6 or sigma <= 1e-6 or S <= 0 or K <= 0:
        return max(0.01, intrinsic)
    d1 = (math.log(S / K) + (_RISK_FREE + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        price = S * _ncdf(d1) - K * math.exp(-_RISK_FREE * T) * _ncdf(d2)
    else:
        price = K * math.exp(-_RISK_FREE * T) * _ncdf(-d2) - S * _ncdf(-d1)
    return max(0.01, price)


def _hv(closes: list[float]) -> float:
    """Annualised realized volatility from 5-min closes (floor 10%, cap 200%)."""
    if len(closes) < 5:
        return 0.30
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if not rets:
        return 0.30
    var = sum(r ** 2 for r in rets) / len(rets)
    return max(0.10, min(2.0, math.sqrt(var * _BARS_PER_DAY * _TRADING_DAYS)))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class ORBOptionsConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    position_size_usd: float = 500.0
    orb_range_bars: PositiveInt = 6
    profit_multiplier: PositiveFloat = 1.5
    volume_factor: PositiveFloat = 1.0
    stop_buffer: float = 0.05
    close_hour: int = 15
    close_minute: int = 45
    min_or_pct: float = 0.005
    spy_bar_type: BarType | None = None
    # Quality filters are sourced directly from config.py at module level
    # (NautilusTrader frozen configs cannot hold lists or arbitrary objects).


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ORBOptionsStrategy(Strategy):
    """
    NautilusTrader strategy that drives options P&L simulation from equity bars.
    Submits no equity orders — all tracking is done in self.trades.
    """

    def __init__(self, config: ORBOptionsConfig) -> None:
        super().__init__(config)
        self.trades: list[dict] = []
        self._closes: list[float] = []
        self._hv_history: deque = deque(maxlen=_IV_RANK_WINDOW)
        self._open_trade: dict | None = None
        self._current_date: date | None = None
        self._spy_open: float | None = None
        self._spy_last: float | None = None
        self._spy_date: date | None = None
        self._reset_day()

    # ------------------------------------------------------------------
    # Daily state
    # ------------------------------------------------------------------

    def _reset_day(self) -> None:
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_vol_sum: float = 0.0
        self._or_bars: int = 0
        self._range_ready: bool = False
        self._range_skip: bool = False
        self._traded: bool = False

    def _close_time(self) -> time:
        return time(self.config.close_hour, self.config.close_minute)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def _bar_et(self, bar: Bar) -> datetime:
        return datetime.fromtimestamp(
            bar.ts_event / 1_000_000_000, tz=timezone.utc
        ).astimezone(ET)

    def _years_remaining(self, dt_et: datetime) -> float:
        close_dt = dt_et.replace(
            hour=self.config.close_hour, minute=self.config.close_minute,
            second=0, microsecond=0,
        )
        hours = max(0.0, (close_dt - dt_et).total_seconds() / 3600)
        return hours / (_TRADING_DAYS * 6.5)

    # ------------------------------------------------------------------
    # IV rank (rolling HV percentile — approximates live IV rank)
    # ------------------------------------------------------------------

    def _iv_rank(self, iv: float) -> float:
        """Return IV rank 0–100 based on rolling HV history."""
        if len(self._hv_history) < 20:
            return 50.0   # not enough history — neutral
        hist = list(self._hv_history)
        hv_min, hv_max = min(hist), max(hist)
        if hv_max <= hv_min:
            return 50.0
        return max(0.0, min(100.0, (iv - hv_min) / (hv_max - hv_min) * 100))

    # ------------------------------------------------------------------
    # SPY trend
    # ------------------------------------------------------------------

    def _handle_spy(self, bar: Bar) -> None:
        dt = self._bar_et(bar)
        if self._spy_date != dt.date():
            self._spy_date = dt.date()
            self._spy_open = bar.open.as_double()
        self._spy_last = bar.close.as_double()

    def _spy_bullish(self) -> bool | None:
        if self.config.spy_bar_type is None:
            return None
        if self._spy_open is None or self._spy_last is None:
            return None
        return self._spy_last >= self._spy_open

    # ------------------------------------------------------------------
    # Options entry
    # ------------------------------------------------------------------

    def _enter(self, price: float, dt_et: datetime, signal: str,
               iv: float, iv_rank: float) -> None:

        # --- Pre-entry filters ---

        # Minimum underlying price (illiquid options on cheap stocks)
        if price < MIN_UNDERLYING_PRICE:
            return

        # Monday skip — 8.7% win rate on IC-dominant symbols
        if SKIP_MONDAY_ENTRIES and dt_et.weekday() == 0:
            return

        # Entry time cutoff — win rate drops to 7.5% after 12:30 PM
        if dt_et.time() >= time(IC_MAX_ENTRY_HOUR, IC_MAX_ENTRY_MINUTE):
            return

        or_high = self._or_high
        or_low  = self._or_low
        or_range = or_high - or_low
        T = self._years_remaining(dt_et)
        spy = self._spy_bullish()

        strategy = None
        entry_cost = 0.0
        meta: dict = {}

        if iv_rank > _IV_RANK_THRESHOLD and signal == "range_bound":
            # ----------------------------------------------------------------
            # Iron Condor — sell premium around sigma-based OTM strikes
            # ----------------------------------------------------------------
            strategy = "Iron Condor"
            w = 5.0 if price > 100 else (2.0 if price > 50 else 1.0)

            # Sigma-based strike placement: short strikes ≥ IC_SIGMA_MULTIPLE × expected move
            expected_move = price * iv * math.sqrt(T) if T > 0 else 0.0
            min_call_k = price + IC_SIGMA_MULTIPLE * expected_move
            max_put_k  = price - IC_SIGMA_MULTIPLE * expected_move

            sc_k = max(or_high, min_call_k)
            lc_k = sc_k + w
            sp_k = min(or_low,  max_put_k)
            lp_k = sp_k - w

            if lp_k <= 0:
                return

            sc = _bs(price, sc_k, T, iv, "call")
            lc = _bs(price, lc_k, T, iv, "call")
            sp = _bs(price, sp_k, T, iv, "put")
            lp = _bs(price, lp_k, T, iv, "put")

            net_credit = (sc + sp) - (lc + lp)
            if net_credit <= 0:
                return

            # Minimum credit-to-width ratio
            if net_credit / w < IC_MIN_CREDIT_RATIO:
                return

            entry_cost = -net_credit
            risk = w - net_credit
            qty = max(1, int(self.config.position_size_usd / max(0.01, risk * 100)))
            meta = dict(sc_k=sc_k, lc_k=lc_k, sp_k=sp_k, lp_k=lp_k,
                        width=w, qty=qty, initial_credit=net_credit)

        elif iv_rank <= _IV_RANK_THRESHOLD:
            if signal == "breakout_above":
                # Minimum breakout strength filter for directional trades
                strength = (price - or_high) / or_high
                if strength < MIN_BREAKOUT_STRENGTH_PCT:
                    return

                if spy is True:
                    strategy = "Bull Call Spread"
                    k_long, k_short = or_high, or_high + or_range
                    lc = _bs(price, k_long,  T, iv, "call")
                    sc = _bs(price, k_short, T, iv, "call")
                    entry_cost = max(0.05, lc - sc)
                    qty = max(1, int(self.config.position_size_usd / (entry_cost * 100)))
                    meta = dict(k_long=k_long, k_short=k_short, qty=qty)
                else:
                    strategy = "Straddle"
                    k = price
                    entry_cost = _bs(price, k, T, iv, "call") + _bs(price, k, T, iv, "put")
                    qty = max(1, int(self.config.position_size_usd / (entry_cost * 100)))
                    meta = dict(k=k, qty=qty)

            elif signal == "breakout_below":
                # Minimum breakout strength filter for directional trades
                strength = (or_low - price) / or_low
                if strength < MIN_BREAKOUT_STRENGTH_PCT:
                    return

                if spy is False:
                    strategy = "Bear Put Spread"
                    k_long, k_short = or_low, or_low - or_range
                    lp = _bs(price, k_long,  T, iv, "put")
                    sp = _bs(price, k_short, T, iv, "put")
                    entry_cost = max(0.05, lp - sp)
                    qty = max(1, int(self.config.position_size_usd / (entry_cost * 100)))
                    meta = dict(k_long=k_long, k_short=k_short, qty=qty)
                else:
                    strategy = "Straddle"
                    k = price
                    entry_cost = _bs(price, k, T, iv, "call") + _bs(price, k, T, iv, "put")
                    qty = max(1, int(self.config.position_size_usd / (entry_cost * 100)))
                    meta = dict(k=k, qty=qty)

        if strategy is None:
            return

        self._open_trade = dict(
            strategy=strategy,
            entry_price=price,
            entry_cost=entry_cost,
            iv=iv,
            or_high=or_high,
            or_low=or_low,
            or_range=or_range,
            T_entry=T,
            meta=meta,
        )
        self._traded = True
        self.log.info(
            f"ENTRY {strategy} | px={price:.2f} iv={iv:.1%} iv_rank={iv_rank:.0f}"
            f" cost=${entry_cost:.2f} qty={meta.get('qty', 1)}"
            f" total=${entry_cost * meta.get('qty', 1) * 100:.0f}"
        )

    # ------------------------------------------------------------------
    # Options exit
    # ------------------------------------------------------------------

    def _exit(self, price: float, dt_et: datetime, reason: str) -> None:
        if self._open_trade is None:
            return

        t = self._open_trade
        strategy = t["strategy"]
        iv = t["iv"]
        T = self._years_remaining(dt_et)
        meta = t["meta"]
        qty = meta.get("qty", 1)

        if strategy == "Iron Condor":
            sc = _bs(price, meta["sc_k"], T, iv, "call")
            lc = _bs(price, meta["lc_k"], T, iv, "call")
            sp = _bs(price, meta["sp_k"], T, iv, "put")
            lp = _bs(price, meta["lp_k"], T, iv, "put")
            cost_to_close = (sc + sp) - (lc + lp)
            exit_value = -cost_to_close

        elif strategy == "Bull Call Spread":
            exit_value = (_bs(price, meta["k_long"],  T, iv, "call") -
                          _bs(price, meta["k_short"], T, iv, "call"))

        elif strategy == "Bear Put Spread":
            exit_value = (_bs(price, meta["k_long"],  T, iv, "put") -
                          _bs(price, meta["k_short"], T, iv, "put"))

        else:  # Straddle
            exit_value = (_bs(price, meta["k"], T, iv, "call") +
                          _bs(price, meta["k"], T, iv, "put"))

        pnl = (exit_value - t["entry_cost"]) * qty * 100

        record = {**t, "exit_price": price, "exit_value": exit_value,
                  "pnl": pnl, "exit_reason": reason, "qty": qty}
        self.trades.append(record)
        self._open_trade = None

        self.log.info(
            f"EXIT  {strategy} | px={price:.2f} | "
            f"entry=${t['entry_cost']:.2f} exit=${exit_value:.2f} | "
            f"P&L=${pnl:+.2f} ({reason})"
        )

    # ------------------------------------------------------------------
    # NautilusTrader callbacks
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)
        if self.config.spy_bar_type is not None:
            self.subscribe_bars(self.config.spy_bar_type)

    def on_bar(self, bar: Bar) -> None:
        if self.config.spy_bar_type is not None and bar.bar_type == self.config.spy_bar_type:
            self._handle_spy(bar)
            return

        dt_et = self._bar_et(bar)
        bar_date = dt_et.date()
        bar_time = dt_et.time()

        # Day rollover
        if self._current_date != bar_date:
            if self._open_trade is not None:
                self._exit(bar.open.as_double(), dt_et, "EOD")
            self._current_date = bar_date
            self._reset_day()
            self._closes.clear()

        close = bar.close.as_double()
        self._closes.append(close)
        if len(self._closes) > 30:
            self._closes.pop(0)

        # Update rolling IV history every bar
        iv = _hv(self._closes)
        self._hv_history.append(iv)

        # Phase 1 — build opening range
        if not self._range_ready:
            if time(9, 30) <= bar_time < time(10, 0):
                h = bar.high.as_double()
                l = bar.low.as_double()
                self._or_high = max(self._or_high, h) if self._or_high else h
                self._or_low  = min(self._or_low,  l) if self._or_low  else l
                self._or_vol_sum += bar.volume.as_double()
                self._or_bars += 1
                if self._or_bars >= self.config.orb_range_bars:
                    self._range_ready = True
                    r = self._or_high - self._or_low
                    if self.config.min_or_pct > 0 and r / self._or_high < self.config.min_or_pct:
                        self._range_skip = True
            return

        if self._range_skip:
            return

        # Phase 2 — EOD forced close
        if bar_time >= self._close_time():
            if self._open_trade is not None:
                self._exit(close, dt_et, "EOD")
            return

        # Phase 3 — manage open position
        if self._open_trade is not None:
            t    = self._open_trade
            strat = t["strategy"]
            or_h  = t["or_high"]
            or_l  = t["or_low"]
            or_r  = t["or_range"]
            mult  = self.config.profit_multiplier

            if strat in ("Bull Call Spread", "Straddle") and t["entry_price"] > or_h:
                if close <= or_l - self.config.stop_buffer:
                    self._exit(close, dt_et, "STOP")
                elif close >= or_h + or_r * mult:
                    self._exit(close, dt_et, "TARGET")

            elif strat in ("Bear Put Spread", "Straddle") and t["entry_price"] < or_l:
                if close >= or_h + self.config.stop_buffer:
                    self._exit(close, dt_et, "STOP")
                elif close <= or_l - or_r * mult:
                    self._exit(close, dt_et, "TARGET")

            elif strat == "Iron Condor":
                sc_k = t["meta"]["sc_k"]
                lc_k = t["meta"]["lc_k"]
                sp_k = t["meta"]["sp_k"]
                lp_k = t["meta"]["lp_k"]
                initial_credit = t["meta"].get("initial_credit", 0.0)
                T_now = self._years_remaining(dt_et)

                # P&L-based exits (profit target + credit-multiple stop)
                if initial_credit > 0:
                    sc_now = _bs(close, sc_k, T_now, iv, "call")
                    lc_now = _bs(close, lc_k, T_now, iv, "call")
                    sp_now = _bs(close, sp_k, T_now, iv, "put")
                    lp_now = _bs(close, lp_k, T_now, iv, "put")
                    cost_to_close = (sc_now + sp_now) - (lc_now + lp_now)
                    unrealized_pnl = initial_credit - cost_to_close

                    if unrealized_pnl >= IC_PROFIT_TARGET_PCT * initial_credit:
                        self._exit(close, dt_et, "TARGET")
                        return
                    if unrealized_pnl <= -(IC_PNL_STOP_MULTIPLE * initial_credit):
                        self._exit(close, dt_et, "STOP")
                        return

                # Price-based stop at the sigma-placed short strikes
                if close >= sc_k or close <= sp_k:
                    self._exit(close, dt_et, "STOP")
            return

        # Phase 4 — look for entry signal
        if self._traded:
            return

        avg_vol = self._or_vol_sum / max(self._or_bars, 1)
        vol_ok  = bar.volume.as_double() >= avg_vol * self.config.volume_factor
        or_r    = self._or_high - self._or_low
        after_1030 = bar_time >= time(10, 30)

        is_range = (
            after_1030
            and (self._or_low + 0.2 * or_r) <= close <= (self._or_high - 0.2 * or_r)
        )

        iv_rank = self._iv_rank(iv)

        if close > self._or_high and vol_ok:
            self._enter(close, dt_et, "breakout_above", iv, iv_rank)
        elif close < self._or_low and vol_ok:
            self._enter(close, dt_et, "breakout_below", iv, iv_rank)
        elif is_range and iv_rank > _IV_RANK_THRESHOLD:
            self._enter(close, dt_et, "range_bound", iv, iv_rank)

    def on_stop(self) -> None:
        if self._open_trade is not None:
            self._open_trade.update(exit_price=0, exit_value=0, pnl=0,
                                    exit_reason="ENGINE_STOP",
                                    qty=self._open_trade["meta"].get("qty", 1))
            self.trades.append(self._open_trade)
            self._open_trade = None
        self.unsubscribe_bars(self.config.bar_type)
        if self.config.spy_bar_type is not None:
            self.unsubscribe_bars(self.config.spy_bar_type)

    def on_reset(self) -> None:
        self._reset_day()
        self._current_date = None
        self._open_trade = None
        self._spy_open = None
        self._spy_last = None
        self._spy_date = None
        self._closes = []
        self._hv_history.clear()

    def on_save(self) -> dict[str, bytes]:
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        pass

    def on_dispose(self) -> None:
        pass
