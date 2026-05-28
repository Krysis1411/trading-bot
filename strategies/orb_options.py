"""
ORB Options Strategy — NautilusTrader backtesting implementation.

Replays equity 5-min bars through the same ORB signal logic as ORBStrategy,
but simulates options P&L instead of submitting equity orders.

Pricing
-------
Options are priced at entry and exit using Black-Scholes with realized
historical volatility (20-bar annualized HV) as the IV proxy.

Strategy selection (mirrors orb_options_bot.py)
-----------------------------------------------
  High IV + range-bound          → Iron Condor
  Low IV + breakout up + SPY up  → Bull Call Spread
  Low IV + breakout up + SPY ??  → Straddle
  Low IV + breakout dn + SPY dn  → Bear Put Spread
  Low IV + breakout dn + SPY ??  → Straddle

P&L convention
--------------
  entry_cost  : net cash paid at open  (positive = debit,  negative = credit)
  exit_value  : net cash received at close
  pnl         : exit_value - entry_cost  (positive = profit)
"""
import math
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from nautilus_trader.config import PositiveFloat, PositiveInt, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

ET = ZoneInfo("America/New_York")
_RISK_FREE = 0.05          # annualized risk-free rate
_BARS_PER_DAY = 78         # 5-min bars in a regular session
_TRADING_DAYS = 252


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
    """Annualized realized volatility from 5-min closes (floor 10%, cap 200%)."""
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
    iv_threshold: float = 0.45
    spy_bar_type: BarType | None = None


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

    def _enter(self, price: float, dt_et: datetime, signal: str, iv: float) -> None:
        or_high = self._or_high
        or_low = self._or_low
        or_range = or_high - or_low
        T = self._years_remaining(dt_et)
        spy = self._spy_bullish()

        strategy = None
        entry_cost = 0.0
        meta: dict = {}

        if iv > self.config.iv_threshold and signal == "range_bound":
            # Iron Condor — sell premium around the OR boundaries
            strategy = "Iron Condor"
            w = 5.0 if price > 100 else (2.0 if price > 50 else 1.0)
            sc_k, lc_k = or_high, or_high + w
            sp_k, lp_k = or_low, or_low - w

            if lp_k <= 0:   # stock too cheap to fit put spread below OR low
                return

            sc = _bs(price, sc_k, T, iv, "call")
            lc = _bs(price, lc_k, T, iv, "call")
            sp = _bs(price, sp_k, T, iv, "put")
            lp = _bs(price, lp_k, T, iv, "put")

            net_credit = (sc + sp) - (lc + lp)
            if net_credit <= 0:
                return
            entry_cost = -net_credit          # negative = we received premium
            risk = w - net_credit
            qty = max(1, int(self.config.position_size_usd / max(0.01, risk * 100)))
            meta = dict(sc_k=sc_k, lc_k=lc_k, sp_k=sp_k, lp_k=lp_k, width=w, qty=qty)

        elif iv <= self.config.iv_threshold:
            if signal == "breakout_above":
                if spy is True:
                    strategy = "Bull Call Spread"
                    k_long, k_short = or_high, or_high + or_range
                    lc = _bs(price, k_long, T, iv, "call")
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
                if spy is False:
                    strategy = "Bear Put Spread"
                    k_long, k_short = or_low, or_low - or_range
                    lp = _bs(price, k_long, T, iv, "put")
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
            f"ENTRY {strategy} | px={price:.2f} iv={iv:.1%}"
            f" cost=${entry_cost:.2f} qty={meta.get('qty',1)}"
            f" total=${entry_cost*meta.get('qty',1)*100:.0f}"
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
            # Cost to close = buy back short legs, sell long legs
            cost_to_close = (sc + sp) - (lc + lp)
            exit_value = -cost_to_close   # negative = we pay to close

        elif strategy == "Bull Call Spread":
            exit_value = _bs(price, meta["k_long"], T, iv, "call") - \
                         _bs(price, meta["k_short"], T, iv, "call")

        elif strategy == "Bear Put Spread":
            exit_value = _bs(price, meta["k_long"], T, iv, "put") - \
                         _bs(price, meta["k_short"], T, iv, "put")

        else:  # Straddle
            exit_value = _bs(price, meta["k"], T, iv, "call") + \
                         _bs(price, meta["k"], T, iv, "put")

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

        # Phase 1 — build opening range
        if not self._range_ready:
            if time(9, 30) <= bar_time < time(10, 0):
                h = bar.high.as_double()
                l = bar.low.as_double()
                self._or_high = max(self._or_high, h) if self._or_high else h
                self._or_low = min(self._or_low, l) if self._or_low else l
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
            strat = self._open_trade["strategy"]
            or_h = self._open_trade["or_high"]
            or_l = self._open_trade["or_low"]
            or_r = self._open_trade["or_range"]
            stop_buf = self.config.stop_buffer
            mult = self.config.profit_multiplier

            if strat in ("Bull Call Spread", "Straddle") and self._open_trade["entry_price"] > or_h:
                if close <= or_l - stop_buf:
                    self._exit(close, dt_et, "STOP")
                elif close >= or_h + or_r * mult:
                    self._exit(close, dt_et, "TARGET")

            elif strat in ("Bear Put Spread", "Straddle") and self._open_trade["entry_price"] < or_l:
                if close >= or_h + stop_buf:
                    self._exit(close, dt_et, "STOP")
                elif close <= or_l - or_r * mult:
                    self._exit(close, dt_et, "TARGET")

            elif strat == "Iron Condor":
                if close >= or_h or close <= or_l:
                    self._exit(close, dt_et, "STOP")
            return

        # Phase 4 — look for entry signal
        if self._traded:
            return

        avg_vol = self._or_vol_sum / max(self._or_bars, 1)
        vol_ok = bar.volume.as_double() >= avg_vol * self.config.volume_factor
        or_r = self._or_high - self._or_low
        after_1030 = bar_time >= time(10, 30)

        is_range = (
            after_1030
            and (self._or_low + 0.2 * or_r) <= close <= (self._or_high - 0.2 * or_r)
        )

        iv = _hv(self._closes)

        if close > self._or_high and vol_ok:
            self._enter(close, dt_et, "breakout_above", iv)
        elif close < self._or_low and vol_ok:
            self._enter(close, dt_et, "breakout_below", iv)
        elif is_range and iv > self.config.iv_threshold:
            self._enter(close, dt_et, "range_bound", iv)

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

    def on_save(self) -> dict[str, bytes]:
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        pass

    def on_dispose(self) -> None:
        pass
