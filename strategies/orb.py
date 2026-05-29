"""
Opening Range Breakout (ORB) strategy — NautilusTrader implementation.

Logic
-----
Opening range : first N × 5-min bars (default 6 = 9:30–10:00 ET)
Entry         : 5-min close breaks ABOVE OR high AND bar volume >= avg OR volume × factor
Stop          : just below OR low (OR low - stop_buffer)
Target        : OR high + (OR range × profit_multiplier)
EOD close     : force-close any open position at or after close_hour:close_minute ET
One trade/day : no re-entry after an exit

Improvements (v2)
------------------
1. SPY trend filter   : only enter when SPY latest close > SPY session open
2. Min OR range filter : skip symbols where OR range / price < min_or_pct
3. Per-symbol multiplier passed in via config.profit_multiplier (caller looks up the dict)
"""
from datetime import date, datetime, time, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from nautilus_trader.config import PositiveFloat, PositiveInt, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

ET = ZoneInfo("America/New_York")


class ORBConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    position_size_usd: float = 1_000.0   # dollars per trade; qty = floor(size / or_high)
    orb_range_bars: PositiveInt = 6
    profit_multiplier: PositiveFloat = 1.5
    volume_factor: PositiveFloat = 1.0
    stop_buffer: float = 0.05
    close_hour: int = 15
    close_minute: int = 45
    # Improvement 1: subscribe to SPY bars for trend filter (None = disabled)
    spy_bar_type: BarType | None = None
    # Improvement 2: skip if (OR high - OR low) / OR high < this fraction (0 = disabled)
    min_or_pct: float = 0.005


class ORBStrategy(Strategy):
    def __init__(self, config: ORBConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self.trade_log: list[dict] = []        # populated during backtest
        self._entry_features: dict | None = None
        self._reset_daily_state()
        self._current_date: date | None = None
        self._spy_open: float | None = None
        self._spy_last: float | None = None
        self._spy_date: date | None = None

    def _reset_daily_state(self) -> None:
        self._or_high: float | None = None
        self._or_low: float | None = None
        self._or_volume_sum: float = 0.0
        self._or_bars_seen: int = 0
        self._range_ready: bool = False
        self._range_skipped: bool = False
        self._trade_qty: int = 0
        self._entry_price: float | None = None
        self._stop_price: float | None = None
        self._target_price: float | None = None
        self._traded_today: bool = False

    def _record_exit(self, price: float, reason: str) -> None:
        """Attach outcome to the pending entry features and commit to trade_log."""
        if self._entry_features is None:
            return
        self._entry_features.update({
            "hit_target": 1 if reason == "TARGET" else 0,
            "exit_reason": reason,
            "exit_price": price,
        })
        self.trade_log.append(dict(self._entry_features))
        self._entry_features = None

    def _bar_et_time(self, bar: Bar) -> tuple[date, time]:
        dt_utc = datetime.fromtimestamp(bar.ts_event / 1_000_000_000, tz=timezone.utc)
        dt_et = dt_utc.astimezone(ET)
        return dt_et.date(), dt_et.time()

    def _close_time(self) -> time:
        return time(self.config.close_hour, self.config.close_minute)

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return
        self.subscribe_bars(self.config.bar_type)
        if self.config.spy_bar_type is not None:
            self.subscribe_bars(self.config.spy_bar_type)

    def _handle_spy_bar(self, bar: Bar) -> None:
        bar_date, _ = self._bar_et_time(bar)
        if self._spy_date != bar_date:
            self._spy_date = bar_date
            self._spy_open = bar.open.as_double()
        self._spy_last = bar.close.as_double()

    def on_bar(self, bar: Bar) -> None:
        # Route SPY bars to their own handler
        if self.config.spy_bar_type is not None and bar.bar_type == self.config.spy_bar_type:
            self._handle_spy_bar(bar)
            return

        bar_date, bar_time = self._bar_et_time(bar)

        if self._current_date != bar_date:
            self._current_date = bar_date
            self._reset_daily_state()

        high = bar.high.as_double()
        low = bar.low.as_double()
        close = bar.close.as_double()
        volume = bar.volume.as_double()

        # ---- Phase 1: build opening range (bars before 10:00 ET) ----
        if not self._range_ready:
            if bar_time >= time(9, 30) and bar_time < time(10, 0):
                self._or_high = max(self._or_high, high) if self._or_high else high
                self._or_low = min(self._or_low, low) if self._or_low else low
                self._or_volume_sum += volume
                self._or_bars_seen += 1

                if self._or_bars_seen >= self.config.orb_range_bars:
                    self._range_ready = True
                    or_range = self._or_high - self._or_low
                    # Improvement 2: filter out narrow / indecisive opens
                    if self.config.min_or_pct > 0 and or_range / self._or_high < self.config.min_or_pct:
                        self._range_skipped = True
                        self.log.debug(
                            f"OR too narrow ({or_range / self._or_high:.3%} < {self.config.min_or_pct:.3%}) — skipping day"
                        )
                    else:
                        self._trade_qty = max(1, int(self.config.position_size_usd / self._or_high))
                        self.log.info(
                            f"OR ready | H={self._or_high:.2f}  L={self._or_low:.2f}"
                            f"  Range={or_range:.2f}  Qty={self._trade_qty}"
                        )
            return

        if self._range_skipped:
            return

        # ---- Phase 2: EOD forced close ----
        if bar_time >= self._close_time():
            if self.portfolio.is_net_long(self.config.instrument_id):
                self._record_exit(close, "EOD")
                self.close_all_positions(self.config.instrument_id)
                self.log.info(f"EOD CLOSE at {close:.2f}")
                self._traded_today = True
            return

        # ---- Phase 3: manage open position ----
        if self.portfolio.is_net_long(self.config.instrument_id):
            if close <= self._stop_price:
                self._record_exit(close, "STOP")
                self.close_all_positions(self.config.instrument_id)
                self.log.info(f"STOP LOSS {close:.2f}  (stop={self._stop_price:.2f})")
                self._traded_today = True
                return

            if close >= self._target_price:
                self._record_exit(close, "TARGET")
                self.close_all_positions(self.config.instrument_id)
                self.log.info(f"TAKE PROFIT {close:.2f}  (target={self._target_price:.2f})")
                self._traded_today = True
                return

            return

        # ---- Phase 4: look for breakout entry ----
        if self._traded_today or not self.portfolio.is_flat(self.config.instrument_id):
            return

        # Improvement 1: SPY trend filter — only enter on up-trending days
        if self.config.spy_bar_type is not None:
            if self._spy_open is None or self._spy_last is None:
                return
            if self._spy_last < self._spy_open:
                return

        avg_or_vol = self._or_volume_sum / max(self._or_bars_seen, 1)
        vol_ok = volume >= avg_or_vol * self.config.volume_factor

        if close > self._or_high and vol_ok:
            or_range = self._or_high - self._or_low
            self._stop_price = self._or_low - self.config.stop_buffer
            self._target_price = self._or_high + or_range * self.config.profit_multiplier

            # Capture entry features for ML training
            spy_trend = (
                (self._spy_last - self._spy_open) / self._spy_open
                if (self._spy_open and self._spy_last and self._spy_open > 0)
                else 0.0
            )
            self._entry_features = {
                "symbol":             str(self.config.instrument_id.symbol),
                "date":               bar_date.isoformat(),
                "entry_time":         bar_time.strftime("%H:%M"),
                "or_range_pct":       or_range / self._or_high,
                "volume_ratio":       volume / max(avg_or_vol, 1.0),
                "breakout_pct":       (close - self._or_high) / self._or_high,
                "minutes_after_open": (bar_time.hour - 10) * 60 + bar_time.minute,
                "spy_trend_pct":      spy_trend,
                "day_of_week":        float(bar_date.weekday()),
                "entry_price":        close,
            }

            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(Decimal(self._trade_qty)),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self._entry_price = close
            self.log.info(
                f"BUY BREAKOUT {close:.2f}  qty={self._trade_qty}"
                f"  stop={self._stop_price:.2f}"
                f"  target={self._target_price:.2f}"
                f"  vol_ratio={volume / avg_or_vol:.2f}x"
            )

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        if self.portfolio.is_net_long(self.config.instrument_id):
            self._record_exit(0.0, "ENGINE_STOP")
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.bar_type)
        if self.config.spy_bar_type is not None:
            self.unsubscribe_bars(self.config.spy_bar_type)

    def on_reset(self) -> None:
        self._reset_daily_state()
        self._current_date = None
        self._entry_features = None   # clear pending entry; trade_log is kept
        self._spy_open = None
        self._spy_last = None
        self._spy_date = None

    def on_save(self) -> dict[str, bytes]:
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        pass

    def on_dispose(self) -> None:
        pass
