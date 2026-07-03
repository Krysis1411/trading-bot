"""
India Combined Strategy — ORB + VWAP + EMA Confluence.

Three independent signals must agree before entering a trade:

  Signal 1 — ORB direction bias  (09:15–09:45 IST)
      After the opening range is built, the direction is "long" if price
      closes above OR high with volume confirmation, "short" if below OR low.
      Once set, the direction stays for the whole session (it is a daily bias,
      not a per-bar trigger).

  Signal 2 — EMA trend alignment
      9-period EMA vs 21-period EMA on 5-min bars.
      Long when 9EMA > 21EMA; short when 9EMA < 21EMA.
      Requires at least 9 bars of data before the signal is trusted.

  Signal 3 — VWAP proximity  (pullback entry)
      VWAP is calculated from the first bar of each day using the standard
      formula: VWAP = Σ(typical_price × volume) / Σ(volume).
      Standard deviation bands are derived from the running price-variance.
      Signal fires when price is within ±vwap_band_sigma of VWAP — meaning
      price has pulled back from the OR breakout toward the VWAP anchor, giving
      a better entry than chasing the initial breakout.

Entry rules:
  • Minimum 2 of 3 signals agree → enter at standard position size
  • All 3 signals agree           → enter at boosted position size (×1.5)
  • Only 1 signal                 → no trade

Exit rules  (same as india_orb.py):
  • Target  : OR high + OR range × profit_multiplier  (long)
  • Stop    : OR low  × (1 − stop_buffer_pct)          (long)
  • Trailing: stop moves to breakeven at 0.5× target
  • EOD     : force-close at INDIA_CLOSE_HOUR:INDIA_CLOSE_MINUTE IST
"""
from __future__ import annotations

import math
from datetime import date, datetime, time as _time, timezone
from zoneinfo import ZoneInfo

from nautilus_trader.config import PositiveFloat, PositiveInt, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from config import (
    INDIA_ALLOW_SHORTS,
    INDIA_CLOSE_HOUR,
    INDIA_CLOSE_MINUTE,
    INDIA_MAX_ENTRY_HOUR,
    INDIA_MAX_ENTRY_MINUTE,
    INDIA_ORB_MAX_OR_PCT,
    INDIA_ORB_MIN_OR_PCT,
    INDIA_ORB_PROFIT_MULTIPLIER,
    INDIA_ORB_RANGE_BARS,
    INDIA_ORB_STOP_BUFFER_PCT,
    INDIA_ORB_VOLUME_FACTOR,
    INDIA_POSITION_SIZE_INR,
    INDIA_SKIP_MONDAY_ENTRIES,
)

IST = ZoneInfo("Asia/Kolkata")


def _ist_to_utc_time(hour: int, minute: int) -> _time:
    total = hour * 60 + minute - 330
    return _time(total // 60, total % 60)


_NSE_OPEN_UTC    = _ist_to_utc_time(9, 15)
_ORB_END_UTC     = _ist_to_utc_time(9, 15 + INDIA_ORB_RANGE_BARS * 5)
_ENTRY_CUTOFF_UTC = _ist_to_utc_time(INDIA_MAX_ENTRY_HOUR, INDIA_MAX_ENTRY_MINUTE)
_EOD_CLOSE_UTC   = _ist_to_utc_time(INDIA_CLOSE_HOUR, INDIA_CLOSE_MINUTE)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class IndiaCombinedConfig(StrategyConfig, frozen=True):
    instrument_id:        InstrumentId
    bar_type:             BarType
    position_size_inr:    float         = float(INDIA_POSITION_SIZE_INR)
    orb_range_bars:       PositiveInt   = INDIA_ORB_RANGE_BARS
    profit_multiplier:    PositiveFloat = INDIA_ORB_PROFIT_MULTIPLIER
    volume_factor:        PositiveFloat = INDIA_ORB_VOLUME_FACTOR
    stop_buffer_pct:      float         = INDIA_ORB_STOP_BUFFER_PCT
    min_or_pct:           float         = INDIA_ORB_MIN_OR_PCT
    max_or_pct:           float         = INDIA_ORB_MAX_OR_PCT
    allow_shorts:         bool          = INDIA_ALLOW_SHORTS
    trailing_stop:        bool          = True
    nifty_bar_type:       BarType | None = None

    # Combined-strategy parameters
    ema_short_period:     int   = 9     # fast EMA period
    ema_long_period:      int   = 21    # slow EMA period
    vwap_band_sigma:      float = 1.0   # entry within ±N σ of VWAP
    min_signals:          int   = 2     # minimum signals (including ORB) to enter
    position_size_boost:  float = 1.5   # size multiplier when all 3 signals agree


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class IndiaCombinedStrategy(Strategy):
    """
    NautilusTrader strategy: ORB direction + VWAP proximity + EMA alignment.
    No real orders — results tracked in self.trades.
    """

    def __init__(self, config: IndiaCombinedConfig) -> None:
        super().__init__(config)
        self.trades: list[dict] = []

        # Daily ORB state
        self._current_date:  date | None = None
        self._or_bars_seen:  int  = 0
        self._or_high:       float | None = None
        self._or_low:        float | None = None
        self._or_vol_sum:    float = 0.0
        self._avg_or_vol:    float = 0.0
        self._range_ready:   bool = False
        self._range_skip:    bool = False
        self._traded:        bool = False
        self._orb_direction: str | None = None   # "long" | "short" | None

        # Open trade state
        self._open_trade: dict | None = None

        # Running VWAP state (reset each day at 09:15)
        self._vwap_cum_vol:  float = 0.0   # Σ volume
        self._vwap_cum_pv:   float = 0.0   # Σ (typical_price × volume)
        self._vwap_cum_pv2:  float = 0.0   # Σ (typical_price² × volume)
        self._vwap:          float | None = None
        self._vwap_std:      float = 0.0

        # Running EMA state — NOT reset daily; cross-day EMA gives more stable signal
        # than session-scoped EMA which barely warms up before the ORB window closes.
        self._ema_short:      float | None = None
        self._ema_long:       float | None = None
        self._ema_bars_seen:  int = 0
        self._alpha_short:    float = 2 / (config.ema_short_period + 1)
        self._alpha_long:     float = 2 / (config.ema_long_period  + 1)

        # Nifty trend
        self._nifty_open: float | None = None
        self._nifty_last: float | None = None
        self._nifty_date: date | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self.subscribe_bars(self.config.bar_type)
        if self.config.nifty_bar_type is not None:
            self.subscribe_bars(self.config.nifty_bar_type)

    def on_stop(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _reset_day(self) -> None:
        self._or_bars_seen  = 0
        self._or_high       = None
        self._or_low        = None
        self._or_vol_sum    = 0.0
        self._avg_or_vol    = 0.0
        self._range_ready   = False
        self._range_skip    = False
        self._traded        = False
        self._orb_direction = None
        # VWAP resets each session (calculated from 09:15 on the current day)
        self._vwap_cum_vol  = 0.0
        self._vwap_cum_pv   = 0.0
        self._vwap_cum_pv2  = 0.0
        self._vwap          = None
        self._vwap_std      = 0.0
        # EMAs are NOT reset here — they accumulate across days for stability

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def _bar_utc(self, bar: Bar) -> datetime:
        return datetime.fromtimestamp(bar.ts_event / 1_000_000_000, tz=timezone.utc)

    def _bar_ist(self, bar: Bar) -> datetime:
        return self._bar_utc(bar).astimezone(IST)

    # ------------------------------------------------------------------
    # Nifty trend
    # ------------------------------------------------------------------

    def _handle_nifty(self, bar: Bar) -> None:
        dt = self._bar_ist(bar)
        if self._nifty_date != dt.date():
            self._nifty_date = dt.date()
            self._nifty_open = float(bar.open)
        self._nifty_last = float(bar.close)

    def _nifty_up(self) -> bool | None:
        if self.config.nifty_bar_type is None:
            return True
        if self._nifty_open is None or self._nifty_last is None:
            return None
        return self._nifty_last >= self._nifty_open

    # ------------------------------------------------------------------
    # VWAP update  (called every bar, including OR bars)
    # ------------------------------------------------------------------

    def _update_vwap(self, bar: Bar) -> None:
        high   = float(bar.high)
        low    = float(bar.low)
        close  = float(bar.close)
        volume = float(bar.volume)
        if volume <= 0:
            return
        tp = (high + low + close) / 3.0
        self._vwap_cum_vol  += volume
        self._vwap_cum_pv   += tp * volume
        self._vwap_cum_pv2  += tp * tp * volume
        self._vwap   = self._vwap_cum_pv / self._vwap_cum_vol
        variance     = self._vwap_cum_pv2 / self._vwap_cum_vol - self._vwap ** 2
        self._vwap_std = math.sqrt(max(0.0, variance))

    # ------------------------------------------------------------------
    # EMA update  (called every bar, including OR bars)
    # ------------------------------------------------------------------

    def _update_ema(self, close: float) -> None:
        if self._ema_short is None:
            self._ema_short = close
            self._ema_long  = close
        else:
            self._ema_short = self._alpha_short * close + (1 - self._alpha_short) * self._ema_short
            self._ema_long  = self._alpha_long  * close + (1 - self._alpha_long)  * self._ema_long
        self._ema_bars_seen += 1

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def _in_vwap_zone(self, close: float) -> bool:
        """True if price is within ±vwap_band_sigma of current VWAP."""
        if self._vwap is None or self._vwap_std <= 0:
            return False
        sigma = self.config.vwap_band_sigma
        return (self._vwap - self._vwap_std * sigma) <= close <= (self._vwap + self._vwap_std * sigma)

    def _ema_agrees(self, direction: str) -> bool:
        """True if cross-day EMA is aligned with direction (requires warmup)."""
        if self._ema_bars_seen < self.config.ema_short_period or \
           self._ema_short is None or self._ema_long is None:
            return False
        if direction == "long":
            return self._ema_short > self._ema_long
        return self._ema_short < self._ema_long

    # ------------------------------------------------------------------
    # Trade recording
    # ------------------------------------------------------------------

    def _open_position(
        self, bar: Bar, price: float, stop: float,
        target: float, qty: int, direction: str,
        signals: int, entry_type: str,
    ) -> None:
        self._open_trade = dict(
            direction=direction,
            entry_price=price,
            stop=stop,
            target=target,
            qty=qty,
            trailing_activated=False,
            entry_ts=bar.ts_event,
            entry_ist=self._bar_ist(bar).strftime("%H:%M"),
            entry_weekday=self._bar_ist(bar).strftime("%A"),
            or_high=self._or_high,
            or_low=self._or_low,
            or_range=self._or_high - self._or_low,
            signals=signals,
            entry_type=entry_type,
            vwap_at_entry=round(self._vwap, 2) if self._vwap else None,
            vwap_std_at_entry=round(self._vwap_std, 4),
        )
        self._traded = True

    def _exit_trade(self, bar: Bar, exit_price: float, reason: str) -> None:
        if self._open_trade is None:
            return
        t = self._open_trade
        if t["direction"] == "short":
            pnl     = round((t["entry_price"] - exit_price) * t["qty"], 2)
            pnl_pct = round((t["entry_price"] - exit_price) / t["entry_price"] * 100, 3)
        else:
            pnl     = round((exit_price - t["entry_price"]) * t["qty"], 2)
            pnl_pct = round((exit_price - t["entry_price"]) / t["entry_price"] * 100, 3)
        self.trades.append({
            "symbol":           str(self.config.instrument_id.symbol),
            "direction":        t["direction"],
            "entry_price":      t["entry_price"],
            "exit_price":       exit_price,
            "qty":              t["qty"],
            "pnl":              pnl,
            "pnl_pct":          pnl_pct,
            "exit_reason":      reason,
            "entry_time_ist":   t["entry_ist"],
            "entry_weekday":    t["entry_weekday"],
            "or_range":         t["or_range"],
            "or_range_pct":     round(t["or_range"] / t["or_high"] * 100, 3),
            "entry_ts":         t["entry_ts"],
            "exit_ts":          bar.ts_event,
            "signals":          t["signals"],
            "entry_type":       t["entry_type"],
            "vwap_at_entry":    t["vwap_at_entry"],
        })
        self._open_trade = None

    # ------------------------------------------------------------------
    # Main bar handler
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if self.config.nifty_bar_type and bar.bar_type == self.config.nifty_bar_type:
            self._handle_nifty(bar)
            return

        dt_utc = self._bar_utc(bar)
        t_utc  = dt_utc.time().replace(second=0, microsecond=0)
        dt_ist = dt_utc.astimezone(IST)
        today  = dt_ist.date()
        close  = float(bar.close)
        volume = float(bar.volume)

        # ── New trading day ────────────────────────────────────────────
        if today != self._current_date:
            self._current_date = today
            self._reset_day()

        # ── Skip pre-market bars ───────────────────────────────────────
        if t_utc < _NSE_OPEN_UTC:
            return

        # Always update VWAP and EMA (even during OR window and after entry)
        self._update_vwap(bar)
        self._update_ema(close)

        # ── EOD forced close ───────────────────────────────────────────
        if t_utc >= _EOD_CLOSE_UTC:
            if self._open_trade is not None:
                self._exit_trade(bar, close, "eod")
            return

        # ── Phase 1: accumulate ORB bars (09:15–09:45 IST) ────────────
        if t_utc < _ORB_END_UTC:
            if self._or_bars_seen == 0:
                self._or_high = float(bar.high)
                self._or_low  = float(bar.low)
            else:
                self._or_high = max(self._or_high, float(bar.high))
                self._or_low  = min(self._or_low,  float(bar.low))
            self._or_vol_sum  += volume
            self._or_bars_seen += 1
            return

        # ── Phase 2: finalise ORB on first post-window bar ────────────
        if not self._range_ready and not self._range_skip:
            if self._or_bars_seen < self.config.orb_range_bars:
                self._range_skip = True
                return
            or_range = self._or_high - self._or_low
            or_pct   = or_range / self._or_high if self._or_high > 0 else 0
            if or_pct < self.config.min_or_pct or (self.config.max_or_pct > 0 and or_pct > self.config.max_or_pct):
                self._range_skip = True
                return
            self._range_ready  = True
            self._avg_or_vol   = self._or_vol_sum / self._or_bars_seen

        if self._range_skip:
            return

        or_range = self._or_high - self._or_low

        # ── Phase 3: manage open position ─────────────────────────────
        if self._open_trade is not None:
            t        = self._open_trade
            is_long  = t["direction"] == "long"
            is_short = t["direction"] == "short"

            if self.config.trailing_stop and not t["trailing_activated"]:
                if is_long:
                    half_target = t["or_high"] + or_range * self.config.profit_multiplier * 0.5
                    if close >= half_target:
                        t["stop"] = max(t["stop"], t["entry_price"])
                        t["trailing_activated"] = True
                else:
                    half_target = t["or_low"] - or_range * self.config.profit_multiplier * 0.5
                    if close <= half_target:
                        t["stop"] = min(t["stop"], t["entry_price"])
                        t["trailing_activated"] = True

            if is_long:
                if close <= t["stop"]:
                    self._exit_trade(bar, close, "trailing_stop" if t["trailing_activated"] else "stop_loss")
                elif close >= t["target"]:
                    self._exit_trade(bar, close, "take_profit")
            else:
                if close >= t["stop"]:
                    self._exit_trade(bar, close, "trailing_stop" if t["trailing_activated"] else "stop_loss")
                elif close <= t["target"]:
                    self._exit_trade(bar, close, "take_profit")
            return

        # ── Phase 4: entry checks ──────────────────────────────────────
        if self._traded:
            return

        if INDIA_SKIP_MONDAY_ENTRIES and dt_ist.weekday() == 0:
            return

        if t_utc >= _ENTRY_CUTOFF_UTC:
            return

        vol_ok = volume >= self._avg_or_vol * self.config.volume_factor
        if not vol_ok:
            return

        nifty = self._nifty_up()

        # ── Set ORB direction bias on the first breakout bar ──────────
        # Direction is set but we do NOT enter here — we wait for VWAP proximity.
        # This avoids chasing the initial breakout which typically fires far from VWAP.
        if self._orb_direction is None:
            if close > self._or_high and nifty is not False:
                self._orb_direction = "long"
            elif close < self._or_low and self.config.allow_shorts and nifty is not True:
                self._orb_direction = "short"
            return   # never enter on the direction-setting bar itself

        # ── Gate 1: VWAP proximity (REQUIRED) ─────────────────────────
        # All entries must be near VWAP — this is the pullback entry.
        # If price is still far from VWAP, wait.
        if not self._in_vwap_zone(close):
            return

        # ── Gate 2: direction must still make sense ────────────────────
        # If direction is long, price must be above VWAP (not crashed through it)
        direction = self._orb_direction
        if direction == "long"  and self._vwap is not None and close < self._vwap - self._vwap_std:
            return   # blown through VWAP — trade invalidated
        if direction == "short" and self._vwap is not None and close > self._vwap + self._vwap_std:
            return   # blown through VWAP — trade invalidated

        # ── Count confirmed signals ────────────────────────────────────
        # Signal 1: ORB direction set (always — we're past the None check above)
        # Signal 2: VWAP proximity (always — we passed _in_vwap_zone above)
        # Signal 3: EMA aligned (optional booster)
        n_signals = 2  # ORB + VWAP are both confirmed
        ema_ok = self._ema_agrees(direction)
        if ema_ok:
            n_signals += 1

        # ── Entry ─────────────────────────────────────────────────────
        if direction == "long":
            long_stop   = self._or_low  * (1 - self.config.stop_buffer_pct)
            long_target = self._or_high + or_range * self.config.profit_multiplier
            if close >= long_target:
                return   # stale — target already passed

            boost    = n_signals >= 3
            raw_size = self.config.position_size_inr * (self.config.position_size_boost if boost else 1.0)
            qty      = max(1, int(raw_size / close))
            entry_type = "vwap_ema_confluence" if boost else "vwap_pullback"
            self._open_position(bar, close, long_stop, long_target, qty, "long", n_signals, entry_type)

        elif direction == "short":
            short_stop   = self._or_high * (1 + self.config.stop_buffer_pct)
            short_target = self._or_low  - or_range * self.config.profit_multiplier
            if close <= short_target or short_target <= 0:
                return   # stale

            boost    = n_signals >= 3
            raw_size = self.config.position_size_inr * (self.config.position_size_boost if boost else 1.0)
            qty      = max(1, int(raw_size / close))
            entry_type = "vwap_ema_confluence" if boost else "vwap_pullback"
            self._open_position(bar, close, short_stop, short_target, qty, "short", n_signals, entry_type)
