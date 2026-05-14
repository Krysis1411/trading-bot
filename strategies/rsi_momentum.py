from decimal import Decimal

from nautilus_trader.config import PositiveInt, StrategyConfig
from nautilus_trader.indicators.averages import SimpleMovingAverage
from nautilus_trader.indicators.momentum import RelativeStrengthIndex
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class RSIMomentumConfig(StrategyConfig, frozen=True):
    """
    RSI + 200-day MA momentum strategy config.

    Buys when hourly RSI is oversold and price is above the 200-day MA.
    Sells when RSI is overbought or the stop-loss threshold is breached.
    """

    instrument_id: InstrumentId
    hourly_bar_type: BarType
    daily_bar_type: BarType
    trade_size: Decimal
    rsi_period: PositiveInt = 14
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    ma_period: PositiveInt = 200
    stop_loss_pct: float = 2.0


class RSIMomentumStrategy(Strategy):
    """
    Long-only strategy: buy RSI oversold dips above the 200-day MA trend filter.

    Entry : hourly RSI < rsi_oversold AND close > 200-day SMA
    Exit  : hourly RSI > rsi_overbought  OR  position down >= stop_loss_pct %
    """

    def __init__(self, config: RSIMomentumConfig) -> None:
        super().__init__(config)
        self.instrument: Instrument | None = None
        self.rsi = RelativeStrengthIndex(config.rsi_period)
        self.ma200 = SimpleMovingAverage(config.ma_period)
        self._entry_price: float | None = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found")
            self.stop()
            return

        # Auto-update RSI on every hourly bar, MA on every daily bar
        self.register_indicator_for_bars(self.config.hourly_bar_type, self.rsi)
        self.register_indicator_for_bars(self.config.daily_bar_type, self.ma200)

        self.subscribe_bars(self.config.hourly_bar_type)
        self.subscribe_bars(self.config.daily_bar_type)

    def on_bar(self, bar: Bar) -> None:
        # Daily bars only update the MA — trading decisions happen on hourly bars
        if bar.bar_type != self.config.hourly_bar_type:
            return

        # Both indicators must be warmed up (14 hourly + 200 daily bars)
        if not self.indicators_initialized():
            return

        price = bar.close.as_double()
        rsi_val = self.rsi.value
        ma_val = self.ma200.value
        is_long = self.portfolio.is_net_long(self.config.instrument_id)
        is_flat = self.portfolio.is_flat(self.config.instrument_id)

        # --- Exit: stop loss ---
        if is_long and self._entry_price is not None:
            pnl_pct = (price - self._entry_price) / self._entry_price * 100
            if pnl_pct <= -self.config.stop_loss_pct:
                self.close_all_positions(self.config.instrument_id)
                self._entry_price = None
                self.log.info(f"STOP LOSS {price:.2f} | P&L: {pnl_pct:.2f}%")
                return

        # --- Exit: RSI overbought ---
        if is_long and rsi_val > self.config.rsi_overbought:
            self.close_all_positions(self.config.instrument_id)
            self._entry_price = None
            self.log.info(f"TAKE PROFIT {price:.2f} | RSI: {rsi_val:.2f}")
            return

        # --- Entry ---
        if is_flat and rsi_val < self.config.rsi_oversold and price > ma_val:
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(self.config.trade_size),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self._entry_price = price
            self.log.info(
                f"BUY {price:.2f} | RSI: {rsi_val:.2f} | MA200: {ma_val:.2f}"
            )

    def on_stop(self) -> None:
        self.cancel_all_orders(self.config.instrument_id)
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_bars(self.config.hourly_bar_type)
        self.unsubscribe_bars(self.config.daily_bar_type)

    def on_reset(self) -> None:
        self.rsi.reset()
        self.ma200.reset()
        self._entry_price = None

    def on_save(self) -> dict[str, bytes]:
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        pass

    def on_dispose(self) -> None:
        pass
