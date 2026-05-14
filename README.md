# Trading Bot — RSI + 200-day MA Strategy

Automated trading bot built on **NautilusTrader** (backtesting) and **Alpaca** (live paper trading).

## Strategy

| Signal | Condition |
|--------|-----------|
| **Buy**  | Hourly RSI < 35 **AND** price above 200-day MA (uptrend filter) |
| **Sell** | Hourly RSI > 65 **OR** position down ≥ 2% (stop loss) |

Symbols traded: AAPL, MSFT, GOOGL, NVDA, AMZN, META, SPY, QQQ, SOFI, HOOD, RBLX, DKNG, MARA, RIOT, IONQ

---

## Project Structure

```
trading-bot/
├── bot.py                     # Live trading entry point (runs via GitHub Actions)
├── config.py                  # All strategy parameters — edit this to tune the strategy
├── strategies/
│   └── rsi_momentum.py        # NautilusTrader Strategy class (backtest + live)
├── backtest/
│   ├── fetch_data.py          # Downloads OHLCV data via yfinance
│   └── run_backtest.py        # Runs a full NautilusTrader backtest
├── requirements.txt           # Live bot dependencies
└── requirements-backtest.txt  # Backtesting dependencies (heavier)
```

---

## Quick Start

### 1. Live Trading (Alpaca Paper)

```bash
pip install -r requirements.txt

# Set your Alpaca paper trading credentials
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret

python bot.py
```

The GitHub Actions workflow (`.github/workflows/run-bot.yml`) runs `bot.py` automatically every 5 minutes during US market hours (Mon–Fri, 9:30am–4pm ET).

### 2. Backtesting (NautilusTrader)

```bash
pip install -r requirements-backtest.txt

# Backtest a single symbol (downloads data automatically)
python -m backtest.run_backtest AAPL

# Backtest multiple symbols
python -m backtest.run_backtest AAPL MSFT SPY
```

The backtest downloads up to 2 years of hourly data and 10 years of daily data via yfinance, then runs the exact same strategy logic through NautilusTrader's event-driven engine.

---

## Configuration

All parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RSI_PERIOD` | 14 | RSI lookback period |
| `RSI_OVERSOLD` | 35 | Buy threshold |
| `RSI_OVERBOUGHT` | 65 | Sell threshold |
| `MA_TREND_PERIOD` | 200 | Trend filter MA period |
| `STOP_LOSS_PCT` | 2.0 | Max loss before exit (%) |
| `TRADE_QUANTITY` | 1 | Shares per trade (live) |
| `BACKTEST_STARTING_BALANCE` | 100,000 | Starting capital for backtests |

---

## How Backtesting Works

The backtest uses **NautilusTrader** — a professional-grade, Rust-accelerated event-driven engine:

1. Historical OHLCV data is fetched from Yahoo Finance via `yfinance`
2. Data is converted to NautilusTrader `Bar` objects using `BarDataWrangler`
3. The same `RSIMomentumStrategy` class used here runs identically in the backtest
4. After the run, the engine prints an account report, order fills, and positions report

The strategy in `strategies/rsi_momentum.py` uses NautilusTrader's built-in `RelativeStrengthIndex` and `SimpleMovingAverage` indicators — no manual pandas math.

---

## Secrets Setup (GitHub Actions)

Add these secrets in your repo → **Settings → Secrets → Actions**:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
