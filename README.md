# Trading Bot — ORB Day Trading (Equity + Options)

Automated day-trading bot built on the **Alpaca** paper-trading API.  
Two parallel bots share the same ORB (Opening Range Breakout) signal but execute differently:

| Bot | File | Strategy |
|-----|------|----------|
| **Equity** | `orb_bot.py` | Buy stock on breakout, hold intraday, sell at target/stop/EOD |
| **Options** | `orb_options_bot.py` | Select an options structure (spread / straddle / iron condor) based on IV and direction |

---

## Strategy Overview

### Opening Range (shared)
The opening range is computed from the first 6 × 5-min bars (9:30–10:00 ET).

| Signal | Condition |
|--------|-----------|
| **Breakout above** | Close > OR high **AND** bar volume ≥ avg OR volume |
| **Breakout below** | Close < OR low **AND** bar volume ≥ avg OR volume |
| **Range-bound** | After 10:30 ET, price stays within the middle 60% of the OR |

A SPY trend filter (latest close vs. session open) biases directional trades.

---

### Equity bot (`orb_bot.py`)
- Entry: breakout above OR high, SPY trending up
- Stop: OR low − $0.05 buffer
- Target: OR high + (OR range × per-symbol multiplier)
- EOD: force-close all positions at 3:45 PM ET

### Options bot (`orb_options_bot.py`)
Strategy selection is driven by IV vs. the `ORB_OPTIONS_IV_THRESHOLD` (default 45%):

| Condition | Strategy |
|-----------|----------|
| High IV + range-bound | Iron Condor (sell premium around OR boundaries) |
| Low IV + breakout above + SPY up | Bull Call Spread |
| Low IV + breakout below + SPY down | Bear Put Spread |
| Low IV + breakout, SPY uncorrelated | Straddle |

Exits: stop at OR boundary breach (condor), P&L-based stop (spreads/straddle), EOD close.

---

## Project Structure

```
trading-bot/
├── orb_bot.py              # Equity ORB bot (GitHub Actions)
├── orb_options_bot.py      # Options ORB bot (GitHub Actions)
├── screener.py             # Yahoo Finance most-actives screener + static fallback
├── check_options.py        # One-off: verify Alpaca options access + sample chain
├── close_all.py            # One-off: emergency close all open positions
├── config.py               # All strategy parameters — tune here
├── strategies/
│   └── rsi_momentum.py     # Legacy RSI strategy (reference only)
├── backtest/
│   ├── run_orb_backtest.py # ORB backtest runner
│   ├── rank_symbols.py     # Ranks symbols by backtest P&L
│   └── results/
│       └── orb_ranking.csv # Latest backtest ranking
├── requirements.txt        # Bot dependencies (pinned)
└── .github/workflows/
    ├── trading-bot.yml         # Runs orb_bot.py every 5 min during market hours
    └── orb-options-bot.yml     # Runs orb_options_bot.py every 5 min during market hours
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set credentials

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

### 3. (Optional) Verify options access

```bash
python check_options.py
```

### 4. Run

```bash
# Equity bot
python orb_bot.py

# Options bot
python orb_options_bot.py
```

Both bots run automatically via GitHub Actions every 5 minutes during US market hours (Mon–Fri, 9:30am–4:30pm ET).

---

## Configuration (`config.py`)

### ORB shared settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ORB_RANGE_BARS` | 6 | Bars for opening range (6 × 5 min = 30 min) |
| `ORB_SCREENER_LIMIT` | 20 | Symbols fetched from Yahoo Finance screener |
| `ORB_SYMBOLS` | (list) | Static fallback if screener fails |
| `ORB_MIN_OR_PCT` | 0.5% | Skip symbols with a narrow, indecisive opening range |
| `ORB_VOLUME_FACTOR` | 1.0× | Breakout bar volume must be ≥ this × avg OR volume |
| `ORB_STOP_BUFFER` | $0.05 | Stop placed this far below/above OR boundary |
| `ORB_CLOSE_HOUR/MINUTE` | 15:45 | Force-close time (ET) |

### Equity bot settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ORB_POSITION_SIZE` | $500 | Dollar allocation per trade |
| `MAX_TOTAL_INVESTMENT` | $2000 | Total budget across all equity positions |
| `ORB_PROFIT_MULTIPLIER` | 1.5× | Default take-profit = OR range × this factor |
| `ORB_PROFIT_MULTIPLIERS` | (dict) | Per-symbol overrides |

### Options bot settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ORB_OPTIONS_POSITION_SIZE` | $500 | Dollar allocation per options strategy |
| `MAX_OPTIONS_INVESTMENT` | $2000 | Total budget across all options positions |
| `ORB_OPTIONS_IV_THRESHOLD` | 45% | Above = sell premium; below = buy premium |

---

## Secrets Setup (GitHub Actions)

Add these in your repo → **Settings → Secrets → Actions**:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

---

## Backtest Results (ORB, top performers)

| Rank | Symbol | Win Rate | Total P&L |
|------|--------|----------|-----------|
| 1 | META | 72.7% | +$1,858 |
| 2 | COIN | 100.0% | +$1,642 |
| 3 | AMD | 66.7% | +$1,367 |
| 4 | TSLA | 70.0% | +$1,286 |
| 5 | QQQ | 65.0% | +$1,122 |

Full ranking: [backtest/results/orb_ranking.csv](backtest/results/orb_ranking.csv)
