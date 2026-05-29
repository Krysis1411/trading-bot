"""
Market Regime Detector — adapted from trading00money/QuantTrading (core/regime_detector.py)

Classifies the current market environment using three independent metrics:
  1. Volatility Percentile  — where today's vol sits in its 1-year distribution
  2. Efficiency Ratio       — net price movement / total path (trend strength)
  3. Hurst Exponent         — <0.5 mean-reverting, >0.5 trending, ~0.5 random

Regime output drives strategy selection in the options bot:
  CRISIS   → no new debit spreads; reduce size on Iron Condors
  TRENDING → prefer debit spreads (directional plays)
  RANGING  → prefer Iron Condors (range-bound, sell premium)
  NORMAL   → use IV Rank as the primary selector (existing logic)

Public API
----------
    detect_regime(symbol, bars_df)  -> RegimeResult
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Regime = Literal["CRISIS", "TRENDING", "RANGING", "NORMAL"]


@dataclass
class RegimeResult:
    regime: Regime
    vol_percentile: float      # 0–100
    efficiency_ratio: float    # 0–1
    hurst: float               # 0–1
    confidence: float          # 0–1 composite

    def __str__(self) -> str:
        return (
            f"{self.regime} "
            f"(vol_pct={self.vol_percentile:.0f} "
            f"eff={self.efficiency_ratio:.2f} "
            f"hurst={self.hurst:.2f} "
            f"conf={self.confidence:.0%})"
        )


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def _rolling_hv(closes: pd.Series, window: int = 20) -> pd.Series:
    """Annualised historical volatility using log returns."""
    log_rets = np.log(closes / closes.shift(1)).dropna()
    return log_rets.rolling(window).std() * math.sqrt(252)


def volatility_percentile(bars: pd.DataFrame, window: int = 20) -> float:
    """
    Percentile (0–100) of today's 20-day HV in its 1-year distribution.
    ≥95 = crisis territory; ≤25 = historically quiet.
    """
    hvs = _rolling_hv(bars["close"], window).dropna()
    if len(hvs) < 2:
        return 50.0
    current = float(hvs.iloc[-1])
    return float((hvs < current).mean() * 100)


def efficiency_ratio(bars: pd.DataFrame, period: int = 20) -> float:
    """
    Kaufman Efficiency Ratio: |net move| / sum(|bar moves|).
    1.0 = perfectly trending; 0.0 = random walk.
    Adapted from QuantTrading core/regime_detector.py trend_strength metric.
    """
    if len(bars) < period + 1:
        return 0.5
    closes = bars["close"].iloc[-period - 1:]
    net_move = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))
    total_path = float(closes.diff().abs().sum())
    return net_move / total_path if total_path > 0 else 0.5


def hurst_exponent(bars: pd.DataFrame, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent via R/S analysis (simplified).
      H < 0.45  → mean-reverting (RANGING regime)
      H ≈ 0.50  → random walk (NORMAL)
      H > 0.55  → trending (TRENDING regime)

    Adapted from QuantTrading cycle_engine + Ehlers methodology.
    """
    if len(bars) < max_lag + 2:
        return 0.5
    log_rets = np.log(bars["close"] / bars["close"].shift(1)).dropna().values
    lags = range(2, min(max_lag, len(log_rets) // 2))
    if not lags:
        return 0.5
    tau = []
    for lag in lags:
        diff = log_rets[lag:] - log_rets[:-lag]
        std = np.std(diff)
        tau.append(std if std > 0 else 1e-10)
    log_lags = np.log(list(lags))
    log_tau  = np.log(tau)
    if len(log_lags) < 2:
        return 0.5
    poly = np.polyfit(log_lags, log_tau, 1)
    return float(poly[0] * 2.0)   # H = slope × 2


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def detect_regime(bars: pd.DataFrame) -> RegimeResult:
    """
    Classify the current market regime from a 5-min or daily bar DataFrame.
    Requires columns: open, high, low, close, volume.
    Needs at least 60 bars for reliable estimates.

    Decision logic (from QuantTrading regime_detector.py):
      CRISIS   : vol_percentile ≥ 95  (always overrides)
      TRENDING : efficiency_ratio ≥ 0.60 AND hurst > 0.55
      RANGING  : efficiency_ratio ≤ 0.20 OR hurst < 0.45
      NORMAL   : otherwise
    """
    if len(bars) < 30:
        return RegimeResult("NORMAL", 50.0, 0.5, 0.5, 0.5)

    vp  = volatility_percentile(bars)
    er  = efficiency_ratio(bars)
    h   = hurst_exponent(bars)

    # Clamp Hurst to [0, 1] — numerical instability on very short series
    h = max(0.0, min(1.0, h))

    if vp >= 95:
        regime     = "CRISIS"
        confidence = 0.90 + (vp - 95) / 100
    elif er >= 0.60 and h > 0.55:
        regime     = "TRENDING"
        confidence = er * 0.6 + (h - 0.5) * 0.4
    elif er <= 0.20 or h < 0.45:
        regime     = "RANGING"
        confidence = (1 - er) * 0.6 + max(0, 0.5 - h) * 0.4
    else:
        regime     = "NORMAL"
        confidence = 0.5

    return RegimeResult(
        regime=regime,
        vol_percentile=vp,
        efficiency_ratio=er,
        hurst=min(1.0, max(0.0, h)),
        confidence=min(1.0, confidence),
    )


# ---------------------------------------------------------------------------
# Daily trend filter (from MTF engine weighting hierarchy)
# ---------------------------------------------------------------------------

def daily_trend(bars_daily: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    """
    Returns 'BULLISH', 'BEARISH', or 'NEUTRAL' based on EMA crossover
    on daily bars.  Mirrors QuantTrading mtf_engine.py EMA-based direction.

    bars_daily must have at least `slow` rows and a 'close' column.
    """
    if bars_daily is None or len(bars_daily) < slow:
        return "NEUTRAL"
    close = bars_daily["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    last_fast = float(ema_fast.iloc[-1])
    last_slow = float(ema_slow.iloc[-1])
    if last_fast > last_slow * 1.001:
        return "BULLISH"
    elif last_fast < last_slow * 0.999:
        return "BEARISH"
    return "NEUTRAL"
