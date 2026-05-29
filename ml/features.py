"""
Canonical feature definitions for the ORB Breakout Quality Scorer.

This is the single source of truth for feature names and computation.
Both the training pipeline (backtest) and live bot inference import from here
to guarantee the feature vector is identical at train and predict time.
"""
from datetime import datetime

# Ordered list of feature column names — order must never change after the
# first model is trained, because the pkl stores no column names.
FEATURE_COLS: list[str] = [
    "or_range_pct",       # OR width / OR high  — wider OR = more decisive open
    "volume_ratio",        # breakout-bar vol / avg OR vol — confirms conviction
    "breakout_pct",        # (entry - OR high) / OR high  — momentum at entry
    "minutes_after_open",  # minutes past 10:00 AM ET — earlier = cleaner signal
    "spy_trend_pct",       # (SPY last - SPY open) / SPY open — market tailwind
    "day_of_week",         # 0=Mon … 4=Fri — session character
]

ML_CONFIDENCE_THRESHOLD = 0.58   # default gate: skip entry if score < this


def compute_features(
    or_high: float,
    or_low: float,
    breakout_price: float,
    volume: float,
    avg_or_volume: float,
    bar_et: datetime,
    spy_open: float | None = None,
    spy_last: float | None = None,
) -> list[float]:
    """
    Compute the feature vector for a single breakout entry.
    Returns values in FEATURE_COLS order.

    Parameters
    ----------
    or_high, or_low     : opening-range high and low
    breakout_price      : close price of the breakout bar (entry price)
    volume              : volume of the breakout bar
    avg_or_volume       : mean volume of the OR bars
    bar_et              : bar timestamp in US/Eastern timezone
    spy_open, spy_last  : SPY session open and most-recent close (optional)
    """
    or_range = or_high - or_low
    spy_trend = (
        (spy_last - spy_open) / spy_open
        if (spy_open and spy_last and spy_open > 0)
        else 0.0
    )
    # minutes elapsed since 10:00 AM ET (when the OR window closes)
    minutes_after_open = (bar_et.hour - 10) * 60 + bar_et.minute

    return [
        or_range / or_high,                          # or_range_pct
        volume / max(avg_or_volume, 1.0),            # volume_ratio
        (breakout_price - or_high) / or_high,        # breakout_pct
        float(minutes_after_open),                    # minutes_after_open
        spy_trend,                                    # spy_trend_pct
        float(bar_et.weekday()),                      # day_of_week
    ]
