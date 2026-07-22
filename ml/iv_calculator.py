"""
IV Calculator — American option IV solver, IV Rank, IV Skew, and Delta.

IV computation uses the Bjerksund-Stensland (2002) American-option pricer
from optlib (dbrojas/optlib, ml/gbs.py). This correctly accounts for the
early exercise premium in US equity options — the old Black-Scholes NR solver
used the European formula, which understates IV on puts and deep-ITM options.

Public API
----------
    compute_iv(market_price, S, K, T, r, option_type)  -> float | None   (legacy BS, kept as fallback)
    compute_american_iv(market_price, S, K, T, r, option_type, q) -> float | None
    compute_american_delta(S, K, T, r, iv, option_type, q)        -> float | None
    compute_iv_rank(symbol, current_iv)                -> float   (0–100)
    compute_iv_skew(calls, puts, current_price)        -> float   (signed)
    enrich_chain(calls, puts, S, T, r)                 -> (calls, puts)  adds amer_iv + delta cols
"""
import math
from functools import lru_cache

import pandas as pd
import yfinance as yf

try:
    from .gbs import american as _gbs_american, amer_implied_vol as _gbs_amer_iv
    _GBS_AVAILABLE = True
except ImportError:
    _GBS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RISK_FREE = 0.05          # annualized risk-free rate (approximate)
_NR_MAX_ITER = 100         # Newton-Raphson iteration cap
_NR_TOL = 1e-5             # convergence tolerance on price error
_NR_MIN_VEGA = 1e-10       # bail-out threshold for near-zero vega


# ---------------------------------------------------------------------------
# Internal Black-Scholes helpers
# ---------------------------------------------------------------------------

def _ncdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    """Black-Scholes theoretical price."""
    if T <= 0 or sigma <= 1e-8 or S <= 0 or K <= 0:
        intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
        return max(0.0, intrinsic)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    else:
        return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega = dPrice/dSigma — the Newton-Raphson step denominator."""
    if T <= 0 or sigma <= 1e-8 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * _npdf(d1) * math.sqrt(T)


# ---------------------------------------------------------------------------
# Newton-Raphson IV solver
# ---------------------------------------------------------------------------

def compute_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,          # "call" or "put"
    sigma0: float = 0.30,      # initial guess
) -> float | None:
    """
    Compute implied volatility from a market price using Newton-Raphson.

    Adapted from EthanFalcao/Defi_Options_Implied_Volatility (vs.py).
    Converges when |BS_price - market_price| < 1e-5 or vega < 1e-10.

    Returns None if the solver fails to converge (deep ITM/OTM, stale quote).
    """
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    sigma = sigma0
    for _ in range(_NR_MAX_ITER):
        price = _bs_price(S, K, T, r, sigma, option_type)
        diff  = market_price - price
        if abs(diff) < _NR_TOL:
            return sigma
        vega = _bs_vega(S, K, T, r, sigma)
        if abs(vega) < _NR_MIN_VEGA:
            return None                 # can't converge — vega too small
        sigma = sigma + diff / vega     # Newton step
        if sigma <= 0:
            sigma = 1e-4               # stay positive
        if sigma > 20.0:
            return None                 # diverged
    return sigma                        # return best estimate after max_iter


# ---------------------------------------------------------------------------
# American option IV + delta (Bjerksund-Stensland 2002 via ml/gbs.py)
# ---------------------------------------------------------------------------

def compute_american_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,   # "call" or "put"
    q: float = 0.0,     # continuous dividend yield
) -> float | None:
    """
    American-option implied volatility via bisection on the B-S 2002 pricer.

    Uses amer_implied_vol() from ml/gbs.py (optlib). Unlike Newton-Raphson,
    bisection doesn't rely on vega, so it's more robust for deep ITM puts
    where vega is near zero and the European formula understates early-exercise value.

    Falls back to European NR (compute_iv) when ml/gbs.py is not available.
    Returns None on invalid inputs or convergence failure.
    """
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    if not _GBS_AVAILABLE:
        return compute_iv(market_price, S, K, T, r, option_type)
    opt_type = "c" if option_type == "call" else "p"
    try:
        iv = _gbs_amer_iv(opt_type, S, K, T, r, q, market_price)
        return float(iv) if iv and iv > 0 else None
    except Exception:
        return compute_iv(market_price, S, K, T, r, option_type)


def compute_american_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    iv: float,
    option_type: str,   # "call" or "put"
    q: float = 0.0,
) -> float | None:
    """
    Delta from the Bjerksund-Stensland American-option pricer.

    Returns [1] element from american() — the partial derivative dV/dS.
    For calls: 0 < delta < 1  (higher = more ITM, higher probability of expiry ITM)
    For puts:  -1 < delta < 0 (more negative = more ITM)

    delta ≈ 0.15 call / -0.15 put → ~15% chance of expiring ITM → useful for
    Iron Condor short-strike selection (e.g. IC_SHORT_DELTA = 0.15).

    Returns None when ml/gbs.py is unavailable or pricing fails.
    """
    if not _GBS_AVAILABLE or iv is None or iv <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    opt_type = "c" if option_type == "call" else "p"
    try:
        result = _gbs_american(opt_type, S, K, T, r, q, iv)
        delta = float(result[1])
        return delta if math.isfinite(delta) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# IV Rank (0–100)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _fetch_hist_vol(symbol: str) -> tuple[float, float] | None:
    """
    Returns (hv_1y_low, hv_1y_high) using 30-day rolling realized vol as
    a historical IV proxy (yfinance doesn't provide historical IV directly).
    Result is LRU-cached per symbol per session to avoid repeated downloads.
    """
    try:
        hist = yf.Ticker(symbol).history(period="1y", interval="1d")
        if len(hist) < 30:
            return None
        rets = hist["Close"].pct_change().dropna()
        rolling_hv = rets.rolling(30).std() * math.sqrt(252)
        rolling_hv = rolling_hv.dropna()
        if rolling_hv.empty:
            return None
        return float(rolling_hv.min()), float(rolling_hv.max())
    except Exception:
        return None


def compute_iv_rank(symbol: str, current_iv: float) -> float:
    """
    IV Rank = (current_iv - 1y_low) / (1y_high - 1y_low) × 100

    Uses 30-day rolling realized volatility as the historical IV proxy.

    Returns 50.0 as a neutral default when historical data is unavailable.
    Interpretation:
        0–30   : IV historically cheap → prefer buying premium (debit spreads)
       30–70   : neutral regime
       70–100  : IV historically expensive → prefer selling premium (condors)
    """
    bounds = _fetch_hist_vol(symbol)
    if bounds is None:
        return 50.0
    iv_low, iv_high = bounds
    if iv_high <= iv_low:
        return 50.0
    rank = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
    return max(0.0, min(100.0, rank))


# ---------------------------------------------------------------------------
# IV Skew signal
# ---------------------------------------------------------------------------

def compute_iv_skew(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    current_price: float,
    otm_pct: float = 0.05,    # look 5% OTM for each side
) -> float:
    """
    IV Skew = OTM_put_IV - OTM_call_IV

    Positive (put skew)  : puts are expensive → market fears downside
                           → favour Bear Put Spread or Iron Condor
    Negative (call skew) : calls are expensive → market expects upside
                           → favour Bull Call Spread or Iron Condor
    Near zero            : balanced → Straddle or Iron Condor

    Uses yfinance's impliedVolatility column; falls back to 0.0 if unavailable.
    """
    try:
        otm_call_strike = current_price * (1 + otm_pct)
        otm_put_strike  = current_price * (1 - otm_pct)

        call_candidates = calls[calls["strike"] >= otm_call_strike]
        put_candidates  = puts[puts["strike"]  <= otm_put_strike]

        if call_candidates.empty or put_candidates.empty:
            return 0.0

        otm_call = call_candidates.iloc[
            (call_candidates["strike"] - otm_call_strike).abs().argsort()
        ].iloc[0]
        otm_put = put_candidates.iloc[
            (put_candidates["strike"] - otm_put_strike).abs().argsort()
        ].iloc[0]

        call_iv = float(otm_call["impliedVolatility"])
        put_iv  = float(otm_put["impliedVolatility"])

        return put_iv - call_iv          # positive = bearish skew
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Chain enrichment — add amer_iv and delta columns
# ---------------------------------------------------------------------------

def enrich_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    S: float,
    T: float,
    r: float = _RISK_FREE,
    q: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add 'amer_iv' and 'delta' columns to both DataFrames.

    amer_iv  — American IV via Bjerksund-Stensland bisection (ml/gbs.py).
               More accurate than Black-Scholes NR for puts and deep-ITM calls
               because it accounts for early exercise premium.
               Falls back to NR / yfinance IV when gbs.py is unavailable.

    delta    — dV/dS from the American pricer (range: call 0–1, put -1 to 0).
               Useful for delta-based Iron Condor strike selection, where
               abs(delta) ≤ IC_SHORT_DELTA (e.g. 0.15) means ~85% OTM probability.

    Legacy nr_iv column is also populated for backward compatibility.
    """
    def _enrich_row(row: pd.Series, kind: str) -> dict:
        K   = float(row["strike"])
        mid = (float(row.get("bid", 0)) + float(row.get("ask", 0))) / 2.0

        # American IV (preferred)
        amer_iv = compute_american_iv(mid, S, K, T, r, kind, q)
        if amer_iv is None or amer_iv <= 0:
            # NR fallback, then yfinance
            nr = compute_iv(mid, S, K, T, r, kind)
            amer_iv = nr if (nr and nr > 0) else float(row.get("impliedVolatility", 0.30))

        delta = compute_american_delta(S, K, T, r, amer_iv, kind, q)

        return {"amer_iv": amer_iv, "nr_iv": amer_iv, "delta": delta}

    calls = calls.copy()
    puts  = puts.copy()

    call_enriched = calls.apply(lambda row: pd.Series(_enrich_row(row, "call")), axis=1)
    put_enriched  = puts.apply(lambda row: pd.Series(_enrich_row(row, "put")),  axis=1)

    calls[["amer_iv", "nr_iv", "delta"]] = call_enriched
    puts[["amer_iv",  "nr_iv", "delta"]] = put_enriched

    return calls, puts
