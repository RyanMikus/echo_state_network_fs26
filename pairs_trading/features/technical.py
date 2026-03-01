"""
Primitive technische Indikatoren auf Symbol-Ebene.

Scope:
- Reine mathematische Transformationen
- Arbeiten ausschließlich auf Wide-Format Log-Preisen
- Keine Paare, keine Spreads, keine Visualisierung

Designentscheidungen:
- Defensive Validierung der Inputs
- Funktionaler Stil
- Keine Seiteneffekte

Dieses Modul stellt die niedrigste Abstraktionsebene
im Feature-Stack dar.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _validate_wide_prices(log_prices: pd.DataFrame) -> None:
    """
    Interne Validierung für Wide-Format Log-Preis-DataFrames.
    """
    if not isinstance(log_prices.index, pd.DatetimeIndex):
        raise TypeError("log_prices.index must be a pandas DatetimeIndex (datetime).")
    if getattr(log_prices.columns, "nlevels", 1) != 1:
        raise ValueError("log_prices must have a single-level column index (symbols).")
    if log_prices.shape[1] < 2:
        raise ValueError("log_prices must contain at least 2 symbols (columns).")


def log_returns(log_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns from log prices:
        r_t = log(P_t) - log(P_{t-1})
    """
    _validate_wide_prices(log_prices)
    return log_prices.diff()


# -----------------------------
# Trend indicators
# -----------------------------

def sma(
    log_prices: pd.DataFrame,
    window: int = 20,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simple Moving Average over log prices.
    """
    _validate_wide_prices(log_prices)
    if window <= 0:
        raise ValueError("window must be > 0")
    mp = min_periods if min_periods is not None else window
    return log_prices.rolling(window=window, min_periods=mp).mean()


def ema(log_prices: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """
    Exponential Moving Average over log prices.
    Uses adjust=False (common in finance).
    """
    _validate_wide_prices(log_prices)
    if span <= 0:
        raise ValueError("span must be > 0")
    return log_prices.ewm(span=span, adjust=False).mean()


def macd(
    log_prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    MACD based on log prices.
    """
    _validate_wide_prices(log_prices)
    if not (0 < fast < slow):
        raise ValueError("Require 0 < fast < slow")
    if signal <= 0:
        raise ValueError("signal must be > 0")

    ema_fast = log_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = log_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -----------------------------
# Momentum indicators
# -----------------------------

def roc(
    log_prices: pd.DataFrame,
    window: int = 10,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rate of Change (ROC) on PRICE reconstructed from log prices.
    """
    _validate_wide_prices(log_prices)
    if window <= 0:
        raise ValueError("window must be > 0")
    mp = min_periods if min_periods is not None else window

    diff = log_prices - log_prices.shift(window)
    out = np.exp(diff) - 1.0

    enough = log_prices.rolling(window=window, min_periods=mp).count() >= mp
    return out.where(enough)


def momentum(log_prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Momentum on log prices:
        M = logP_t - logP_{t-window}
    """
    _validate_wide_prices(log_prices)
    if window <= 0:
        raise ValueError("window must be > 0")
    return log_prices - log_prices.shift(window)


def rsi(
    log_prices: pd.DataFrame,
    window: int = 14,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Relative Strength Index (RSI) computed from log price deltas.
    """
    _validate_wide_prices(log_prices)
    if window <= 0:
        raise ValueError("window must be > 0")
    mp = min_periods if min_periods is not None else window

    delta = log_prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window=window, min_periods=mp).mean()
    avg_loss = loss.rolling(window=window, min_periods=mp).mean()

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# -----------------------------
# Volatility indicators
# -----------------------------

def rolling_volatility(
    log_prices: pd.DataFrame,
    window: int = 20,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling standard deviation of log returns.
    """
    _validate_wide_prices(log_prices)
    if window <= 0:
        raise ValueError("window must be > 0")
    mp = min_periods if min_periods is not None else window
    rets = log_prices.diff()
    return rets.rolling(window=window, min_periods=mp).std()


def bollinger_bands(
    log_prices: pd.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
    min_periods: Optional[int] = None,
):
    """
    Bollinger Bands over log prices.
    """
    _validate_wide_prices(log_prices)
    if window <= 0:
        raise ValueError("window must be > 0")
    if n_std <= 0:
        raise ValueError("n_std must be > 0")
    mp = min_periods if min_periods is not None else window

    mid = log_prices.rolling(window=window, min_periods=mp).mean()
    sd = log_prices.rolling(window=window, min_periods=mp).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return mid, upper, lower