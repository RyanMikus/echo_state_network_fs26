"""
ML Feature Panel Construction (Spread-basiert, rollenorientiert).

Finales Feature-Set für intraday pairs trading mit Rolling-Window-Training.

Features (pro Pair):
- RSI_14_spread        (Signal)
- Z_60_spread          (State)
- VOL_20_spread        (Regime)
- return_pct_spread   (Timing)

Design:
- Leak-free (keine Zukunftsinformation)
- Skalenstabil über Paare
- RF- und XGBoost-kompatibel
- Keine Rolling-Window-Logik hier
"""

import numpy as np
import pandas as pd

from .technical import (
    rsi,
    rolling_volatility,
)


def build_feature_panel(
    log_prices: pd.DataFrame,
    pairs: list[tuple[str, str]],
    rsi_window: int = 14,
    z_window: int = 60,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Build a compact ML-ready feature panel on SPREAD level.

    Parameters
    ----------
    log_prices : pd.DataFrame
        Wide-format DataFrame of log prices (DatetimeIndex, symbols as columns).
        Expected to be built from close_filled.
    pairs : list of tuple(str, str)
        List of symbol pairs, e.g. [("AAPL", "MSFT"), ...]
    rsi_window : int
    z_window : int
    vol_window : int

    Returns
    -------
    pd.DataFrame
        Feature panel with MultiIndex columns:
        level 0: feature name
        level 1: pair name (e.g. AAPL_MSFT)
    """

    feature_blocks = []

    for s1, s2 in pairs:
        if s1 not in log_prices.columns or s2 not in log_prices.columns:
            continue

        pair_name = f"{s1}_{s2}"
        prices = log_prices[[s1, s2]].dropna()

        if len(prices) < max(rsi_window, z_window, vol_window):
            continue

        # -------------------------------------------------
        # Spread construction (static hedge ratio)
        # -------------------------------------------------
        beta, _ = np.polyfit(prices[s2].values, prices[s1].values, 1)
        spread = prices[s1] - beta * prices[s2]

        df = pd.DataFrame(index=spread.index)
        df["spread"] = spread

        # Wide-format requirement for technical indicators
        spread_wide = pd.concat([spread, spread], axis=1)
        spread_wide.columns = ["SPREAD", "SPREAD_DUMMY"]

        # -------------------------------------------------
        # Features
        # -------------------------------------------------

        # Signal
        df[f"RSI_{rsi_window}_spread"] = rsi(
            spread_wide, window=rsi_window
        )["SPREAD"]

        # State
        mean = spread.rolling(z_window).mean()
        std = spread.rolling(z_window).std().replace(0, np.nan)
        df[f"Z_{z_window}_spread"] = (spread - mean) / std

        # Regime
        df[f"VOL_{vol_window}_spread"] = rolling_volatility(
            spread_wide, window=vol_window
        )["SPREAD"]

        # Timing / Magnitude (skaleninvariant)
        prev_abs = spread.shift(1).abs().replace(0, np.nan)
        df["return_pct_spread"] = (
            (spread - spread.shift(1)) / prev_abs * 100.0
        )

        # -------------------------------------------------
        # Final formatting
        # -------------------------------------------------
        df = df.dropna()

        df.columns = pd.MultiIndex.from_product(
            (df.columns, [pair_name])
        )

        feature_blocks.append(df)

    if not feature_blocks:
        raise ValueError("No valid pair features could be constructed.")

    return (
        pd.concat(feature_blocks, axis=1)
        .sort_index(axis=1)
        .sort_index()
    )