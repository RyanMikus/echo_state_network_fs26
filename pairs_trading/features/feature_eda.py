"""
Pair-level EDA utilities for intraday feature exploration.

Scope:
- Spread-basierte Feature-Konstruktion pro Pair (EDA)
- Visualisierung von Feature-Crossplots über mehrere Paare

Wichtige Hinweise:
- Dieses Modul ist explizit EDA-orientiert.
- Alle Berechnungen und Plots sind bewusst notebook-nah.
- Der Code ist 1:1 aus dem Notebook extrahiert (keine Änderungen).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_pair_feature_df(prices_pair):
    """
    Berechnet Spread + Spread-Features für ein einzelnes Pair.

    Parameter:
    - prices_pair: DataFrame mit genau zwei Spalten (log prices)

    Rückgabe:
    - DataFrame mit Spread, technischen Features und Plot-Hilfsspalten
    """
    s1, s2 = prices_pair.columns
    beta, _ = np.polyfit(prices_pair[s2].values, prices_pair[s1].values, 1)
    spread = prices_pair[s1] - beta * prices_pair[s2]

    df = pd.DataFrame(index=prices_pair.index)
    df["spread"] = spread

    prev_abs = df["spread"].shift(1).abs().replace(0, np.nan)
    df["return_pct_spread"] = (df["spread"] - df["spread"].shift(1)) / prev_abs * 100

    # Package braucht >=2 Spalten
    spread_wide = pd.concat([df["spread"], df["spread"]], axis=1)
    spread_wide.columns = ["SPREAD", "SPREAD_DUMMY"]

    from .technical import (
        rsi,
        roc,
        momentum,
        rolling_volatility,
        bollinger_bands,
    )

    RSI_W = 14
    Z_W   = 60
    VOL_W = 20
    ROC_W = 10
    MOM_W = 10
    BB_W  = 20

    df[f"RSI_{RSI_W}_spread"] = rsi(spread_wide, window=RSI_W)["SPREAD"]
    df[f"ROC_{ROC_W}_spread"] = roc(spread_wide, window=ROC_W)["SPREAD"]
    df[f"MOM_{MOM_W}_spread"] = momentum(spread_wide, window=MOM_W)["SPREAD"]
    df[f"VOL_{VOL_W}_spread"] = rolling_volatility(spread_wide, window=VOL_W)["SPREAD"]

    mid, upper, lower = bollinger_bands(spread_wide, window=BB_W)
    df[f"BB_WIDTH_{BB_W}_spread"] = upper["SPREAD"] - lower["SPREAD"]

    roll_mean = df["spread"].rolling(Z_W, min_periods=1).mean()
    roll_std  = df["spread"].rolling(Z_W, min_periods=1).std().replace(0, np.nan)
    df[f"zscore_{Z_W}"] = (df["spread"] - roll_mean) / roll_std

    df["dir"] = np.sign(df["spread"].diff()).map({1: "up", -1: "down"}).fillna("flat")

    abs_move = df["return_pct_spread"].abs().fillna(0)
    scale = np.nanpercentile(abs_move, 90) or 1.0
    df["size"] = np.clip(abs_move / scale * 400, 20, 400)

    return df.replace([np.inf, -np.inf], np.nan)


def plot_pairs_subplots(pair_data_dict, x_col, y_col, title):
    """
    Scatterplots pro Pair (ein Subplot je Pair).

    Parameter:
    - pair_data_dict: dict[(s1, s2) -> DataFrame]
    - x_col, y_col: Feature-Spalten
    - title: Figure-Titel
    """
    color_map = {"up": "green", "down": "red", "flat": "gray"}

    n_pairs = len(pair_data_dict)
    ncols = 3
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    i = 0
    for (s1, s2), dfp in pair_data_dict.items():
        ax = axes[i]
        i += 1
        ax.scatter(
            dfp[x_col],
            dfp[y_col],
            s=dfp["size"],
            c=dfp["dir"].map(color_map),
            alpha=0.6,
            edgecolors="w",
            linewidth=0.2,
        )
        ax.set_title(f"{s1} – {s2}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(alpha=0.2)

    while i < len(axes):
        axes[i].axis("off")
        i += 1

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()