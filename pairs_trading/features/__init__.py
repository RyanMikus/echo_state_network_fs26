"""
pairs_trading.features

Öffentliche Feature-API für das pairs_trading Projekt.

Dieses Subpackage ist bewusst in drei Ebenen gegliedert:

1) technical.py
   Primitive, symbolweise technische Indikatoren.
   - Keine Paare
   - Keine Spreads
   - Keine Visualisierung

2) feature_set.py
   Kompakte, ML-taugliche Feature-Panels auf Pair-Ebene.
   - Rollenorientiert (Signal / State / Regime / Timing)
   - Strukturiert für Modell-Input
   - Keine Plots

3) feature_eda.py
   Explorative Feature-Konstruktion und Visualisierung auf Pair-Ebene.
   - Intraday-fokussiert
   - Plot-lastig
   - Notebook-nahe Utilities

Wichtige Hinweise:
- Die öffentliche API wird ausschließlich über __all__ definiert.
- Refactorings dürfen die hier exportierten Namen nicht verändern.
"""

from .technical import (
    log_returns,

    # trend indicators
    sma,
    ema,
    macd,

    # momentum indicators
    rsi,
    roc,
    momentum,

    # volatility indicators
    rolling_volatility,
    bollinger_bands,
)

from .feature_set import build_feature_panel

from .feature_eda import (
    compute_pair_feature_df,
    plot_pairs_subplots,
)

__all__ = [
    # base
    "log_returns",

    # trend
    "sma",
    "ema",
    "macd",

    # momentum
    "rsi",
    "roc",
    "momentum",

    # volatility
    "rolling_volatility",
    "bollinger_bands",

    # feature collections (ML-ready)
    "build_feature_panel",

    # pair-level EDA
    "compute_pair_feature_df",
    "plot_pairs_subplots",
]