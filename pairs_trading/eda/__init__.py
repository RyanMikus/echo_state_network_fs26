"""
pairs_trading.eda

Public API für explorative Datenanalyse (EDA) im pairs_trading Projekt.

Dieses Subpackage bündelt:
- deskriptive Zeitreihen-EDA (returns, Volatilität, Intraday-Profile)
- Korrelationsanalyse
- Cointegration-basierte Paaranalyse inkl. Visualisierung

Wichtige Hinweise:
- Dieses Package ist explizit explorativ ausgelegt.
- Plot-Ausgaben (Inhalt, Layout, Reihenfolge) sind Teil des beobachtbaren Verhaltens.
- Refactorings innerhalb der Module dürfen die hier definierten Importpfade,
  Funktionsnamen und Side-Effects nicht brechen.

Die nach außen sichtbare Oberfläche wird ausschließlich über __all__ definiert.
"""

from .descriptive_time_series import (
    compute_returns,
    cross_section_return_stats,
    rolling_volatility,
    intraday_volatility_profile,
    select_representative_symbols,
    plot_cross_section_hist,
    plot_cross_section_box,
    plot_intraday_profile,
    plot_representative_time_series,
    run_descriptive_eda,
)

from .correlation import correlation_analysis

from .cointegration import (
    compute_daily_cointegration,
    compute_pair_stats,
    rank_pairs,
    plot_daily_pvalues,
    plot_coint_network,
    compute_daily_spread_stats,
    plot_daily_mean_spreads,
    plot_intraday_prices_last_day,
    plot_intraday_spreads_last_day,
)

__all__ = [
    # descriptive_time_series
    "compute_returns",
    "cross_section_return_stats",
    "rolling_volatility",
    "intraday_volatility_profile",
    "select_representative_symbols",
    "plot_cross_section_hist",
    "plot_cross_section_box",
    "plot_intraday_profile",
    "plot_representative_time_series",
    "run_descriptive_eda",

    # correlation
    "correlation_analysis",

    # cointegration
    "compute_daily_cointegration",
    "compute_pair_stats",
    "rank_pairs",
    "plot_daily_pvalues",
    "plot_coint_network",
    "compute_daily_spread_stats",
    "plot_daily_mean_spreads",
    "plot_intraday_prices_last_day",
    "plot_intraday_spreads_last_day",
]