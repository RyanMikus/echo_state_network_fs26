"""
pairs_trading.preprocessing

Public API (Re-Exports) für Preprocessing im pairs_trading Projekt.

Wichtiger Hinweis:
- Diese Datei definiert die nach außen sichtbare Oberfläche des Subpackages.
- Refactorings innerhalb der Module dürfen diese Importpfade und Funktionsnamen nicht brechen.
- Funktionalität und Print-Ausgaben sind Teil des beobachtbaren Verhaltens und dürfen nicht
  still verändert werden.
"""

from .data_quality import (
    analyze_missing_values,
    analyze_missing_coverage,
    plot_intraday_missing_distribution,
    analyze_nan_gaps_global,
    gap_len_counts,
    long_gap_summary,
    gap_stats_stocks_with_min_gap,
    plot_gap_length_hist,
    plot_stocks_vs_gap_threshold,
    validate_and_finalize_df,
)

from .data_cleaning import (
    filter_symbols_by_full_coverage,
    find_symbols_with_long_gaps,
    drop_symbols,
    ensure_datetime_and_sort,
    drop_duplicate_rows,
    fill_close_gaps_per_symbol,
    symbol_lengths,
    expected_length_mode,
    symbols_with_unexpected_length,
    removed_symbols_between,
)

__all__ = [
    # data_quality
    "analyze_missing_values",
    "analyze_missing_coverage",
    "plot_intraday_missing_distribution",
    "analyze_nan_gaps_global",
    "gap_len_counts",
    "long_gap_summary",
    "gap_stats_stocks_with_min_gap",
    "plot_gap_length_hist",
    "plot_stocks_vs_gap_threshold",
    "validate_and_finalize_df",

    # data_cleaning
    "filter_symbols_by_full_coverage",
    "find_symbols_with_long_gaps",
    "drop_symbols",
    "ensure_datetime_and_sort",
    "drop_duplicate_rows",
    "fill_close_gaps_per_symbol",
    "symbol_lengths",
    "expected_length_mode",
    "symbols_with_unexpected_length",
    "removed_symbols_between",
]
