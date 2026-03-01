"""
pairs_trading.modeling

Public API (Re-Exports) für Modellierung im pairs_trading Projekt.

Wichtiger Hinweis:
- Diese Datei definiert die nach außen sichtbare Oberfläche des Subpackages.
- Refactorings innerhalb der Module dürfen diese Importpfade und Funktionsnamen nicht brechen.
- Funktionalität und Print-Ausgaben sind Teil des beobachtbaren Verhaltens.
"""

from .pair_selection import (
    select_pairs_for_day,
    get_pairs_for_test_day,
)

from .dataset import (
    build_Xy_from_feature_panel,
)

from .walkforward import (
    walk_forward_run,
    walk_forward_collect_trading_artifacts,
)

from .evaluation import (
    roc_mean_std_band,
    confusion_for_model,
)

from .plotting import (
    plot_auc_per_day,
    plot_auc_boxplot,
    plot_xgb_feature_importance,
    plot_roc_mean_std_comparison,
    plot_confusion_matrices_side_by_side,
    plot_daily_pnl,
    plot_equity_and_drawdown,
)

__all__ = [
    # pair_selection
    "select_pairs_for_day",
    "get_pairs_for_test_day",

    # dataset
    "build_Xy_from_feature_panel",

    # walkforward
    "walk_forward_run",
    "walk_forward_collect_trading_artifacts",

    # evaluation
    "roc_mean_std_band",
    "confusion_for_model",

    # plotting
    "plot_auc_per_day",
    "plot_auc_boxplot",
    "plot_xgb_feature_importance",
    "plot_roc_mean_std_comparison",
    "plot_confusion_matrices_side_by_side",
    "plot_daily_pnl",
    "plot_equity_and_drawdown",
]