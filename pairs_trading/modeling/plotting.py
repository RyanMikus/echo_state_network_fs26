"""
pairs_trading.modeling.plotting

Plot-Utilities für Modellierung und Evaluation im pairs_trading Projekt.
Diese Funktionen enthalten ausschließlich Visualisierung (matplotlib) und
sind notebook-freundlich.

Wichtiger Hinweis:
- Keine Seiteneffekte außer Plotting.
- Keine Abhängigkeit von Notebook-spezifischen Utilities (z.B. display).
- Refactorings dürfen Funktionalität und Plot-Inhalte nicht verändern.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================================
# AUC- und Modellvergleichs-Plots
# ============================================================================

def plot_auc_per_day(
    final_scores: pd.DataFrame,
    *,
    figsize: Tuple[int, int] = (10, 4),
    marker: str = "o",
    linewidth: float = 1.0,
    title: str = "Walk-forward AUC pro Tag (Signal-Stabilität)",
) -> None:
    """
    Plottet AUC je Tag und Modell.

    Erwartet Spalten:
    - 'day'
    - 'model'
    - 'auc'
    """
    plt.figure(figsize=figsize)
    for m in final_scores["model"].unique():
        tmp = final_scores[final_scores["model"] == m].sort_values("day")
        plt.plot(tmp["day"], tmp["auc"], marker=marker, linewidth=linewidth, label=m)

    plt.xlabel("Day")
    plt.ylabel("AUC (per day)")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_auc_boxplot(
    final_scores: pd.DataFrame,
    *,
    figsize: Tuple[int, int] = (6, 4),
    showfliers: bool = False,
    title: str = "Bias–Variance Vergleich (AUC-Verteilung über Tage)",
) -> None:
    """
    Boxplot der AUC-Verteilung pro Modell.

    Erwartet Spalten:
    - 'model'
    - 'auc'
    """
    plt.figure(figsize=figsize)

    models = list(final_scores["model"].unique())
    data = [final_scores.loc[final_scores["model"] == m, "auc"].values for m in models]

    plt.boxplot(data, labels=models, showfliers=showfliers)
    plt.ylabel("AUC")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Feature-Importance- und ROC-Plots
# ============================================================================

def plot_xgb_feature_importance(
    feature_importances: Sequence[float],
    feature_names: Sequence[str],
    *,
    figsize: Tuple[int, int] = (6, 4),
    title: str = "Feature Importance (XGB)",
) -> pd.DataFrame:
    """
    Horizontaler Barplot der Feature Importances.

    Rückgabe:
    - DataFrame mit Spalten ('feature', 'importance'),
      sortiert aufsteigend für sauberen Barh-Plot.
    """
    if len(feature_importances) != len(feature_names):
        raise ValueError("feature_importances und feature_names müssen gleich lang sein.")

    imp_df = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "importance": list(feature_importances),
            }
        )
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    plt.figure(figsize=figsize)
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("Relative Importance")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    return imp_df


def plot_roc_mean_std_comparison(
    roc_a: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
    roc_b: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
    *,
    label_a: str = "Model A",
    label_b: str = "Model B",
    figsize: Tuple[int, int] = (6, 6),
    title: str = "ROC – Vergleich (Mean ± Std über Tage)",
) -> None:
    """
    Vergleich zweier ROC-Bänder (Mean ± Std).

    Erwartet je Modell:
    - (fpr, mean_tpr, std_tpr, mean_auc)
    """
    fpr_a, mean_a, std_a, auc_a = roc_a
    fpr_b, mean_b, std_b, auc_b = roc_b

    plt.figure(figsize=figsize)

    plt.plot(fpr_a, mean_a, label=f"{label_a} (mean AUC={auc_a:.3f})")
    plt.fill_between(fpr_a, mean_a - std_a, mean_a + std_a, alpha=0.20)

    plt.plot(fpr_b, mean_b, label=f"{label_b} (mean AUC={auc_b:.3f})")
    plt.fill_between(fpr_b, mean_b - std_b, mean_b + std_b, alpha=0.20)

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Confusion-Matrix- und Trading-Plots
# ============================================================================

def plot_confusion_matrices_side_by_side(
    cm_left: np.ndarray,
    cm_right: np.ndarray,
    *,
    title_left: str = "Left",
    title_right: str = "Right",
    cmap: str = "Blues",
    values_format: str = "d",
    figsize: Tuple[int, int] = (10, 4),
    suptitle: Optional[str] = None,
) -> None:
    """
    Zwei Confusion Matrices nebeneinander.
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    disp1 = ConfusionMatrixDisplay(cm_left)
    disp1.plot(cmap=cmap, values_format=values_format, ax=plt.gca(), colorbar=False)
    plt.title(title_left)

    plt.subplot(1, 2, 2)
    disp2 = ConfusionMatrixDisplay(cm_right)
    disp2.plot(cmap=cmap, values_format=values_format, ax=plt.gca(), colorbar=False)
    plt.title(title_right)

    if suptitle:
        plt.suptitle(suptitle)

    plt.tight_layout()
    plt.show()


def plot_daily_pnl(
    daily: pd.DataFrame,
    *,
    day_col: str = "day",
    pnl_col: str = "pnl",
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Daily PnL",
) -> None:
    """
    Plot der täglichen PnL-Serie.

    Erwartet Spalten:
    - day_col
    - pnl_col
    """
    plt.figure(figsize=figsize)

    plt.plot(daily[day_col], daily[pnl_col], alpha=0.6, label="Daily PnL")
    plt.axhline(0, color="grey", linestyle="--")

    plt.xlabel("Day")
    plt.ylabel("PnL")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_equity_and_drawdown(
    daily: pd.DataFrame,
    *,
    day_col: str = "day",
    equity_col: str = "equity",
    drawdown_col: str = "drawdown",
    figsize: Tuple[int, int] = (10, 4),
    title: str = "Equity Curve and Drawdown (Trading Sanity Check)",
) -> None:
    """
    Equity- und Drawdown-Plot.

    Erwartet Spalten:
    - day_col
    - equity_col
    - drawdown_col
    """
    plt.figure(figsize=figsize)

    plt.plot(daily[day_col], daily[equity_col], label="Equity")
    plt.fill_between(daily[day_col], daily[drawdown_col], 0, alpha=0.3, label="Drawdown")

    plt.xlabel("Day")
    plt.ylabel("Cumulative PnL")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()