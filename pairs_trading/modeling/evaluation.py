"""
pairs_trading.modeling.evaluation

Evaluationsfunktionen für Modelle im Walk-forward-Setting
(ROC-Bänder, Confusion Matrices).

Wichtiger Hinweis:
- Diese Funktionen führen Modelltraining und -evaluation durch.
- Rückgaben und numerische Ergebnisse dürfen durch Refactorings nicht verändert werden.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from pairs_trading import features
from pairs_trading.modeling import dataset
from pairs_trading.modeling import pair_selection


# ============================================================================
# Interne Hilfsfunktionen
# ============================================================================

def _slice_by_days(df_wide: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    mask = df_wide.index.normalize().isin(days)
    return df_wide.loc[mask].copy()


# ============================================================================
# ROC-Auswertung
# ============================================================================

def roc_mean_std_band(
    model_name: str,
    k: int,
    models,
    log_prices: pd.DataFrame,
    all_days: pd.Index,
    test_days: pd.Index,
    lookback_train_days: int,
    lookback_pair_days: int,
    top_corr_pairs: int,
    top_daily_pairs: int,
    p_cointegration: float,
    rebalance_every_n_days: int,
    cache_dir,
    rsi_w: int,
    z_w: int,
    vol_w: int,
    min_train_samples: int = 200,
    min_test_samples: int = 50,
):
    """
    Berechnet ROC Mean ± Std über alle Testtage.

    Rückgabe:
    - fpr_grid
    - mean_tpr
    - std_tpr
    - mean_auc
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fpr_grid = np.linspace(0.0, 1.0, 200)
    tprs = []
    aucs = []

    model = models[model_name]

    for day in test_days:
        idx_day = all_days.get_loc(day)
        train_days = all_days[idx_day - lookback_train_days: idx_day]

        lp_train = _slice_by_days(log_prices, train_days)
        lp_test = _slice_by_days(log_prices, pd.Index([day]))

        _, top_pairs_df = pair_selection.get_pairs_for_test_day(
            day=day,
            all_days=all_days,
            log_prices=log_prices,
            lookback_pair_days=lookback_pair_days,
            top_corr_pairs=top_corr_pairs,
            top_daily_pairs=top_daily_pairs,
            p_cointegration=p_cointegration,
            rebalance_every_n_days=rebalance_every_n_days,
            cache_dir=cache_dir,
        )
        if top_pairs_df.empty:
            continue

        pairs = list(top_pairs_df[["symbol1", "symbol2"]].itertuples(index=False, name=None))

        panel_train = features.build_feature_panel(lp_train, pairs, rsi_w, z_w, vol_w)
        panel_test = features.build_feature_panel(lp_test, pairs, rsi_w, z_w, vol_w)

        X_train, y_train, _ = dataset.build_Xy_from_feature_panel(panel_train, k, rsi_w, z_w, vol_w)
        X_test, y_test, _ = dataset.build_Xy_from_feature_panel(panel_test, k, rsi_w, z_w, vol_w)

        if len(y_train) < min_train_samples or len(y_test) < min_test_samples:
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, proba)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0

        tprs.append(tpr_interp)
        aucs.append(roc_auc_score(y_test, proba))

    if not tprs:
        return (
            fpr_grid,
            np.full_like(fpr_grid, np.nan),
            np.full_like(fpr_grid, np.nan),
            float("nan"),
        )

    tprs = np.asarray(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    return fpr_grid, mean_tpr, std_tpr, mean_auc


# ============================================================================
# Confusion-Matrix-Auswertung
# ============================================================================

def confusion_for_model(
    model_name: str,
    k: int,
    threshold: float,
    models,
    log_prices: pd.DataFrame,
    all_days: pd.Index,
    test_days: pd.Index,
    lookback_train_days: int,
    lookback_pair_days: int,
    top_corr_pairs: int,
    top_daily_pairs: int,
    p_cointegration: float,
    rebalance_every_n_days: int,
    cache_dir,
    rsi_w: int,
    z_w: int,
    vol_w: int,
    min_train_samples: int = 200,
    min_test_samples: int = 50,
):
    """
    Aggregiert Confusion Matrix über alle Walk-forward-Testtage.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    y_true_all = []
    y_pred_all = []

    model = models[model_name]

    for day in test_days:
        idx_day = all_days.get_loc(day)
        train_days = all_days[idx_day - lookback_train_days: idx_day]

        lp_train = _slice_by_days(log_prices, train_days)
        lp_test = _slice_by_days(log_prices, pd.Index([day]))

        _, top_pairs_df = pair_selection.get_pairs_for_test_day(
            day=day,
            all_days=all_days,
            log_prices=log_prices,
            lookback_pair_days=lookback_pair_days,
            top_corr_pairs=top_corr_pairs,
            top_daily_pairs=top_daily_pairs,
            p_cointegration=p_cointegration,
            rebalance_every_n_days=rebalance_every_n_days,
            cache_dir=cache_dir,
        )
        if top_pairs_df.empty:
            continue

        pairs = list(top_pairs_df[["symbol1", "symbol2"]].itertuples(index=False, name=None))

        panel_train = features.build_feature_panel(lp_train, pairs, rsi_w, z_w, vol_w)
        panel_test = features.build_feature_panel(lp_test, pairs, rsi_w, z_w, vol_w)

        X_train, y_train, _ = dataset.build_Xy_from_feature_panel(panel_train, k, rsi_w, z_w, vol_w)
        X_test, y_test, _ = dataset.build_Xy_from_feature_panel(panel_test, k, rsi_w, z_w, vol_w)

        if len(y_train) < min_train_samples or len(y_test) < min_test_samples:
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test.values.tolist())
        y_pred_all.extend((proba > threshold).astype(int).tolist())

    return confusion_matrix(y_true_all, y_pred_all)