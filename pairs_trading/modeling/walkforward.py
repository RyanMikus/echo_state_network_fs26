"""
pairs_trading.modeling.walkforward

Walk-forward-Logik für Modelltraining, Evaluation und Trading-Artefakte
im pairs_trading Projekt.

Wichtiger Hinweis:
- Diese Implementierung definiert den vollständigen Walk-forward-Ablauf.
- Print-Ausgaben (verbose) und Rückgaben sind Teil des beobachtbaren Verhaltens.
- Refactorings dürfen Logik, Trainingsreihenfolge und Resultate nicht verändern.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pairs_trading import features
from pairs_trading.modeling import dataset
from pairs_trading.modeling import pair_selection


# ============================================================================
# Interne Hilfsfunktionen
# ============================================================================

def _slice_by_days(df_wide: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    mask = df_wide.index.normalize().isin(days)
    return df_wide.loc[mask].copy()


def _get_pairs(
    day,
    all_days: pd.Index,
    log_prices: pd.DataFrame,
    lookback_pair_days: int,
    top_corr_pairs: int,
    top_daily_pairs: int,
    p_cointegration: float,
    rebalance_every_n_days: int,
    cache_dir,
) -> pd.DataFrame:
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
    return top_pairs_df


# ============================================================================
# Walk-forward Evaluation
# ============================================================================

def walk_forward_run(
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
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Führt eine vollständige Walk-forward-Evaluation über mehrere Modelle durch.

    Rückgabe:
    - DataFrame mit Tages-AUCs und Metadaten je Modell.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for day in test_days:
        idx_day = all_days.get_loc(day)

        train_days = all_days[idx_day - lookback_train_days: idx_day]
        lp_train = _slice_by_days(log_prices, train_days)
        lp_test = _slice_by_days(log_prices, pd.Index([day]))

        top_pairs_df = _get_pairs(
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

        day_ts = pd.to_datetime(day)

        best_auc_for_day = None
        for model_name, model in models.items():
            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba))

            records.append({
                "day": day_ts,
                "model": model_name,
                "auc": auc,
                "n_pairs": len(pairs),
                "n_train": len(y_train),
                "n_test": len(y_test),
            })

            best_auc_for_day = auc if best_auc_for_day is None else max(best_auc_for_day, auc)

        if verbose and best_auc_for_day is not None:
            print(f"{day_ts.date()} | best_auc={best_auc_for_day:.3f}")

    return pd.DataFrame(records)


# ============================================================================
# Walk-forward Trading-Artefakte
# ============================================================================

def walk_forward_collect_trading_artifacts(
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
) -> pd.DataFrame:
    """
    Sammelte Trading-relevante Artefakte (Proba, Z-Werte, Spread-Renditen)
    für ein einzelnes Modell im Walk-forward-Setting.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = models[model_name]
    records = []

    z_col = f"Z_{z_w}_spread"

    for day in test_days:
        idx_day = all_days.get_loc(day)
        train_days = all_days[idx_day - lookback_train_days: idx_day]

        lp_train = _slice_by_days(log_prices, train_days)
        lp_test = _slice_by_days(log_prices, pd.Index([day]))

        top_pairs_df = _get_pairs(
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
        X_test, y_test, meta = dataset.build_Xy_from_feature_panel(
            panel_test, k, rsi_w, z_w, vol_w, return_meta=True
        )

        if len(y_train) < min_train_samples or len(y_test) < min_test_samples:
            continue

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        z_now = panel_test.xs(z_col, level=0, axis=1).stack().reset_index(drop=True)
        z_fut = z_now.groupby(meta["pair"]).shift(-k)
        spread_ret_k = z_now - z_fut

        day_ts = pd.to_datetime(day)

        for i in range(len(X_test)):
            records.append({
                "day": day_ts,
                "datetime": meta.loc[i, "datetime"],
                "pair": meta.loc[i, "pair"],
                "z": float(z_now.iloc[i]),
                "proba": float(proba[i]),
                "spread_ret_k": float(spread_ret_k.iloc[i]),
            })

    return pd.DataFrame(records).dropna()