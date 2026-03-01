"""
pairs_trading.modeling.pair_selection

Paarselektion für Pairs-Trading auf Basis von Korrelation
und Cointegration mit Caching.

Wichtiger Hinweis:
- Diese Logik bestimmt direkt das Trading-Universum.
- Refactorings dürfen Auswahl, Sortierung oder Caching nicht verändern.
"""

from pathlib import Path

import pandas as pd

from pairs_trading import eda


# ============================================================================
# Interne Hilfsfunktionen
# ============================================================================

def _upper_triangle_pairs_from_corr(corr: pd.DataFrame, top_n: int):
    cols = corr.columns
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iat[i, j]
            if pd.notna(v):
                pairs.append((cols[i], cols[j], float(abs(v))))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in pairs[:top_n]]


def _cache_path_for_rebalance_day(
    rebalance_day,
    lookback_pair_days: int,
    top_corr_pairs: int,
    top_daily_pairs: int,
    p_cointegration: float,
    cache_dir: Path,
) -> Path:
    d = pd.to_datetime(rebalance_day).strftime("%Y%m%d")
    fname = (
        f"pairs_{d}"
        f"_lb{lookback_pair_days}"
        f"_corr{top_corr_pairs}"
        f"_top{top_daily_pairs}"
        f"_p{str(p_cointegration).replace('.', '')}.pkl"
    )
    return cache_dir / fname


# ============================================================================
# Öffentliche API
# ============================================================================

def select_pairs_for_day(
    log_prices_window: pd.DataFrame,
    top_corr_pairs: int,
    top_daily_pairs: int,
    p_thresh: float,
) -> pd.DataFrame:
    """
    Selektiert konsistente Pairs für ein gegebenes Zeitfenster.
    """
    rets = log_prices_window.diff().dropna(how="all")
    corr = rets.corr(method="pearson")
    corr_pairs = _upper_triangle_pairs_from_corr(corr, top_n=top_corr_pairs)

    daily_coint = eda.compute_daily_cointegration(
        log_prices_window,
        corr_pairs,
        min_obs_per_day=30,
        sort_index=True,
        sort_symbols_in_pair=False,
        verbose=False,
    )
    pair_stats = eda.compute_pair_stats(daily_coint, p_thresh=p_thresh)
    top_consistency, _ = eda.rank_pairs(pair_stats, top_n=top_daily_pairs)
    return top_consistency


def get_pairs_for_test_day(
    day,
    all_days: pd.Index,
    log_prices: pd.DataFrame,
    lookback_pair_days: int,
    top_corr_pairs: int,
    top_daily_pairs: int,
    p_cointegration: float,
    rebalance_every_n_days: int,
    cache_dir,
):
    """
    Liefert (ggf. aus Cache) die zu handelnden Pairs für einen Testtag.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    idx_day = all_days.get_loc(day)

    block_start_idx = (idx_day // rebalance_every_n_days) * rebalance_every_n_days
    rebalance_day = all_days[block_start_idx]

    path = _cache_path_for_rebalance_day(
        rebalance_day=rebalance_day,
        lookback_pair_days=lookback_pair_days,
        top_corr_pairs=top_corr_pairs,
        top_daily_pairs=top_daily_pairs,
        p_cointegration=p_cointegration,
        cache_dir=cache_dir,
    )
    if path.exists():
        return rebalance_day, pd.read_pickle(path)

    idx_rb = all_days.get_loc(rebalance_day)
    pair_days = all_days[idx_rb - lookback_pair_days: idx_rb]

    lp_pair = log_prices.loc[log_prices.index.normalize().isin(pair_days)].copy()

    top_pairs_df = select_pairs_for_day(
        lp_pair,
        top_corr_pairs=top_corr_pairs,
        top_daily_pairs=top_daily_pairs,
        p_thresh=p_cointegration,
    )

    top_pairs_df.to_pickle(path)
    return rebalance_day, top_pairs_df