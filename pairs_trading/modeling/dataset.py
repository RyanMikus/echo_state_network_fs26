"""
pairs_trading.modeling.dataset

Utilities zur Konstruktion von Trainings- und Test-Datasets aus Feature-Panels.

Wichtiger Hinweis:
- Diese Funktionen definieren die Zielvariable (Label) explizit.
- Refactorings dürfen Feature-Selektion, Label-Definition und Rückgaben nicht verändern.
"""

import pandas as pd


# ============================================================================
# Dataset-Konstruktion
# ============================================================================

def build_Xy_from_feature_panel(
    panel: pd.DataFrame,
    k: int,
    rsi_w: int,
    z_w: int,
    vol_w: int,
    return_meta: bool = False,
):
    """
    Erzeugt Feature-Matrix X und Zielvariable y aus einem Feature-Panel.

    Zielvariable:
    - y = 1, falls |Z_{t+k}| < |Z_t| (Mean-Reversion-Indikator)

    Rückgabe:
    - X : Feature-Matrix
    - y : Zielvariable
    - meta (optional): DataFrame mit ['datetime', 'pair']
    """
    pair_names = panel.columns.get_level_values(1).unique().tolist()

    blocks = []
    for p in pair_names:
        dfp = panel.xs(p, axis=1, level=1).copy()
        dfp["pair"] = p
        blocks.append(dfp)

    long_df = (
        pd.concat(blocks, axis=0)
        .reset_index()
        .rename(columns={"index": "datetime"})
    ).sort_values(["pair", "datetime"])

    z_col = f"Z_{z_w}_spread"
    if z_col not in long_df.columns:
        raise ValueError(f"Missing column: {z_col}")

    z_now = long_df[z_col].astype(float)
    z_fut = long_df.groupby("pair")[z_col].shift(-k).astype(float)

    y = (z_fut.abs() < z_now.abs()).astype(int)

    X = long_df[
        [
            f"RSI_{rsi_w}_spread",
            f"Z_{z_w}_spread",
            f"VOL_{vol_w}_spread",
            "return_pct_spread",
        ]
    ].copy()

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    meta = None
    if return_meta:
        meta = long_df.loc[valid, ["datetime", "pair"]].reset_index(drop=True)

    return X, y, meta