"""
Explorative Korrelationsanalyse für Wide-Format Preis- oder Logpreis-Zeitreihen.

Scope:
- Berechnung von Korrelationsmatrizen (Pearson / Spearman)
- Identifikation stark korrelierter Paare
- Auswahl repräsentativer Symbole für Visualisierung
- Optionale Heatmap-Darstellung

Designentscheidungen:
- Funktionaler Stil, keine Klassen
- Rückgabe eines Ergebnis-Dictionaries für flexible Weiterverarbeitung
- Plotting ist optional, aber Teil dieser Funktion

Wichtige Hinweise:
- Die konkrete Heatmap-Gestaltung (Layout, Farben, Skalierung)
  ist Teil des beobachtbaren Verhaltens.
- Refactorings dürfen numerische Ergebnisse oder Plots nicht verändern.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def correlation_analysis(
    prices_wide,
    method="pearson",
    strong_threshold=0.97,
    top_n_heatmap=19,
    plot=True,
    title=None,
):
    """
    Führt eine explorative Korrelationsanalyse auf Wide-Format Zeitreihen durch.

    Parameter:
    - prices_wide: DataFrame (index=datetime, columns=symbol)
    - method: 'pearson' oder 'spearman'
    - strong_threshold: Schwelle für stark korrelierte Paare
      (nur obere Dreiecksmatrix)
    - top_n_heatmap: Anzahl repräsentativer Symbole für Heatmap
    - plot: wenn True, wird eine Heatmap geplottet
    - title: optionaler Plot-Titel

    Rückgabe:
    dict mit Schlüsseln:
      - corr: vollständige Korrelationsmatrix
      - strong_pairs: DataFrame stark korrelierter Paare
      - representative_symbols: Index der repräsentativen Symbole
      - vmin_offdiag: minimale Off-Diagonal-Korrelation
      - vmax_offdiag: maximale Off-Diagonal-Korrelation
      - corr_sub: Submatrix für Heatmap

    Side effects:
    - Bei plot=True: Erzeugt eine Seaborn/Matplotlib-Heatmap und ruft plt.show().
    """
    if prices_wide.empty:
        raise ValueError("prices_wide ist leer.")

    # 1) Korrelationsmatrix
    corr = prices_wide.corr(method=method)

    # 2) Stark korrelierte Paare (nur obere Dreiecksmatrix)
    pairs = []
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and val > strong_threshold:
                pairs.append((cols[i], cols[j], float(val)))

    strong_pairs = (
        pd.DataFrame(pairs, columns=["symbol_1", "symbol_2", "correlation"])
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )

    # 3) Repräsentative Symbole (nach mittlerer Korrelation)
    mean_corr = corr.mean().sort_values(ascending=False)
    representative_symbols = mean_corr.head(top_n_heatmap).index

    # 4) Submatrix + Off-Diagonal-Skalierung
    corr_sub = corr.loc[representative_symbols, representative_symbols]

    mask = np.eye(len(corr_sub), dtype=bool)
    offdiag = corr_sub.where(
        ~pd.DataFrame(mask, index=corr_sub.index, columns=corr_sub.columns)
    )

    vmin = offdiag.min().min()
    vmax = offdiag.max().max()

    # 5) Plot
    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_sub,
            mask=mask,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".2f",
            linewidths=0.2,
            annot_kws={"size": 8},
        )
        plt.title(title or f"Korrelations-Heatmap der Top {top_n_heatmap} Aktien (reskaliert)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    return {
        "corr": corr,
        "strong_pairs": strong_pairs,
        "representative_symbols": representative_symbols,
        "vmin_offdiag": vmin,
        "vmax_offdiag": vmax,
        "corr_sub": corr_sub,
    }