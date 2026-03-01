"""
pairs_trading.data_io.utils

Interne Utilities für das data_io Subpackage.

Wichtiger Hinweis:
- Funktionen hier sind bewusst "internal" und nicht Teil der öffentlichen API.
- Insbesondere Print-Layouts gelten als beobachtbares Verhalten und dürfen nicht
  still verändert werden.
"""

import pandas as pd


def _print_df_overview(df: pd.DataFrame, title: str) -> None:
    """
    Druckt eine kurze Übersicht über einen DataFrame.

    Hinweis:
    - Das genaue Print-Layout ist Teil des aktuellen Verhaltens.
      (Titel, Reihenfolge der Zeilen, Head-Ausgabe.)
    """
    print(f"\n[{title}]")
    print("Shape:", df.shape)
    print("Spalten:", list(df.columns))
    print("\nHead:")
    print(df.head())
