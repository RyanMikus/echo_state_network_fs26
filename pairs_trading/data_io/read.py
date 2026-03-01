"""
pairs_trading.data_io.read

Lese-Funktionen für Rohdaten (CSV) und vorverarbeitete Daten (Parquet).

Wichtiger Hinweis:
- Die Print-Ausgaben (Inhalt/Reihenfolge) sind Teil des beobachtbaren Verhaltens.
- Refactorings müssen diese Ausgaben unverändert lassen.
"""

import pandas as pd
from typing import Optional, List

from .paths import unprocessed_files, parquet_path
from .utils import _print_df_overview


def load_unprocessed_data(
    *,
    data_dir: str,
    pattern: str,
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lädt Rohdaten (unprocessed) aus CSV-Dateien, die per glob-pattern gefunden werden.

    Parameter:
    - data_dir: Verzeichnis, in dem gesucht wird
    - pattern: glob pattern (z. B. 'sp500_1min_2026-01-05*.csv')
    - parse_dates: wird direkt an pandas.read_csv(parse_dates=...) weitergereicht

    Rückgabe:
    - Ein DataFrame, der die Inhalte aller gefundenen Dateien konkateniert.

    Raises:
    - FileNotFoundError: falls keine Dateien zum Pattern gefunden werden.

    Side effects:
    - Gibt Statusinformationen und eine DataFrame-Übersicht via print aus.
    """
    files = unprocessed_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden für Pattern '{pattern}'")

    print(f"\n[Unprocessed Data Load]")
    print("Verzeichnis:", data_dir)
    print("Pattern:", pattern)
    print("Gefundene Dateien:", len(files))

    dfs = [
        pd.read_csv(fp, parse_dates=parse_dates)
        for fp in files
    ]
    df = pd.concat(dfs, ignore_index=True)

    _print_df_overview(df, "Unprocessed Data")

    return df


def load_preprocessed_data(
    *,
    data_dir: str,
    basename: str,
) -> pd.DataFrame:
    """
    Lädt vorverarbeitete Daten (preprocessed) aus einer Parquet-Datei.

    Parameter:
    - data_dir: Verzeichnis der Daten
    - basename: Dateibasis (mit oder ohne '.parquet'); wird über paths.parquet_path normalisiert

    Rückgabe:
    - DataFrame aus der Parquet-Datei

    Side effects:
    - Gibt Statusinformationen und eine DataFrame-Übersicht via print aus.
    """
    fp = parquet_path(data_dir, basename)

    print(f"\n[Preprocessed Data Load]")
    print("Pfad:", fp)

    df = pd.read_parquet(fp)

    _print_df_overview(df, "Preprocessed Data")

    return df