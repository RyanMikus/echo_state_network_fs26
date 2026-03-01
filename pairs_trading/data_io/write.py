"""
pairs_trading.data_io.write

Schreib-Funktionen für Rohdaten (CSV) und vorverarbeitete Daten (Parquet + optionale Metadata).

Wichtiger Hinweis:
- Die Print-Ausgaben (Inhalt/Reihenfolge) sind Teil des beobachtbaren Verhaltens.
- Refactorings müssen diese Ausgaben unverändert lassen.
"""

import pandas as pd

from .paths import (
    as_path,
    parquet_path,
    metadata_path,
    intraday_csv_path,
)
from .utils import _print_df_overview


def save_unprocessed_data(
    df: pd.DataFrame,
    *,
    date: str,
    interval: str,
    data_dir: str = "Daten",
) -> None:
    """
    Speichert Rohdaten (unprocessed) als CSV.

    Parameter:
    - df: zu speichernder DataFrame
    - date: Datum als ISO-String (z. B. '2026-01-05'), wird in den Dateinamen übernommen
    - interval: Intervall-Tag (z. B. '1m'), wird via paths.intraday_csv_path in interval_tag überführt
    - data_dir: Zielverzeichnis (Default: 'Daten')

    Side effects:
    - Erstellt data_dir (mkdir parents=True, exist_ok=True).
    - Schreibt eine CSV-Datei.
    - Gibt Übersicht und Speicherpfad via print aus.
    """
    _print_df_overview(df, "Unprocessed Data (to be saved)")

    data_dir_p = as_path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)

    fp = intraday_csv_path(data_dir_p, date=date, interval=interval)
    df.to_csv(fp, index=False)

    print(f"\nGespeichert: {fp}")


def save_preprocessed_data(
    df: pd.DataFrame,
    *,
    data_dir: str,
    basename: str,
    metadata=None,
) -> None:
    """
    Speichert vorverarbeitete Daten als Parquet (+ optional Metadata).

    Parameter:
    - df: zu speichernder DataFrame
    - data_dir: Zielverzeichnis
    - basename: Dateibasis für Parquet (mit/ohne '.parquet' möglich)
    - metadata: optionales JSON-serialisierbares Objekt; wenn vorhanden, wird zusätzlich eine
      '<base>_metadata.json' geschrieben

    Side effects:
    - Erstellt data_dir (mkdir parents=True, exist_ok=True).
    - Schreibt Parquet und optional Metadata-JSON.
    - Gibt Übersicht und Speicherstatus via print aus.

    Hinweis:
    - Der Import von 'json' erfolgt bewusst innerhalb des Metadata-Blocks.
      (Nicht verändern, um Verhalten/Abhängigkeiten nicht zu verschieben.)
    """
    _print_df_overview(df, "Preprocessed Data (to be saved)")

    data_dir_p = as_path(data_dir)
    data_dir_p.mkdir(parents=True, exist_ok=True)

    parquet_fp = parquet_path(data_dir_p, basename)
    df.to_parquet(parquet_fp)

    if metadata is not None:
        meta_fp = metadata_path(data_dir_p, basename)
        import json
        with open(meta_fp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("Metadata gespeichert als", meta_fp)

    print(f"\nGespeichert als {parquet_fp}")