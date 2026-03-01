"""
pairs_trading.data_io.paths

Hilfsfunktionen zur konsistenten Erzeugung von Dateipfaden innerhalb des Projekts.

Ziele:
- Eine zentrale Stelle für Namenskonventionen (CSV/Parquet/Metadata).
- Pfad-Normalisierung, damit Aufrufer wahlweise str oder pathlib.Path übergeben können.

Wichtiger Hinweis:
- Pfadstrings sind Teil des beobachtbaren Verhaltens (z. B. in Prints).
- Änderungen an Namensregeln (z. B. "1m" -> "1min") sind API-relevant und dürfen nicht still geändert werden.
"""

from pathlib import Path


def as_path(data_dir):
    """Normalisiert ein Verzeichnisargument zu pathlib.Path (falls nötig)."""
    return data_dir if isinstance(data_dir, Path) else Path(data_dir)


def _strip_parquet_suffix(basename: str) -> str:
    """
    Entfernt optional die Endung '.parquet' von basename.

    Hintergrund:
    - Externe Aufrufer können basename sowohl mit als auch ohne .parquet liefern.
    - Damit die restliche Pfadlogik stabil bleibt, wird hier normalisiert.
    """
    return basename[:-8] if basename.endswith(".parquet") else basename


def unprocessed_files(data_dir, pattern: str):
    """
    Liefert eine sortierte Liste von Dateien (glob) für Rohdaten.

    Parameter:
    - data_dir: str oder Path
    - pattern: glob pattern (z. B. 'sp500_1min_2026-01-05*.csv')

    Rückgabe:
    - sortierte Liste der gefundenen Pfade
    """
    data_dir_p = as_path(data_dir)
    return sorted(data_dir_p.glob(pattern))


def parquet_path(data_dir, basename: str) -> Path:
    """
    Erzeugt den Parquet-Pfad zu einem Basenamen.

    Konvention:
    - basename ohne/mit '.parquet' wird akzeptiert
    - Ausgabe ist immer '<data_dir>/<base>.parquet'
    """
    data_dir_p = as_path(data_dir)
    base = _strip_parquet_suffix(basename)
    return data_dir_p / f"{base}.parquet"


def metadata_path(data_dir, basename: str) -> Path:
    """
    Erzeugt den Pfad für die JSON-Metadata-Datei zu einem Basenamen.

    Konvention:
    - '<base>_metadata.json'
    - Falls basename bereits auf '_metadata' endet, wird dieses Suffix entfernt,
      um Doppelungen zu vermeiden.
    """
    data_dir_p = as_path(data_dir)
    base = _strip_parquet_suffix(basename)
    if base.endswith("_metadata"):
        base = base[:-9]
    return data_dir_p / f"{base}_metadata.json"


def intraday_csv_path(data_dir, date: str, interval: str) -> Path:
    """
    Erzeugt den CSV-Pfad für Intraday-Rohdaten.

    Konvention:
    - Dateiname: 'sp500_<interval_tag>_<date>.csv'
    - interval_tag: wenn interval auf 'm' endet, wird 'm' zu 'min' ersetzt
      (z. B. '1m' -> '1min'). Dieses Verhalten ist bewusst so implementiert
      und darf nicht verändert werden, da es die Dateinamen festlegt.
    """
    interval_tag = interval.replace("m", "min") if interval.endswith("m") else interval
    return as_path(data_dir) / f"sp500_{interval_tag}_{date}.csv"