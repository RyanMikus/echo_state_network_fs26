"""
pairs_trading.data_io

Public API (Re-Exports) für I/O und Datensammlung im Projekt.

Wichtiger Hinweis:
- Diese Datei definiert die nach außen sichtbare Oberfläche des Subpackages.
- Refactorings innerhalb der Module dürfen diese Importpfade und Funktionsnamen nicht brechen.
"""

from .read import (
    load_unprocessed_data,
    load_preprocessed_data,
)

from .write import (
    save_unprocessed_data,
    save_preprocessed_data,
)

from .collect import (
    get_sp500_universe,
    collect_sp500_intraday_data,
)

__all__ = [
    "load_unprocessed_data",
    "load_preprocessed_data",
    "save_unprocessed_data",
    "save_preprocessed_data",
    "get_sp500_universe",
    "collect_sp500_intraday_data",
]