"""
pairs_trading

Top-level Package für das pairs_trading Projekt.

Wichtiger Hinweis:
- Dieses Package bündelt alle fachlichen Subpackages (Data, EDA, Features, Modeling).
- Diese Datei definiert die öffentlich sichtbare Paketstruktur.
- Refactorings innerhalb der Subpackages dürfen diese Importpfade nicht brechen.
"""

from . import data_io
# from . import preprocessing
# from . import eda
# from . import features
# from . import modeling


__all__ = [
    "data_io",
    # "preprocessing",
    # "eda",
    # "features",
    # "modeling",
]