"""
pairs_trading.preprocessing.data_cleaning

Bereinigungs- und Transformationsfunktionen für das Preprocessing.

Wichtiger Hinweis:
- Print-Ausgaben (falls vorhanden) und Rückgaben sind Teil des beobachtbaren Verhaltens.
- Refactorings dürfen Funktionalität nicht verändern.
- Dieses Modul ist bewusst "funktional" gehalten (DataFrame rein → DataFrame raus).
"""

import pandas as pd


# ============================================================================
# Coverage / universe filtering
# ============================================================================

def filter_symbols_by_full_coverage(
    df: pd.DataFrame,
    time_col: str = "datetime",
    symbol_col: str = "symbol",
    return_removed: bool = True,
):
    """
    Filtert Symbole, die über den gesamten Zeitraum des Datensatzes präsent sind.

    Ein Symbol gilt als vollständig, wenn es:
      - am globalen Startzeitpunkt vorkommt
      - am globalen Endzeitpunkt vorkommt

    Entfernt damit z.B. neu aufgenommene oder ausgeschlossene Index-Mitglieder.

    Returns:
      - gefiltertes DataFrame
      - entfernte Symbole (optional)
    """
    if time_col not in df.columns:
        raise KeyError(f"Spalte '{time_col}' fehlt im DataFrame")
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    start = df[time_col].min()
    end = df[time_col].max()

    symbols_start = set(df.loc[df[time_col] == start, symbol_col].unique())
    symbols_end = set(df.loc[df[time_col] == end, symbol_col].unique())

    valid_symbols = symbols_start & symbols_end
    all_symbols = set(df[symbol_col].unique())

    removed_symbols = sorted(all_symbols - valid_symbols)

    filtered_df = df[df[symbol_col].isin(valid_symbols)].reset_index(drop=True)

    if return_removed:
        return filtered_df, pd.Index(removed_symbols, name=symbol_col)

    return filtered_df


def find_symbols_with_long_gaps(
    nan_gaps_all: pd.DataFrame,
    min_gap_len: int = 6,
    symbol_col: str = "symbol",
    gap_len_col: str = "gap_len",
) -> pd.Index:
    """
    Liefert Symbole, deren maximale Gap-Länge >= min_gap_len ist.

    Erwartet nan_gaps_all mit Spalten: symbol_col, gap_len_col
    """
    if nan_gaps_all is None or nan_gaps_all.empty:
        return pd.Index([], name=symbol_col)

    missing = {symbol_col, gap_len_col} - set(nan_gaps_all.columns)
    if missing:
        raise KeyError(f"nan_gaps_all fehlt Spalten: {sorted(missing)}")

    max_gap = nan_gaps_all.groupby(symbol_col)[gap_len_col].max()
    return pd.Index(max_gap[max_gap >= min_gap_len].index, name=symbol_col)


def drop_symbols(
    df: pd.DataFrame,
    symbols_to_drop,
    symbol_col: str = "symbol",
    sort_by=None,
    reset_index: bool = True,
):
    """
    Entfernt alle Zeilen mit Symbolen in symbols_to_drop.
    Optional: sort_by, reset_index.

    Returns:
      - df_new
    """
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")

    symbols_to_drop = pd.Index(symbols_to_drop).astype(str)
    df_new = df.loc[~df[symbol_col].isin(symbols_to_drop)].copy()

    if sort_by is not None:
        df_new = df_new.sort_values(list(sort_by)).copy()
    if reset_index:
        df_new = df_new.reset_index(drop=True)

    return df_new


# ============================================================================
# Time handling, sorting, duplicates
# ============================================================================

def ensure_datetime_and_sort(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "datetime",
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Konvertiert time_col nach datetime und sortiert nach (symbol_col, time_col).
    """
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")
    if time_col not in df.columns:
        raise KeyError(f"Spalte '{time_col}' fehlt im DataFrame")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors=errors)
    df = df.sort_values([symbol_col, time_col]).copy()
    return df


def drop_duplicate_rows(
    df,
    subset=None,
    keep="first",
    return_duplicates=True,
    sort_by=None,
    reset_index=True,
):
    """
    Findet und entfernt doppelte Zeilen.

    Standardmäßig werden Duplikate anhand von (datetime, symbol) entfernt.

    Returns:
      - df_clean
      - duplicates_removed (optional, wenn return_duplicates=True)
    """
    if subset is None:
        subset = ["datetime", "symbol"]

    missing = set(subset) - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten für subset: {sorted(missing)}")

    df_work = df.copy()

    if sort_by is not None:
        df_work = df_work.sort_values(list(sort_by)).copy()

    dup_mask = df_work.duplicated(subset=subset, keep=keep)

    duplicates_removed = df_work.loc[dup_mask].copy()
    df_clean = df_work.loc[~dup_mask].copy()

    if reset_index:
        df_clean = df_clean.reset_index(drop=True)
        duplicates_removed = duplicates_removed.reset_index(drop=True)

    if return_duplicates:
        return df_clean, duplicates_removed

    return df_clean


# ============================================================================
# Gap filling / interpolation
# ============================================================================

def fill_close_gaps_per_symbol(
    df: pd.DataFrame,
    value_col: str = "close",
    out_col: str = "close_filled",
    symbol_col: str = "symbol",
    time_col: str = "datetime",
    ensure_sorted: bool = True,
) -> pd.DataFrame:
    """
    Interpoliert fehlende Close-Werte pro Symbol zeitbasiert und füllt Ränder:
      1) interpolate(method="time", limit_area="inside")
      2) bfill() / ffill() für Randbereiche

    Returns:
      - df mit neuer Spalte out_col
    """
    missing = {symbol_col, time_col, value_col} - set(df.columns)
    if missing:
        raise KeyError(f"df fehlt Spalten: {sorted(missing)}")

    df_work = ensure_datetime_and_sort(df, symbol_col=symbol_col, time_col=time_col) if ensure_sorted else df.copy()

    def _fill_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).set_index(time_col)
        s = g[value_col]

        s_interp = s.interpolate(method="time", limit_area="inside")
        s_filled = s_interp.bfill().ffill()

        g[out_col] = s_filled
        return g.reset_index()

    return df_work.groupby(symbol_col, group_keys=False).apply(_fill_group)


# ============================================================================
# Length diagnostics / reporting
# ============================================================================

def symbol_lengths(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.Series:
    """
    Anzahl Zeilen pro Symbol (aufsteigend sortiert).
    """
    if symbol_col not in df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt im DataFrame")
    return df.groupby(symbol_col).size().sort_values()


def expected_length_mode(lengths: pd.Series) -> int:
    """
    Häufigste Länge als Referenz (Mode).
    """
    if lengths is None or lengths.empty:
        raise ValueError("lengths ist leer")
    return int(lengths.mode().iloc[0])


def symbols_with_unexpected_length(lengths: pd.Series, expected_length: int) -> pd.Series:
    """
    Liefert Series der Symbole, deren Länge != expected_length ist.
    """
    return lengths[lengths != expected_length]


def removed_symbols_between(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    symbol_col: str = "symbol",
) -> pd.Index:
    """
    Liefert Symbole, die in df_before vorkommen, aber nicht mehr in df_after.
    """
    if symbol_col not in df_before.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt in df_before")
    if symbol_col not in df_after.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt in df_after")

    before = pd.Index(df_before[symbol_col].dropna().unique()).astype(str)
    after = pd.Index(df_after[symbol_col].dropna().unique()).astype(str)
    return before.difference(after)
