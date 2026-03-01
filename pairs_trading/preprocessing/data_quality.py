"""
pairs_trading.preprocessing.data_quality

Qualitätsanalyse und Validierung für Preprocessing.

Wichtiger Hinweis:
- Print-Ausgaben (Inhalt/Reihenfolge) sind Teil des beobachtbaren Verhaltens.
- Refactorings dürfen Funktionalität und Prints nicht verändern.
- Funktionen in diesem Modul mutieren Eingabe-DataFrames nur, wenn dies im Docstring
  ausdrücklich so passiert; ansonsten wird auf Kopien / lokale Series gearbeitet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ============================================================================
# Missing values – global & coverage analysis
# ============================================================================

def analyze_missing_values(
    df: pd.DataFrame,
    top_n: int = 15,
    bins: int = 30,
    show_plot: bool = True,
) -> pd.Series:
    """
    Analysiert fehlende Werte pro Aktie (über alle Tage).

    Returns:
        pd.Series: Anzahl fehlender Zeilen pro Symbol (absteigend sortiert).
    """
    if top_n < 1:
        raise ValueError("top_n muss >= 1 sein")
    if bins < 1:
        raise ValueError("bins muss >= 1 sein")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    # Hilfsspalten (optional, aber ok wenn du sie brauchst)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    is_missing_row = df.isna().any(axis=1)
    missing_rows = df[is_missing_row]

    missing_by_symbol = (
        missing_rows.groupby("symbol")
        .size()
        .sort_values(ascending=False)
    )

    print(f"Es gibt {missing_by_symbol.size} Aktien mit mindestens einem fehlenden Wert.\n")
    print(f"Top {min(top_n, missing_by_symbol.size)} betroffene Ticker (fehlende Zeilen):")
    print(missing_by_symbol.head(top_n))

    if show_plot:
        plt.figure(figsize=(10, 4))
        missing_by_symbol.plot(kind="hist", bins=bins, edgecolor="black", linewidth=0.8)
        plt.title("Verteilung der Anzahl fehlender Zeilen pro Aktie (über alle Tage)")
        plt.xlabel("Anzahl fehlender Zeilen")
        plt.ylabel("Häufigkeit")
        plt.tight_layout()
        plt.show()

    return missing_by_symbol


def analyze_missing_coverage(
    df: pd.DataFrame,
    smoothing_window: int = 15,
    show_plot: bool = True,
    print_table: bool = True,
) -> pd.DataFrame:
    """
    Berechnet Tages-Coverage (rows_total / rows_missing / missing_%)
    und plottet optional den intraday Verlauf fehlender Zeilen je Minute, getrennt nach Tagen.

    Wichtig: Mutiert df nicht (keine neuen Spalten werden hinzugefügt).

    Returns:
        coverage: DataFrame index=date mit rows_total, rows_missing, missing_%
    """
    if smoothing_window < 1:
        raise ValueError("smoothing_window muss >= 1 sein")

    if "datetime" not in df.columns:
        raise KeyError("Spalte 'datetime' fehlt im DataFrame")

    # Nur lokale Series; df bleibt unverändert
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    order = dt.argsort(kind="mergesort")  # stabil, optional
    dt = dt.iloc[order].reset_index(drop=True)

    is_missing_row = df.isna().any(axis=1).iloc[order].reset_index(drop=True)

    # date/time lokal erzeugen
    dates = dt.dt.date
    times = dt.dt.time

    # Tages-Coverage
    daily_rows = dates.value_counts(sort=False).rename("rows_total").sort_index()
    daily_missing = (
        is_missing_row.groupby(dates).sum().rename("rows_missing").sort_index()
    )

    coverage = pd.concat([daily_rows, daily_missing], axis=1)
    coverage["missing_%"] = 100 * coverage["rows_missing"] / coverage["rows_total"]

    if print_table:
        print("Tages-Coverage (Zeilen gesamt / Fehlzeilen / Prozent):")
        print(coverage)

    # Intraday-Missing Verlauf (nur für fehlende Zeilen)
    if show_plot:
        # Subset nur fehlende Zeilen
        miss_dates = dates[is_missing_row]
        miss_times = times[is_missing_row]

        if miss_dates.empty:
            print("Keine fehlenden Zeilen gefunden – kein intraday Plot.")
            return coverage

        missing_time_by_day = (
            pd.DataFrame({"date": miss_dates, "time": miss_times})
            .groupby(["date", "time"])
            .size()
            .rename("count")
            .reset_index()
        )

        missing_time_by_day["time_dt"] = pd.to_datetime(missing_time_by_day["time"].astype(str))

        smoothed_parts = []
        for d, sub in missing_time_by_day.groupby("date"):
            sub = sub.sort_values("time_dt").copy()
            sub["count_smooth"] = (
                sub["count"].rolling(window=smoothing_window, min_periods=1, center=True).mean()
            )
            smoothed_parts.append(sub)

        missing_time_smoothed = pd.concat(smoothed_parts, ignore_index=True)

        plt.figure(figsize=(12, 5))
        for d, sub in missing_time_smoothed.groupby("date"):
            plt.plot(sub["time_dt"], sub["count_smooth"], label=str(d))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.title("Geglättete Anzahl fehlender Zeilen je Minute – getrennt nach Tagen")
        plt.xlabel("Uhrzeit")
        plt.ylabel(f"Geglättete Anzahl fehlender Zeilen ({smoothing_window}-Minuten-Mittel)")
        # plt.legend(title="Datum", ncol=3) # Wird unübersichtlch
        plt.tight_layout()
        plt.show()

    return coverage


def plot_intraday_missing_distribution(
    df: pd.DataFrame,
    bins: int = 40,
    kde: bool = True,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Plottet die Verteilung der fehlenden Zeilen über die Uhrzeit (intraday),
    aggregiert über alle Tage. Gibt das zugrundeliegende DataFrame zurück.

    Wichtig: Mutiert df nicht.
    """
    if "datetime" not in df.columns:
        raise KeyError("Spalte 'datetime' fehlt im DataFrame")

    dt = pd.to_datetime(df["datetime"], errors="coerce")
    is_missing_row = df.isna().any(axis=1)

    # fehlende Zeilen pro Zeitstempel (über alle Tage)
    missing_by_timestamp = (
        pd.Series(1, index=df.index)
        .where(is_missing_row)
        .groupby(dt)
        .sum()
        .dropna()
        .rename("count")
    )

    missing_time_df = missing_by_timestamp.reset_index().rename(columns={"index": "datetime"})
    if missing_time_df.empty:
        print("Keine fehlenden Zeilen gefunden – kein Plot.")
        return missing_time_df

    # Uhrzeit-Komponente (Dummy-Datum für x-Achse)
    missing_time_df["time_only"] = pd.to_datetime(missing_time_df["datetime"]).dt.time
    missing_time_df["time_dt"] = pd.to_datetime(missing_time_df["time_only"].astype(str))

    if show_plot:
        plt.figure(figsize=(10, 4))
        sns.histplot(
            data=missing_time_df.sort_values("time_dt"),
            x="time_dt",
            weights="count",
            bins=bins,
            kde=kde,
        )
        plt.title("Verteilung der Anzahl fehlender Zeilen über den Tag (intraday, über alle Tage)")
        plt.xlabel("Uhrzeit")
        plt.ylabel("Anzahl fehlender Zeilen")

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return missing_time_df


# ============================================================================
# NaN gap analysis
# ============================================================================

def analyze_nan_gaps_global(
    df: pd.DataFrame,
    col: str = "close",
    time_col: str = "datetime",
    symbol_col: str = "symbol",
    step: str = "1min",
) -> pd.DataFrame:
    """
    Findet konsekutive NaN-Blöcke in `col` je Symbol über den gesamten Datensatz.

    Für jeden Block:
      - gap_len: Lückenlänge in step-Einheiten (default: Minuten)
      - rel_change: relative Preisänderung zwischen Wert vor und nach der Lücke

    Returns:
        pd.DataFrame mit Spalten [symbol, gap_len, rel_change]
    """
    required = {col, time_col, symbol_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Fehlende Spalten im DataFrame: {sorted(missing)}")

    normal_step = pd.Timedelta(step)
    all_gaps = []

    for ticker, data in df.groupby(symbol_col):
        data = data.sort_values(time_col).reset_index(drop=True)

        is_nan = data[col].isna()
        if not is_nan.any():
            continue

        nan_idx = np.where(is_nan.to_numpy())[0]
        gap_blocks = np.split(nan_idx, np.where(np.diff(nan_idx) > 1)[0] + 1)

        for block in gap_blocks:
            gap_start_idx = int(block[0]) - 1
            gap_end_idx = int(block[-1]) + 1

            # Randfälle: kein Wert direkt davor oder danach
            if gap_start_idx < 0 or gap_end_idx >= len(data):
                continue

            t_before = data.loc[gap_start_idx, time_col]
            t_after = data.loc[gap_end_idx, time_col]
            close_before = data.loc[gap_start_idx, col]
            close_after = data.loc[gap_end_idx, col]

            if pd.isna(close_before) or pd.isna(close_after):
                continue

            gap_len = (t_after - t_before) / normal_step
            rel_change = (close_after - close_before) / close_before

            all_gaps.append(
                {
                    symbol_col: ticker,
                    "gap_len": float(gap_len),
                    "rel_change": float(rel_change),
                }
            )

    return pd.DataFrame(all_gaps)


# ============================================================================
# Gap utilities (internal helpers)
# ============================================================================

def _gap_len_series(nan_gaps_df: pd.DataFrame, round_to_int: bool = True) -> pd.Series:
    """
    Interner Helper: gap_len als Series, optional gerundet + int.
    Mutiert nan_gaps_df nicht.
    """
    if "gap_len" not in nan_gaps_df.columns:
        raise KeyError("Spalte 'gap_len' fehlt in nan_gaps_df")

    s = nan_gaps_df["gap_len"].copy()
    if round_to_int:
        s = s.round().astype(int)
    return s


def gap_len_counts(
    nan_gaps_df: pd.DataFrame,
    min_gap: int = 0,
    round_to_int: bool = True,
) -> pd.DataFrame:
    """
    Anzahl vorkommender Lücken pro Lückengröße.

    Returns:
        DataFrame mit Spalten [gap_len, count]
    """
    s = _gap_len_series(nan_gaps_df, round_to_int=round_to_int)
    counts = s.value_counts().sort_index()
    counts = counts[counts.index >= min_gap]

    out = counts.rename("count").reset_index().rename(columns={"index": "gap_len"})
    return out


def long_gap_summary(
    nan_gaps_df: pd.DataFrame,
    threshold_min: int = 5,
    symbol_col: str = "symbol",
    round_to_int: bool = True,
) -> pd.DataFrame:
    """
    Summary je Aktie mit mindestens einer Lücke > threshold_min.

    Returns:
        DataFrame mit Spalten [symbol, n_gaps, max_gap]
    """
    if symbol_col not in nan_gaps_df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt in nan_gaps_df")

    tmp = nan_gaps_df[[symbol_col]].copy()
    tmp["gap_len"] = _gap_len_series(nan_gaps_df, round_to_int=round_to_int)

    long_df = tmp[tmp["gap_len"] > threshold_min]
    if long_df.empty:
        return pd.DataFrame(columns=[symbol_col, "n_gaps", "max_gap"])

    summary = (
        long_df.groupby(symbol_col)["gap_len"]
        .agg(n_gaps="count", max_gap="max")
        .reset_index()
        .sort_values(["max_gap", "n_gaps", symbol_col], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return summary


def gap_stats_stocks_with_min_gap(
    nan_gaps_df: pd.DataFrame,
    symbol_col: str = "symbol",
    round_to_int: bool = True,
) -> pd.DataFrame:
    """
    Für jede Minute m: Anzahl Aktien mit mindestens einer Lücke >= m Minuten.

    Returns:
        DataFrame mit Spalten [gap_length_min, n_stocks]
    """
    if symbol_col not in nan_gaps_df.columns:
        raise KeyError(f"Spalte '{symbol_col}' fehlt in nan_gaps_df")

    tmp = nan_gaps_df[[symbol_col]].copy()
    tmp["gap_len"] = _gap_len_series(nan_gaps_df, round_to_int=round_to_int)

    max_gap_per_symbol = tmp.groupby(symbol_col)["gap_len"].max()
    if max_gap_per_symbol.empty:
        return pd.DataFrame(columns=["gap_length_min", "n_stocks"])

    max_len = int(max_gap_per_symbol.max())
    minutes = np.arange(1, max_len + 1, dtype=int)

    n_stocks = [(max_gap_per_symbol >= m).sum() for m in minutes]
    return pd.DataFrame({"gap_length_min": minutes, "n_stocks": n_stocks})


def plot_gap_length_hist(
    nan_gaps_df: pd.DataFrame,
    bins: int = 40,
    round_to_int: bool = True,
    title: str = "Verteilung der Lückengrößen (über alle Aktien, alle Tage)",
) -> None:
    """
    Histogramm der Lückengrößen. Mutiert nan_gaps_df nicht.
    """
    s = _gap_len_series(nan_gaps_df, round_to_int=round_to_int)

    plt.figure(figsize=(10, 4))
    plt.hist(s.dropna(), bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel("Lückenlänge (Minuten)")
    plt.ylabel("Anzahl Lücken")
    plt.tight_layout()
    plt.show()


def plot_stocks_vs_gap_threshold(
    gap_stats: pd.DataFrame,
    title: str = "Anzahl Aktien mit mindestens einer Lücke ≥ X Minuten",
) -> None:
    """
    Linienplot: x=gap_length_min, y=n_stocks.
    """
    required = {"gap_length_min", "n_stocks"}
    missing = required - set(gap_stats.columns)
    if missing:
        raise KeyError(f"gap_stats fehlt Spalten: {sorted(missing)}")

    if gap_stats.empty:
        print("gap_stats ist leer – kein Plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(gap_stats["gap_length_min"], gap_stats["n_stocks"], marker="o")
    plt.title(title)
    plt.xlabel("Lückenlänge (Minuten)")
    plt.ylabel("Anzahl Aktien")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Validation / finalization
# ============================================================================

def validate_and_finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops redundant columns and validates the cleaned long-format price DataFrame.
    Aborts with a clear error if any consistency check fails.
    Returns a validated copy of the DataFrame.
    """

    df = df.copy()

    # --- 1) Drop redundant columns -----------------------------------------
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # --- 2) Required columns -----------------------------------------------
    required_cols = {"datetime", "symbol", "close_filled"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- 3) Dtypes ----------------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        raise TypeError("Column 'datetime' must be datetime64 dtype")

    # --- 4) Key integrity ---------------------------------------------------
    dup_mask = df.duplicated(subset=["datetime", "symbol"], keep=False)
    if dup_mask.any():
        n = int(dup_mask.sum())
        raise ValueError(f"Found {n} duplicate (datetime, symbol) rows")

    # --- 5) close_filled completeness --------------------------------------
    if df["close_filled"].isna().any():
        n = int(df["close_filled"].isna().sum())
        raise ValueError(f"'close_filled' contains {n} NaN values")

    # --- 6) Sorting (deterministic output) ---------------------------------
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    # --- 7) Summary statistics ---------------------------------------------
    summary = {
        "rows": len(df),
        "symbols": int(df["symbol"].nunique()),
        "start": df["datetime"].min(),
        "end": df["datetime"].max(),
        "filled_rows": int(df[["open","high","low","close"]].isna().all(axis=1).sum())
        if {"open","high","low","close"}.issubset(df.columns)
        else None,
    }

    print("DataFrame validation passed")
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    return df
