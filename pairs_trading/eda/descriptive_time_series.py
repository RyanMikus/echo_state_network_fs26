"""
Minimal descriptive EDA for intraday equity time series.

Scope:
- Deskriptive Statistik für viele Symbole (~500)
- Fokus auf Cross-Section-Analysen und repräsentative Drill-Downs
- Unterstützt Long-Format Input (timestamp, symbol, price)

Designentscheidungen:
- Funktionaler Stil (keine Klassen), Notebook- und EDA-freundlich
- Plotting ist bewusst Teil dieses Moduls
- Robuster Umgang mit fehlenden oder unvollständigen Daten

Wichtige Hinweise:
- Plot-Ausgaben gelten als beobachtbares Verhalten.
- Refactorings dürfen Struktur verbessern, aber keine Outputs verändern.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew as _skew
from scipy.stats import kurtosis as _kurtosis


# =========================
# Helpers (internal)
# =========================

def _ensure_datetime(df, timestamp_col):
    """
    Normalisiert die Timestamp-Spalte zu pandas datetime.

    Verhalten:
    - Ungültige Timestamps werden zu NaT
    - Zeilen mit NaT werden verworfen
    """
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=False, errors="coerce")
    out = out.dropna(subset=[timestamp_col])
    return out


def _pivot_prices_long_to_wide(df, timestamp_col, symbol_col, price_col, freq=None):
    """
    Long -> Wide Transformation:
    index = timestamp
    columns = symbol
    values = price

    Falls freq gesetzt ist:
    - Resampling per last observation innerhalb jedes Intervalls
    """
    tmp = df[[timestamp_col, symbol_col, price_col]].copy()
    tmp = tmp.dropna(subset=[price_col, symbol_col, timestamp_col])
    tmp = tmp.sort_values([symbol_col, timestamp_col])

    wide = tmp.pivot(index=timestamp_col, columns=symbol_col, values=price_col).sort_index()

    if freq is not None:
        wide = wide.resample(freq).last()

    return wide


def compute_returns(prices_wide, log_return=True):
    """
    Berechnet (Log-)Returns aus Wide-Format Preisen.

    Parameter:
    - prices_wide: DataFrame (index=datetime, columns=symbol)
    - log_return: wenn True, log returns; sonst pct_change

    Rückgabe:
    - DataFrame gleicher Struktur wie prices_wide
    """
    if log_return:
        return np.log(prices_wide).diff()
    return prices_wide.pct_change()


def _skew_series(x):
    x = x.dropna()
    if len(x) < 3:
        return np.nan
    return float(_skew(x.values, bias=False))


def _kurtosis_series(x):
    x = x.dropna()
    if len(x) < 4:
        return np.nan
    return float(_kurtosis(x.values, fisher=False, bias=False))


def _safe_quantile(s, q):
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(q))


def _count_zeros(s):
    s = s.dropna()
    if s.empty:
        return np.nan
    return float((s == 0).mean())


def _intraday_profile_index(dt_index):
    return pd.Index(dt_index.time, name="time")


# =========================
# Core statistics
# =========================

def cross_section_return_stats(returns_wide):
    """
    Per-Symbol Zusammenfassung der Return-Verteilung.

    Enthält:
    - Lage- und Streuungsmaße
    - Tail-Statistiken
    - einfache Liquiditätsproxies
    """
    stats = pd.DataFrame(index=returns_wide.columns)

    stats["n_obs"] = returns_wide.count()
    stats["mean"] = returns_wide.mean()
    stats["std"] = returns_wide.std()
    stats["q01"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.01))
    stats["q05"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.05))
    stats["median"] = returns_wide.median()
    stats["q95"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.95))
    stats["q99"] = returns_wide.apply(lambda s: _safe_quantile(s, 0.99))
    stats["skew"] = returns_wide.apply(_skew_series)
    stats["kurtosis"] = returns_wide.apply(_kurtosis_series)
    stats["pct_zero_returns"] = returns_wide.apply(_count_zeros)

    abs_r = returns_wide.abs()
    stats["mean_abs_return"] = abs_r.mean()
    stats["abs_q99"] = abs_r.apply(lambda s: _safe_quantile(s, 0.99))

    return stats.sort_values("std", ascending=False)


def rolling_volatility(returns_wide, window):
    """
    Rolling Volatilität (Standardabweichung) pro Symbol.
    """
    return returns_wide.rolling(window).std()


def intraday_volatility_profile(returns_wide):
    """
    Mean absolute return pro Tageszeit.

    Rückgabe:
    - index = time-of-day
    - columns = symbols
    """
    if not isinstance(returns_wide.index, pd.DatetimeIndex):
        raise TypeError("returns_wide index must be DatetimeIndex for intraday profile.")

    times = _intraday_profile_index(returns_wide.index)
    return returns_wide.abs().groupby(times).mean()


# =========================
# Representative selection
# =========================

def select_representative_symbols(stats, n_top_vol=5, n_bottom_vol=5, n_median_vol=5):
    """
    Auswahl repräsentativer Symbole für Drill-Down-Plots:
    - hohe Volatilität
    - niedrige Volatilität
    - Nähe zur Median-Volatilität
    """
    s = stats.dropna(subset=["std"]).copy()
    if s.empty:
        return []

    s_sorted = s.sort_values("std", ascending=False)
    top = list(s_sorted.head(n_top_vol).index)
    bottom = list(s_sorted.tail(n_bottom_vol).index)

    med = float(s["std"].median())
    s["dist_to_median"] = (s["std"] - med).abs()
    median_syms = list(s.sort_values("dist_to_median").head(n_median_vol).index)

    out = []
    for sym in top + median_syms + bottom:
        if sym not in out:
            out.append(sym)
    return out


# =========================
# Plotting (side effects)
# =========================

def plot_cross_section_hist(stats, col, bins=50, title=None):
    """
    Histogramm einer Cross-Section-Statistik.

    Side effect:
    - erzeugt Matplotlib-Plot + plt.show()
    """
    s = stats[col].dropna()
    if s.empty:
        print("[plot] No data for histogram:", col)
        return
    s.hist(bins=bins)
    plt.title(title or f"Cross-section distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_cross_section_box(stats, cols, title=None):
    """
    Boxplot mehrerer Cross-Section-Statistiken.

    Side effect:
    - erzeugt Matplotlib-Plot + plt.show()
    """
    data = [stats[c].dropna().values for c in cols if c in stats.columns]
    labels = [c for c in cols if c in stats.columns]
    if not data:
        print("[plot] No data for boxplot.")
        return
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title or "Cross-section boxplot")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_intraday_profile(profile, title=None):
    """
    Plottet das mittlere Intraday-Volatilitätsprofil.

    Side effect:
    - erzeugt Matplotlib-Plot + plt.show()
    """
    if profile is None or profile.empty:
        print("[plot] No data for intraday profile.")
        return

    profile.mean(axis=1).plot()
    plt.title(title or "Intraday volatility profile (mean |return| across symbols)")
    plt.xlabel("time-of-day")
    plt.ylabel("mean |return|")
    plt.tight_layout()
    plt.show()


def plot_representative_time_series(returns_wide, symbols, kind="returns", title=None):
    """
    Zeitreihen-Plots für eine kleine Auswahl repräsentativer Symbole.

    Side effect:
    - erzeugt Matplotlib-Plots + plt.show()
    """
    symbols = [s for s in symbols if s in returns_wide.columns]
    if not symbols:
        print("[plot] No representative symbols available to plot.")
        return

    n = len(symbols)
    fig_h = min(2.2 * n + 1.5, 30)
    returns_wide[symbols].plot(subplots=True, figsize=(12, fig_h), legend=False)
    plt.suptitle(title or f"Representative {kind} time series ({n} symbols)")
    plt.tight_layout()
    plt.show()


# =========================
# End-to-end EDA pipeline
# =========================

def run_descriptive_eda(
    df,
    timestamp_col="timestamp",
    symbol_col="symbol",
    price_col="price",
    freq=None,
    log_return=True,
    drop_nonpositive_prices=True,
    plot=True,
    n_top_vol=5,
    n_bottom_vol=5,
    n_median_vol=5,
    max_representative_plots=15,
):
    """
    End-to-end deskriptiver EDA-Schritt.

    Rückgabe:
    dict mit:
      - prices_wide
      - returns_wide
      - stats
      - representative_symbols
      - intraday_profile
    """
    df2 = _ensure_datetime(df, timestamp_col)

    if drop_nonpositive_prices:
        df2 = df2[df2[price_col] > 0].copy()

    prices_wide = _pivot_prices_long_to_wide(
        df2,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        price_col=price_col,
        freq=freq,
    )

    returns_wide = compute_returns(prices_wide, log_return=log_return).dropna(how="all")

    stats = cross_section_return_stats(returns_wide)

    reps = select_representative_symbols(
        stats,
        n_top_vol=n_top_vol,
        n_bottom_vol=n_bottom_vol,
        n_median_vol=n_median_vol,
    )

    intraday_prof = None
    if isinstance(returns_wide.index, pd.DatetimeIndex):
        intraday_prof = intraday_volatility_profile(returns_wide)

    if plot:
        plot_cross_section_hist(stats, "std", title="Cross-section: return volatility (std)")
        plot_cross_section_hist(stats, "mean_abs_return", title="Cross-section: mean absolute return")
        plot_cross_section_hist(stats, "pct_zero_returns", title="Cross-section: % zero returns")
        plot_cross_section_box(stats, ["mean", "std", "skew", "kurtosis"])

        if intraday_prof is not None:
            plot_intraday_profile(intraday_prof)

        plot_representative_time_series(
            returns_wide,
            reps[:max_representative_plots],
            kind="log-returns",
        )

    return {
        "prices_wide": prices_wide,
        "returns_wide": returns_wide,
        "stats": stats,
        "representative_symbols": reps,
        "intraday_profile": intraday_prof,
    }