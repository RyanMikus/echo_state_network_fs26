"""
Cointegration-based exploratory analysis for intraday equity time series.

Scope:
- Engle-Granger Cointegration auf Intraday-Logpreisen
- Aggregation zu täglichen Paar-Statistiken
- Ranking stabiler Paare
- Visualisierung (p-Werte, Netzwerke, Spreads, Intraday-Verläufe)

Wichtige Hinweise:
- Dieses Modul ist explizit explorativ (EDA).
- Numerische Ergebnisse UND Plot-Layouts gelten als beobachtbares Verhalten.
- Refactorings dürfen keinerlei Berechnung oder Visualisierung verändern.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import coint

try:
    import networkx as nx
except ImportError:
    nx = None


# ---------------------------
# Helpers (internal)
# ---------------------------

def _ensure_datetime_index(df, name="df"):
    """
    Stellt sicher, dass der Index ein pd.DatetimeIndex ist.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index muss ein pd.DatetimeIndex sein.")


def _try_extract_pairs_from_df(pairs_df):
    """
    Robust: akzeptiert verschiedene Pair-Formate.

    Unterstützt:
      - DataFrame mit 2 relevanten Spalten
      - DataFrame mit MultiIndex aus >=2 Levels
      - DataFrame mit Spalte 'pair' / 'pairs' / 'tuple'
    """
    if pairs_df is None or len(pairs_df) == 0:
        return []

    # 1) MultiIndex mit >=2 Levels
    if isinstance(pairs_df.index, pd.MultiIndex) and pairs_df.index.nlevels >= 2:
        pairs = []
        lvl0 = pairs_df.index.get_level_values(0)
        lvl1 = pairs_df.index.get_level_values(1)
        for a, b in zip(lvl0, lvl1):
            pairs.append((str(a), str(b)))
        return pairs

    # 2) Tuple-Spalte
    tuple_cols = [c for c in pairs_df.columns if str(c).lower() in ("pair", "pairs", "tuple", "pair_tuple")]
    for c in tuple_cols:
        vals = pairs_df[c].dropna().values
        pairs = []
        ok = True
        for v in vals:
            if not (isinstance(v, tuple) or isinstance(v, list)) or len(v) != 2:
                ok = False
                break
            pairs.append((str(v[0]), str(v[1])))
        if ok and len(pairs) > 0:
            return pairs

    # 3) Bevorzugte Symbol-Spalten
    preferred = [
        ("symbol1", "symbol2"),
        ("sym1", "sym2"),
        ("s1", "s2"),
        ("stock1", "stock2"),
        ("ticker1", "ticker2"),
        ("a", "b"),
        ("x", "y"),
        ("left", "right"),
    ]
    lower_map = {str(c).lower(): c for c in pairs_df.columns}

    for a, b in preferred:
        if a in lower_map and b in lower_map:
            c1 = lower_map[a]
            c2 = lower_map[b]
            pairs = []
            for _, row in pairs_df[[c1, c2]].iterrows():
                pairs.append((str(row[c1]), str(row[c2])))
            return pairs

    # 4) Fallback: erste zwei Spalten
    if len(pairs_df.columns) >= 2:
        c1, c2 = pairs_df.columns[:2]
        pairs = []
        for _, row in pairs_df[[c1, c2]].iterrows():
            pairs.append((str(row[c1]), str(row[c2])))
        return pairs

    raise ValueError(
        "Konnte aus strong_pairs keine Paare extrahieren. "
        "Erwarte DataFrame mit 2 Spalten oder MultiIndex mit 2 Levels."
    )


def _normalize_pairs(pairs, sort_symbols=True, dedup=True):
    """
    Normalisiert Pair-Listen:
    - akzeptiert Iterable oder DataFrame
    - optionale Sortierung innerhalb des Paars
    - optionale stabile Deduplikation
    """
    if pairs is None:
        return []

    if isinstance(pairs, pd.DataFrame):
        pairs_list = _try_extract_pairs_from_df(pairs)
    else:
        pairs_list = [(str(a), str(b)) for a, b in pairs]

    if sort_symbols:
        pairs_list = [(a, b) if a <= b else (b, a) for a, b in pairs_list]

    if dedup:
        seen = set()
        uniq = []
        for p in pairs_list:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        pairs_list = uniq

    return pairs_list


# ---------------------------
# Core: Cointegration
# ---------------------------

def compute_daily_cointegration(
    log_prices,
    pairs,
    min_obs_per_day=30,
    sort_index=True,
    sort_symbols_in_pair=True,
    verbose=False,
):
    """
    Pro Tag + Paar: Engle-Granger Cointegration p-value auf Intraday-Logpreisen.

    Output:
    DataFrame mit Spalten [date, symbol1, symbol2, pvalue]
    """
    _ensure_datetime_index(log_prices, "log_prices")

    if sort_index:
        log_prices = log_prices.sort_index()

    pairs_list = _normalize_pairs(pairs, sort_symbols=sort_symbols_in_pair, dedup=True)

    unique_days = log_prices.index.normalize().unique()
    records = []

    for current_day in unique_days:
        day_mask = log_prices.index.normalize() == current_day
        day_prices = log_prices.loc[day_mask, :]

        if len(day_prices) < min_obs_per_day:
            continue

        for s1, s2 in pairs_list:
            if s1 not in day_prices.columns or s2 not in day_prices.columns:
                continue

            s1_series = day_prices[s1].dropna()
            s2_series = day_prices[s2].dropna()

            aligned = pd.concat([s1_series, s2_series], axis=1, join="inner").dropna()
            if len(aligned) < min_obs_per_day:
                continue

            aligned.columns = ["s1", "s2"]

            try:
                _score, pvalue, _ = coint(aligned["s1"], aligned["s2"])
            except Exception as e:
                if verbose:
                    print(f"Cointegration-Fehler am {pd.Timestamp(current_day).date()} für {s1}-{s2}: {e}")
                continue

            records.append({
                "date": pd.to_datetime(current_day),
                "symbol1": s1,
                "symbol2": s2,
                "pvalue": float(pvalue),
            })

    daily_coint_df = pd.DataFrame(records)
    if daily_coint_df.empty:
        return daily_coint_df

    daily_coint_df = daily_coint_df.sort_values(["symbol1", "symbol2", "date"]).reset_index(drop=True)
    return daily_coint_df


def compute_pair_stats(daily_coint_df, p_thresh=0.05):
    """
    Aggregation täglicher Cointegration-Ergebnisse zu Paar-Statistiken.
    """
    if daily_coint_df is None or daily_coint_df.empty:
        return pd.DataFrame()

    required = {"date", "symbol1", "symbol2", "pvalue"}
    missing = required - set(daily_coint_df.columns)
    if missing:
        raise ValueError(f"daily_coint_df fehlt Spalten: {missing}")

    pair_stats = (
        daily_coint_df
        .groupby(["symbol1", "symbol2"])
        .agg(
            n_days=("date", "nunique"),
            n_sig_days=("pvalue", lambda x: int((x < p_thresh).sum())),
            mean_p=("pvalue", "mean"),
            std_p=("pvalue", "std"),
            min_p=("pvalue", "min"),
            max_p=("pvalue", "max"),
        )
    )
    pair_stats["consistency"] = pair_stats["n_sig_days"] / pair_stats["n_days"]
    return pair_stats


def rank_pairs(pair_stats, top_n=30):
    """
    Erstellt zwei Rankings stabiler Cointegration-Paare.
    """
    if pair_stats is None or pair_stats.empty:
        return pd.DataFrame(), pd.DataFrame()

    top_pairs_consistency = (
        pair_stats
        .sort_values(["consistency", "mean_p"], ascending=[False, True])
        .head(top_n)
        .reset_index()
    )

    top_pairs_mean_p = (
        pair_stats
        .sort_values("mean_p", ascending=True)
        .head(top_n)
        .reset_index()
    )

    return top_pairs_consistency, top_pairs_mean_p


# ---------------------------
# Plots: p-values + network
# ---------------------------

def plot_daily_pvalues(daily_coint_df, top_pairs, p_thresh=0.05, top_n_plot=10, title=None):
    if daily_coint_df is None or daily_coint_df.empty:
        raise ValueError("daily_coint_df ist leer.")
    if top_pairs is None or top_pairs.empty:
        raise ValueError("top_pairs ist leer.")

    pairs_to_plot = top_pairs[["symbol1", "symbol2"]].head(top_n_plot).values

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()

    for s1, s2 in pairs_to_plot:
        pair_series = (
            daily_coint_df
            .loc[(daily_coint_df["symbol1"] == s1) & (daily_coint_df["symbol2"] == s2)]
            .sort_values("date")
        )
        ax.plot(pair_series["date"], pair_series["pvalue"], marker="o", linestyle="-", label=f"{s1}-{s2}")

    ax.axhline(p_thresh, linestyle="--", alpha=0.7, color="black", label=f"p-Schwelle {p_thresh}")
    ax.invert_yaxis()
    ax.set_xlabel("Datum")
    ax.set_ylabel("p-Wert")
    ax.tick_params(axis="x", rotation=45)

    if title is None:
        title = f"Tägliche Cointegration (p-Werte) – Top {min(top_n_plot, len(top_pairs))} Paare"
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    return fig


def plot_coint_network(edges_df, title, seed=42, k=4, iterations=300):
    if nx is None:
        raise ImportError("networkx ist nicht installiert (pip install networkx).")

    if edges_df is None or edges_df.empty:
        raise ValueError("edges_df ist leer.")

    required = {"symbol1", "symbol2", "mean_p", "consistency", "n_days", "n_sig_days"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"edges_df fehlt Spalten: {missing}")

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(
            row["symbol1"], row["symbol2"],
            mean_p=float(row["mean_p"]),
            consistency=float(row["consistency"]),
            n_days=int(row["n_days"]),
            n_sig_days=int(row["n_sig_days"])
        )

    if G.number_of_edges() == 0:
        raise ValueError("Keine Kanten im Graphen.")

    fig = plt.figure(figsize=(16, 12))
    ax = plt.gca()

    pos = nx.spring_layout(G, k=k, iterations=iterations, seed=seed)

    degrees = dict(G.degree())
    node_sizes = [2500 + degrees[n] * 900 for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_color="#77bdfb",
        node_size=node_sizes,
        edgecolors="black",
        linewidths=2.0,
        ax=ax
    )
    nx.draw_networkx_edges(G, pos, width=3.5, edge_color="#555555", alpha=0.75, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=20, font_weight="bold", ax=ax)

    edge_labels = {(u, v): f"{d['mean_p']:.3f} | {d['consistency']:.0%}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=16,
        font_color="darkred",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
        ax=ax
    )

    ax.set_title(title, fontsize=26, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------
# Spreads
# ---------------------------

def compute_daily_spread_stats(log_prices, pairs):
    """
    Tägliche Spread-Statistiken (log spread) für gegebene Paare.
    """
    _ensure_datetime_index(log_prices, "log_prices")

    pairs_list = _normalize_pairs(pairs, sort_symbols=True, dedup=True)

    all_days = log_prices.index.normalize().unique()
    records = []

    for s1, s2 in pairs_list:
        if s1 not in log_prices.columns or s2 not in log_prices.columns:
            continue

        for day in all_days:
            day_mask = log_prices.index.normalize() == day
            day_prices = log_prices.loc[day_mask, [s1, s2]].dropna()
            if day_prices.empty:
                continue

            spread = day_prices[s1] - day_prices[s2]

            records.append({
                "date": pd.to_datetime(day),
                "symbol1": s1,
                "symbol2": s2,
                "mean_spread": float(spread.mean()),
                "std_spread": float(spread.std()),
                "min_spread": float(spread.min()),
                "max_spread": float(spread.max()),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values(["symbol1", "symbol2", "date"]).reset_index(drop=True)
    return df


def plot_daily_mean_spreads(daily_spread_df, pairs, title="Tägliche Durchschnitts-Spreads der Top-Paare"):
    if daily_spread_df is None or daily_spread_df.empty:
        raise ValueError("daily_spread_df ist leer.")

    pairs_list = _normalize_pairs(pairs, sort_symbols=True, dedup=True)

    n_pairs = len(pairs_list)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.0 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, (s1, s2) in enumerate(pairs_list):
        ax = axes[idx]

        pair_spreads = (
            daily_spread_df
            .loc[(daily_spread_df["symbol1"] == s1) & (daily_spread_df["symbol2"] == s2)]
            .sort_values("date")
        )

        if pair_spreads.empty:
            ax.set_visible(False)
            continue

        mean_overall = pair_spreads["mean_spread"].mean()

        ax.plot(pair_spreads["date"], pair_spreads["mean_spread"], marker="o", linestyle="-", label="Täglicher Mittelwert")
        ax.axhline(mean_overall, color="red", linestyle="--", label=f"Gesamtmittel: {mean_overall:.4f}")

        ax.set_title(f"{s1} - {s2}: täglicher Mean-Spread")
        ax.set_ylabel("Mean Spread")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Datum")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_intraday_spreads_last_day(log_prices, pairs, title="Intraday-Spreads (letzter Tag) – Top-Paare"):
    _ensure_datetime_index(log_prices, "log_prices")

    pairs_list = _normalize_pairs(pairs, sort_symbols=True, dedup=True)

    last_day = log_prices.index.normalize().max()
    intraday_last = log_prices[log_prices.index.normalize() == last_day]
    if intraday_last.empty:
        raise ValueError("Keine Daten für den letzten Tag gefunden.")

    n_pairs = len(pairs_list)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.0 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, (s1, s2) in enumerate(pairs_list):
        ax = axes[idx]

        if s1 not in intraday_last.columns or s2 not in intraday_last.columns:
            ax.set_visible(False)
            continue

        intraday_pair = intraday_last[[s1, s2]].dropna()
        if intraday_pair.empty:
            ax.set_visible(False)
            continue

        spread = intraday_pair[s1] - intraday_pair[s2]
        mean_spread = spread.mean()

        ax.plot(spread.index, spread, label=f"Spread: {s1} - {s2}", color="steelblue")
        ax.axhline(mean_spread, color="red", linestyle="--", label=f"Tagesmittel: {mean_spread:.4f}")

        ax.set_title(f"Intraday-Spread am {last_day.date()}: {s1} - {s2}")
        ax.set_ylabel("Spread")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Zeit")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_intraday_prices_last_day(
    log_prices,
    top_pairs,
    top_n=5,
    title="Intraday-Preisverläufe – Top stabilste Cointegrations-Paare",
):
    _ensure_datetime_index(log_prices, "log_prices")
    if top_pairs is None or top_pairs.empty:
        raise ValueError("top_pairs ist leer.")

    last_day = log_prices.index.normalize().max()
    intraday = log_prices[log_prices.index.normalize() == last_day]
    if intraday.empty:
        raise ValueError("Keine Intraday-Daten für den letzten Tag gefunden.")

    pairs_to_plot = top_pairs.head(top_n)[["symbol1", "symbol2"]].values

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3.5 * top_n), sharex=True)
    if top_n == 1:
        axes = [axes]

    for idx, (s1, s2) in enumerate(pairs_to_plot):
        aligned = pd.concat([intraday[s1], intraday[s2]], axis=1).dropna()
        aligned.columns = [s1, s2]

        if aligned.empty:
            axes[idx].set_visible(False)
            continue

        p1_lin = np.exp(aligned[s1])
        p2_lin = np.exp(aligned[s2])

        p1_norm = p1_lin / p1_lin.iloc[0]
        p2_norm = p2_lin / p2_lin.iloc[0]

        ax = axes[idx]
        ax.plot(p1_norm, label=s1, color="steelblue")
        ax.plot(p2_norm, label=s2, color="darkorange")

        subtitle = f"{s1} vs {s2} (intraday, {last_day.date()})"
        if "mean_p" in top_pairs.columns and "consistency" in top_pairs.columns:
            row = top_pairs[(top_pairs["symbol1"] == s1) & (top_pairs["symbol2"] == s2)].iloc[0]
            subtitle += f" | mean_p={row['mean_p']:.4f}, consistency={row['consistency']:.0%}"

        ax.set_title(subtitle, fontsize=12)
        ax.set_ylabel("Normierter Intraday-Preis")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Zeit")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig