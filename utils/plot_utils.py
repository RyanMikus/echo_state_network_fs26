"""
Visualization utilities for ESN training and evaluation results.

All functions produce matplotlib or plotly figures and call plt.show()
or fig.show() directly. They do not return figure objects.
"""

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

from .esn_utils import EchoStateNetwork, predict_sequence, predict_sequence_noisy, nrmse


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    title: str,
    skip_first: bool = False,
    train_stds: list[float] | None = None,
    test_stds: list[float] | None = None,
) -> None:
    """Plottet Train- und Test-NRMSE über die Epochen, optional mit Standardabweichungsbändern."""
    train_plot = np.array(train_losses[1:] if skip_first else train_losses)
    test_plot = np.array(test_losses[1:] if skip_first else test_losses)
    start = 2 if skip_first else 1
    epochs = np.arange(start, start + len(train_plot))

    plt.figure(figsize=(9, 4))

    plt.plot(epochs, train_plot, label="Train-NRMSE", marker="o", markersize=3)
    if train_stds is not None:
        train_std_plot = np.array(train_stds[1:] if skip_first else train_stds)
        plt.fill_between(epochs, train_plot - train_std_plot, train_plot + train_std_plot, alpha=0.2)

    plt.plot(epochs, test_plot, label="Test-NRMSE", marker="s", markersize=3)
    if test_stds is not None:
        test_std_plot = np.array(test_stds[1:] if skip_first else test_stds)
        plt.fill_between(epochs, test_plot - test_std_plot, test_plot + test_std_plot, alpha=0.2)

    plt.title(title)
    plt.xlabel("Epoche")
    plt.ylabel("NRMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Sequence predictions
# ---------------------------------------------------------------------------

def plot_sequence_predictions(
    model: EchoStateNetwork,
    data: np.ndarray,
    nrmse_df: pd.DataFrame,
    n_examples: int = 6,
    warmup: int = 100,
    device: Any = None,
) -> None:
    """Plottet echte und vorhergesagte Schwingungen für einige Beispiele."""
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_examples = min(n_examples, len(data))
    n_cols = 2
    n_rows = int(np.ceil(n_examples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.array(axes).flatten()
    fig.suptitle("Echte Schwingung vs. ESN-Vorhersage", fontsize=16)

    for i in range(n_examples):
        osc = data[i]
        result = predict_sequence(model, osc, device=device)
        params = osc["params"]

        axes[i].plot(result["t"][warmup:], result["target"][warmup:], label="Echt")
        axes[i].plot(result["t"][warmup:], result["prediction"][warmup:], linestyle="--", label="Vorhersage")
        axes[i].set_title(
            f"Seq {i} | NRMSE={nrmse_df.loc[i, 'nrmse']:.4f}\n"
            f"A={params['A']:.2f}, α={params['alpha']:.3f}, ω={params['omega'] / np.pi:.2f}π"
        )
        axes[i].set_xlabel("t")
        axes[i].set_ylabel("y")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    for j in range(n_examples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# NRMSE distribution
# ---------------------------------------------------------------------------

def plot_nrmse_distribution(nrmse_df: pd.DataFrame, title_suffix: str = "") -> None:
    """Plottet NRMSE pro Sequenz als Balken und als Histogramm."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(nrmse_df["sequence_id"], nrmse_df["nrmse"])
    axes[0].set_title(f"NRMSE pro Testschwingung{title_suffix}")
    axes[0].set_xlabel("Sequenz-ID")
    axes[0].set_ylabel("NRMSE")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].hist(nrmse_df["nrmse"], bins=10)
    axes[1].set_title(f"Verteilung der NRMSE-Werte{title_suffix}")
    axes[1].set_xlabel("NRMSE")
    axes[1].set_ylabel("Anzahl")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Error over time
# ---------------------------------------------------------------------------

def plot_error_over_time(
    model: EchoStateNetwork,
    data: np.ndarray,
    nrmse_df: pd.DataFrame,
    warmup: int = 100,
    device: Any = None,
) -> None:
    """Plottet den absoluten Fehler über Zeit für beste, mittlere und schlechteste Sequenz."""
    import torch
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sorted_idx = nrmse_df["nrmse"].argsort().values
    selected = {
        "Beste": int(sorted_idx[0]),
        "Mittlere": int(sorted_idx[len(sorted_idx) // 2]),
        "Schlechteste": int(sorted_idx[-1]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Absoluter Fehler über Zeit", fontsize=14)

    for ax, (label, idx) in zip(axes, selected.items()):
        osc = data[idx]
        result = predict_sequence(model, osc, device=device)
        error = np.abs(result["target"][warmup:] - result["prediction"][warmup:])
        nrmse_val = nrmse_df.loc[idx, "nrmse"]
        params = osc["params"]

        ax.plot(result["t"][warmup:], error, color="tab:orange")
        ax.set_title(f"{label} (NRMSE={nrmse_val:.4f})\nA={params['A']:.2f}, α={params['alpha']:.3f}")
        ax.set_xlabel("t")
        ax.set_ylabel("|Fehler|")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# NRMSE vs. oscillation parameters
# ---------------------------------------------------------------------------

def plot_nrmse_vs_params(nrmse_df: pd.DataFrame, title_suffix: str = "") -> None:
    """Plottet NRMSE gegen Amplitude, Dämpfung und Frequenz der Testschwingungen."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    suptitle = "NRMSE vs. Schwingungsparameter"
    if title_suffix:
        suptitle += f" {title_suffix}"
    fig.suptitle(suptitle, fontsize=14)

    params_to_plot = [
        ("A", "Amplitude $A$"),
        ("alpha", "Dämpfung $\\alpha$"),
        ("omega_pi", "Frequenz $\\omega/\\pi$"),
    ]

    for ax, (col, label) in zip(axes, params_to_plot):
        ax.scatter(nrmse_df[col], nrmse_df["nrmse"], alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel("NRMSE")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Grid search heatmap
# ---------------------------------------------------------------------------

def plot_grid_search_heatmap(
    grid_summary: pd.DataFrame,
    best_configs: np.ndarray,
) -> None:
    """Plottet die Grid-Search-Ergebnisse als Heatmap (Mittlerer NRMSE)."""
    heatmap_data = grid_summary.pivot(index="n_reservoir", columns="spectral_radius", values="mean_nrmse")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels([f"{sr:.2f}" for sr in heatmap_data.columns])
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    vmin = heatmap_data.values.min()
    vmax = heatmap_data.values.max()
    threshold = vmin + (vmax - vmin) * 0.5

    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[i, j]
            text_color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8, color=text_color)

    for _n, _sr in best_configs:
        try:
            _ci = list(heatmap_data.columns).index(_sr)
            _ri = list(heatmap_data.index).index(_n)
            ax.add_patch(Rectangle(
                (_ci - 0.5, _ri - 0.5), 1, 1,
                fill=False, edgecolor="lime", linewidth=2.5, zorder=5,
            ))
        except ValueError:
            pass

    plt.colorbar(im, ax=ax, label="Mittlerer NRMSE")
    ax.set_xlabel("Spektralradius")
    ax.set_ylabel("Reservoir-Grösse N")
    ax.set_title("Grid Search: Mittlerer NRMSE (gemittelt über Seeds)")
    plt.tight_layout()
    plt.show()

    return heatmap_data


def plot_grid_search_3d(
    grid_summary: pd.DataFrame,
    heatmap_data: pd.DataFrame,
    best_configs: np.ndarray,
) -> None:
    """Interaktiver 3D-Plot: Mittlerer NRMSE mit ±1-Std-Band."""
    std_pivot = grid_summary.pivot(index="n_reservoir", columns="spectral_radius", values="std_nrmse")

    x_3d = [float(sr) for sr in heatmap_data.columns]
    y_3d = [int(n) for n in heatmap_data.index]
    z_mean_3d = heatmap_data.values
    z_upper_3d = z_mean_3d + std_pivot.values
    z_lower_3d = np.clip(z_mean_3d - std_pivot.values, 0, None)

    fig_3d = go.Figure()

    fig_3d.add_trace(go.Surface(
        x=x_3d, y=y_3d, z=z_mean_3d,
        colorscale="Plasma", reversescale=True,
        name="Mittlerer NRMSE", showscale=True,
        colorbar=dict(title="NRMSE", x=1.02),
        opacity=1.0, showlegend=True,
    ))
    fig_3d.add_trace(go.Surface(
        x=x_3d, y=y_3d, z=z_upper_3d,
        colorscale=[[0, "rgba(255,140,0,0.3)"], [1, "rgba(255,140,0,0.3)"]],
        showscale=False, name="±1 Std", opacity=0.35,
        showlegend=True, legendgroup="std_band",
    ))
    fig_3d.add_trace(go.Surface(
        x=x_3d, y=y_3d, z=z_lower_3d,
        colorscale=[[0, "rgba(255,140,0,0.3)"], [1, "rgba(255,140,0,0.3)"]],
        showscale=False, name="±1 Std (unten)", opacity=0.35,
        showlegend=False, legendgroup="std_band",
    ))

    _z_floor = float(np.nanmin(z_lower_3d))
    _z_ceil = float(np.nanmax(z_upper_3d))

    for _n, _sr in best_configs:
        fig_3d.add_trace(go.Scatter3d(
            x=[float(_sr), float(_sr)], y=[int(_n), int(_n)],
            z=[_z_floor, _z_ceil], mode="lines",
            line=dict(color="red", width=6),
            name="Bestes Modell", showlegend=True,
        ))

    fig_3d.update_layout(
        title="Grid Search: Mittlerer NRMSE mit ±1 Standardabweichung (interaktiv)",
        scene=dict(
            xaxis_title="Spektralradius",
            yaxis_title="Reservoir-Grösse N",
            zaxis_title="NRMSE",
        ),
        legend=dict(x=0.02, y=0.95, bgcolor="rgba(255,255,255,0.75)", bordercolor="gray", borderwidth=1),
        height=650, margin=dict(l=0, r=0, b=0, t=50),
    )
    fig_3d.show()


def plot_grid_search_rel_std_heatmap(
    grid_summary: pd.DataFrame,
    best_configs: np.ndarray,
) -> pd.DataFrame:
    """Plottet die relative Standardabweichung des NRMSE als Heatmap."""
    rel_std_pivot = grid_summary.pivot(index="n_reservoir", columns="spectral_radius", values="rel_std_nrmse")

    fig_rsd, ax_rsd = plt.subplots(figsize=(10, 5))
    im_rsd = ax_rsd.imshow(rel_std_pivot.values, aspect="auto", cmap="YlOrRd")

    ax_rsd.set_xticks(range(len(rel_std_pivot.columns)))
    ax_rsd.set_xticklabels([f"{sr:.2f}" for sr in rel_std_pivot.columns])
    ax_rsd.set_yticks(range(len(rel_std_pivot.index)))
    ax_rsd.set_yticklabels(rel_std_pivot.index)

    vmin_rsd = rel_std_pivot.values.min()
    vmax_rsd = rel_std_pivot.values.max()
    threshold_rsd = vmin_rsd + (vmax_rsd - vmin_rsd) * 0.5

    for i in range(len(rel_std_pivot.index)):
        for j in range(len(rel_std_pivot.columns)):
            val = rel_std_pivot.iloc[i, j]
            text_color = "white" if val > threshold_rsd else "black"
            ax_rsd.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8, color=text_color)

    for _n, _sr in best_configs:
        try:
            _ci = list(rel_std_pivot.columns).index(_sr)
            _ri = list(rel_std_pivot.index).index(_n)
            ax_rsd.add_patch(Rectangle(
                (_ci - 0.5, _ri - 0.5), 1, 1,
                fill=False, edgecolor="lime", linewidth=2.5, zorder=5,
            ))
        except ValueError:
            pass

    plt.colorbar(im_rsd, ax=ax_rsd, label="Rel. Std. NRMSE [%]")
    ax_rsd.set_xlabel("Spektralradius")
    ax_rsd.set_ylabel("Reservoir-Grösse N")
    ax_rsd.set_title("Grid Search: Relative Standardabweichung des NRMSE (Std / Median, in %)")
    plt.tight_layout()
    plt.show()

    return rel_std_pivot


def plot_grid_search_rel_std_3d(
    rel_std_pivot: pd.DataFrame,
    best_configs: np.ndarray,
) -> None:
    """Interaktiver 3D-Plot: Relative Standardabweichung des NRMSE."""
    x_rsd = [float(sr) for sr in rel_std_pivot.columns]
    y_rsd = [int(n) for n in rel_std_pivot.index]
    z_rsd = rel_std_pivot.values

    fig_rsd_3d = go.Figure()
    fig_rsd_3d.add_trace(go.Surface(
        x=x_rsd, y=y_rsd, z=z_rsd,
        colorscale="YlOrRd",
        colorbar=dict(title="Rel. Std. [%]"),
        name="Rel. Std. NRMSE",
    ))

    _z_floor_rsd = float(np.nanmin(z_rsd))
    _z_ceil_rsd = float(np.nanmax(z_rsd))

    for _n, _sr in best_configs:
        fig_rsd_3d.add_trace(go.Scatter3d(
            x=[float(_sr), float(_sr)], y=[int(_n), int(_n)],
            z=[_z_floor_rsd, _z_ceil_rsd], mode="lines",
            line=dict(color="red", width=6),
            name="Bestes Modell", showlegend=True,
        ))

    fig_rsd_3d.update_layout(
        title="Grid Search: Relative Standardabweichung des NRMSE (Std / Median, in %)",
        scene=dict(
            xaxis_title="Spektralradius",
            yaxis_title="Reservoir-Grösse N",
            zaxis_title="Rel. Std. NRMSE [%]",
        ),
        height=650, margin=dict(l=0, r=0, b=0, t=50),
    )
    fig_rsd_3d.show()


# ---------------------------------------------------------------------------
# Noise robustness plots
# ---------------------------------------------------------------------------

def plot_nrmse_vs_noise(
    sigma_levels: list[float],
    results_baseline: list[dict],
    results_opt: list[dict],
    ref_baseline: float,
    ref_opt: float,
) -> None:
    """NRMSE vs. relativer Rauschpegel für Baseline und optimiertes Modell."""
    means_b = np.array([r["mean"] for r in results_baseline])
    stds_b  = np.array([r["std"]  for r in results_baseline])
    means_o = np.array([r["mean"] for r in results_opt])
    stds_o  = np.array([r["std"]  for r in results_opt])
    sigma_arr = np.array(sigma_levels)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(sigma_arr, means_b, "o-", label="Baseline (N=300, SR=0.95)", color="tab:blue")
    ax.fill_between(sigma_arr, means_b - stds_b, means_b + stds_b, alpha=0.2, color="tab:blue")

    ax.plot(sigma_arr, means_o, "s-", label="Optimiert (N=400, SR=0.98)", color="tab:orange")
    ax.fill_between(sigma_arr, means_o - stds_o, means_o + stds_o, alpha=0.2, color="tab:orange")

    ax.axhline(ref_baseline, linestyle=":", color="tab:blue",   alpha=0.6, linewidth=1.5,
               label=f"Baseline (sauber): {ref_baseline:.4f}")
    ax.axhline(ref_opt,      linestyle=":", color="tab:orange", alpha=0.6, linewidth=1.5,
               label=f"Optimiert (sauber): {ref_opt:.4f}")

    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"Relativer Rauschpegel $\sigma_{\mathrm{rel}}$")
    ax.set_ylabel("Mittlerer NRMSE")
    ax.set_title("NRMSE vs. Rauschpegel — Baseline vs. Optimiertes Modell")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()


def plot_noisy_predictions(
    models: dict[str, "EchoStateNetwork"],
    data: np.ndarray,
    sigma_rels: list[float],
    warmup: int = 100,
    seed: int = 42,
    device: Any = None,
) -> None:
    """Vorhersagen einer Sequenz unter verschiedenen Rauschpegeln für mehrere Modelle.

    Layout: eine Zeile pro Modell, eine Spalte pro Rauschpegel.
    Sequenz 0 wird als repräsentatives Beispiel verwendet.
    """
    import torch as _torch
    if device is None:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    osc = data[0]
    y_clean = np.asarray(osc["y"])
    y_range = float(np.max(y_clean) - np.min(y_clean))
    t_np = np.asarray(osc["t"])[1:]
    params = osc["params"]

    model_names = list(models.keys())
    n_rows = len(model_names)
    n_cols = len(sigma_rels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(
        f"Vorhersagen unter Rauschen — Sequenz 0 "
        f"(A={params['A']:.2f}, α={params['alpha']:.3f}, ω/π={params['omega'] / np.pi:.2f})",
        fontsize=13,
    )

    for row_i, model_name in enumerate(model_names):
        model = models[model_name]
        for col_i, sigma_rel in enumerate(sigma_rels):
            rng = np.random.default_rng(seed + col_i)
            sigma = sigma_rel * y_range
            result = predict_sequence_noisy(model, osc, sigma, rng, device=device)

            ax = axes[row_i][col_i]
            ax.plot(t_np[warmup:], result["target"][warmup:], label="Ziel (sauber)", alpha=0.9)
            ax.plot(t_np[warmup:], result["noisy_input"][warmup:], color="gray",
                    alpha=0.35, linewidth=0.8, label="Eingang (verrauscht)")
            ax.plot(t_np[warmup:], result["prediction"][warmup:], "--",
                    color="tab:orange", label="Vorhersage")

            nrmse_val = nrmse(result["target"][warmup:], result["prediction"][warmup:])
            ax.set_title(
                f"{model_name}\n"
                r"$\sigma_{\rm rel}$" + f"={sigma_rel}, NRMSE={nrmse_val:.4f}"
            )
            ax.set_xlabel("t")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.3)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_nrmse_distribution_noisy(
    models: dict[str, "EchoStateNetwork"],
    data: np.ndarray,
    sigma_rels: list[float],
    warmup: int = 100,
    seed: int = 42,
    device: Any = None,
) -> None:
    """NRMSE pro Sequenz als Balkendiagramm bei ausgewählten Rauschpegeln.

    Layout: eine Zeile pro Modell, eine Spalte pro Rauschpegel.
    """
    import torch as _torch
    if device is None:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    model_names = list(models.keys())
    n_rows = len(model_names)
    n_cols = len(sigma_rels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("NRMSE-Verteilung über Testsequenzen bei ausgewählten Rauschpegeln", fontsize=13)

    for row_i, model_name in enumerate(model_names):
        model = models[model_name]
        for col_i, sigma_rel in enumerate(sigma_rels):
            rng = np.random.default_rng(seed)
            seq_nrmses: list[float] = []
            for osc in data:
                y_range = float(np.max(osc["y"]) - np.min(osc["y"]))
                sigma = sigma_rel * y_range if y_range > 0 else 0.0
                result = predict_sequence_noisy(model, osc, sigma, rng, device=device)
                y_true = result["target"][warmup:]
                y_pred = result["prediction"][warmup:]
                seq_nrmses.append(nrmse(y_true, y_pred))

            ax = axes[row_i][col_i]
            ax.bar(range(len(seq_nrmses)), seq_nrmses)
            ax.set_title(
                f"{model_name}\n"
                r"$\sigma_{\rm rel}$" + f"={sigma_rel}"
            )
            ax.set_xlabel("Sequenz-ID")
            ax.set_ylabel("NRMSE")
            ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()
