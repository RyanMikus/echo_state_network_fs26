"""
Data generation and visualization utilities for damped oscillations.

The functions in this module are used to generate synthetic damped
oscillation datasets for training, in-distribution testing, and
out-of-distribution evaluation of Echo State Networks.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def damped_oscillation(
    t: np.ndarray,
    A: float,
    alpha: float,
    omega: float,
    phi: float,
) -> np.ndarray:
    """
    Compute a damped sinusoidal oscillation.

    The oscillation is defined as:

        y(t) = A * exp(-alpha * t) * sin(omega * t + phi)
    """
    return A * np.exp(-alpha * t) * np.sin(omega * t + phi)


def create_oscillation(
    t: np.ndarray,
    A: float,
    alpha: float,
    omega: float,
    phi: float,
) -> dict[str, Any]:
    """
    Create a single damped oscillation sample.
    """
    y = damped_oscillation(t, A, alpha, omega, phi)

    return {
        "t": t,
        "y": y,
        "params": {
            "A": A,
            "alpha": alpha,
            "omega": omega,
            "phi": phi,
        },
    }


def generate_diverse_oscillations(
    n: int,
    t: np.ndarray,
    param_ranges: dict[str, tuple[float, float]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate damped oscillations with structured parameter coverage.

    Amplitude and frequency are distributed approximately evenly over
    their respective parameter ranges. The damping coefficient is sampled
    from weak, medium, and strong damping intervals derived from the
    provided alpha range. The phase is sampled randomly.

    This function is intended for creating training data with broad
    coverage of the in-distribution parameter space.
    """
    rng = np.random.default_rng(seed)
    oscillations = []

    A_min, A_max = param_ranges["A"]
    alpha_min, alpha_max = param_ranges["alpha"]
    omega_factor_min, omega_factor_max = param_ranges["omega_factor"]
    phi_min, phi_max = param_ranges["phi"]

    A_values = np.linspace(A_min, A_max, n)
    omega_factor_values = np.linspace(omega_factor_min, omega_factor_max, n)

    alpha_edges = np.linspace(alpha_min, alpha_max, 4)
    alpha_ranges = [
        (alpha_edges[0], alpha_edges[1]),
        (alpha_edges[1], alpha_edges[2]),
        (alpha_edges[2], alpha_edges[3]),
    ]

    for i in range(n):
        A = A_values[i] + rng.uniform(-0.05, 0.05)
        A = np.clip(A, A_min, A_max)

        alpha_low, alpha_high = alpha_ranges[i % 3]
        alpha = rng.uniform(alpha_low, alpha_high)

        omega_factor = omega_factor_values[i] + rng.uniform(-0.1, 0.1)
        omega_factor = np.clip(
            omega_factor,
            omega_factor_min,
            omega_factor_max,
        )
        omega = omega_factor * np.pi

        phi = rng.uniform(phi_min, phi_max)

        oscillations.append(create_oscillation(t, A, alpha, omega, phi))

    return oscillations


def generate_random_oscillations(
    n: int,
    t: np.ndarray,
    param_ranges: dict[str, tuple[float, float]],
    seed: int,
) -> list[dict[str, Any]]:
    """
    Generate damped oscillations by random parameter sampling.

    This function is used for in-distribution test data and
    out-of-distribution test data.
    """
    rng = np.random.default_rng(seed)
    oscillations = []

    for _ in range(n):
        A = rng.uniform(*param_ranges["A"])
        alpha = rng.uniform(*param_ranges["alpha"])
        omega_factor = rng.uniform(*param_ranges["omega_factor"])
        omega = omega_factor * np.pi
        phi = rng.uniform(*param_ranges["phi"])

        oscillations.append(create_oscillation(t, A, alpha, omega, phi))

    return oscillations


def save_dataset(path: str, data: list[dict[str, Any]]) -> None:
    """
    Save a dataset as a NumPy .npy file.
    """
    np.save(path, data)
    print(f"{path}: saved, {len(data)} oscillations")


def plot_oscillation_grid(
    oscillations: list[dict[str, Any]],
    title: str,
    max_plots: int = 25,
) -> None:
    """
    Plot up to 25 oscillations in a grid.
    """
    n_plots = min(len(oscillations), max_plots)
    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    fig.suptitle(title, fontsize=16)

    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n_plots:
            osc = oscillations[i]
            params = osc["params"]

            ax.plot(osc["t"], osc["y"])
            ax.set_title(
                f"A={params['A']:.2f}, "
                f"alpha={params['alpha']:.2f}, "
                f"omega={params['omega'] / np.pi:.2f}π",
                fontsize=9,
            )
            ax.set_xlabel("t")
            ax.set_ylabel("y")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_ood_examples(
    ood_sets: dict[str, list[dict[str, Any]]],
    n_examples: int = 3,
) -> None:
    """
    Plot example oscillations from multiple out-of-distribution datasets.
    """
    n_rows = len(ood_sets)

    fig, axes = plt.subplots(
        n_rows,
        n_examples,
        figsize=(5 * n_examples, 3 * n_rows),
    )
    fig.suptitle("Examples from the out-of-distribution datasets", fontsize=16)

    axes = np.array(axes)

    for row_idx, (name, data) in enumerate(ood_sets.items()):
        for col_idx in range(n_examples):
            ax = axes[row_idx, col_idx]
            osc = data[col_idx]
            params = osc["params"]

            ax.plot(osc["t"], osc["y"])
            ax.set_title(
                f"{name}\n"
                f"A={params['A']:.2f}, "
                f"alpha={params['alpha']:.2f}, "
                f"omega={params['omega'] / np.pi:.2f}π",
                fontsize=9,
            )
            ax.set_xlabel("t")
            ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


def extract_parameter_table(
    name: str,
    oscillations: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Extract oscillation parameters into a pandas DataFrame.
    """
    rows = []

    for osc in oscillations:
        params = osc["params"]
        rows.append(
            {
                "dataset": name,
                "A": params["A"],
                "alpha": params["alpha"],
                "omega": params["omega"],
                "omega_pi": params["omega"] / np.pi,
                "phi": params["phi"],
            }
        )

    return pd.DataFrame(rows)


def create_parameter_summary(df_params: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact summary of parameter ranges per dataset.
    """
    summary_rows = []

    for dataset_name, group in df_params.groupby("dataset"):
        summary_rows.append(
            {
                "Dataset": dataset_name,
                "A range": f"{group['A'].min():.2f}–{group['A'].max():.2f}",
                "alpha range": (
                    f"{group['alpha'].min():.2f}–"
                    f"{group['alpha'].max():.2f}"
                ),
                "omega range": (
                    f"{group['omega_pi'].min():.2f}π–"
                    f"{group['omega_pi'].max():.2f}π"
                ),
                "Count": len(group),
            }
        )

    return pd.DataFrame(summary_rows)


def plot_parameter_distribution(
    df: pd.DataFrame,
    parameter: str,
    title: str,
    xlabel: str,
) -> None:
    """
    Plot the distribution of a selected parameter for all datasets.
    """
    plt.figure(figsize=(10, 5))

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        plt.hist(subset[parameter], alpha=0.4, label=dataset, bins=10)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
