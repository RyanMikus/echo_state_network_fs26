"""
Echo State Network utilities for damped oscillation prediction.

Contains the ESN model, sequence datasets, reservoir state collection,
readout training, evaluation, and the grid-search helper.
"""

from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Setzt die Seeds für NumPy und PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_oscillation_dataset(data: np.ndarray, name: str) -> None:
    """Prüft die Grundstruktur eines Schwingungsdatensatzes."""
    if len(data) == 0:
        raise ValueError(f"{name} ist leer.")

    required_keys = {"t", "y", "params"}

    for i, osc in enumerate(data):
        missing = required_keys - set(osc.keys())
        if missing:
            raise ValueError(f"{name}[{i}] enthält nicht alle Keys: {missing}")

        if len(osc["t"]) != len(osc["y"]):
            raise ValueError(f"{name}[{i}] hat unterschiedliche Länge für t und y.")

    print(f"{name}: Struktur OK, {len(data)} Schwingungen")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class OscillationSequenceDataset(Dataset):
    """PyTorch-Dataset, das vollständige Schwingungssequenzen liefert."""

    def __init__(self, oscillations: np.ndarray):
        self.oscillations = oscillations

    def __len__(self) -> int:
        return len(self.oscillations)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        osc = self.oscillations[idx]
        return {
            "t": torch.tensor(osc["t"], dtype=torch.float32),
            "y": torch.tensor(osc["y"], dtype=torch.float32),
            "params": osc["params"],
        }


class ReservoirStateDataset(Dataset):
    """Dataset für gesammelte Reservoir-Zustände und Zielwerte."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EchoStateNetwork(nn.Module):
    """Echo State Network mit fixem Reservoir und trainierbarer Ausgabeschicht."""

    def __init__(
        self,
        n_reservoir: int = 300,
        spectral_radius: float = 0.95,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        leaking_rate: float = 1.0,
        seed: int = 42,
        verbose: bool = True,
    ):
        super().__init__()

        if not 0.0 < leaking_rate <= 1.0:
            raise ValueError("leaking_rate muss im Bereich (0, 1] liegen.")
        if not 0.0 < sparsity <= 1.0:
            raise ValueError("sparsity muss im Bereich (0, 1] liegen.")

        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        generator = torch.Generator()
        generator.manual_seed(seed)

        W_in = torch.empty(n_reservoir, 1).uniform_(-1.0, 1.0, generator=generator)
        W_in = input_scaling * W_in
        self.register_buffer("W_in", W_in)

        W = torch.empty(n_reservoir, n_reservoir).uniform_(-1.0, 1.0, generator=generator)
        connection_mask = torch.rand(n_reservoir, n_reservoir, generator=generator) < sparsity
        W = W * connection_mask.float()

        eigenvalues = torch.linalg.eigvals(W)
        current_radius = eigenvalues.abs().max().real

        if current_radius > 0:
            W = W * (spectral_radius / current_radius)
        else:
            raise ValueError("Reservoir-Matrix hat Spektralradius 0. Bitte Seed oder sparsity ändern.")

        self.register_buffer("W", W)
        self.W_out = nn.Linear(n_reservoir, 1)

        if verbose:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            fixed = sum(b.numel() for b in self.buffers())
            print(
                f"ESN: N={n_reservoir}, "
                f"trainierbar={trainable}, "
                f"fix={fixed}, "
                f"spectral_radius={spectral_radius}, "
                f"sparsity={sparsity}"
            )

    def forward(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = u.shape[0]

        if x is None:
            x = torch.zeros(batch_size, self.n_reservoir, device=u.device)

        x_candidate = torch.tanh(u @ self.W_in.T + x @ self.W.T)
        x_new = (1.0 - self.leaking_rate) * x + self.leaking_rate * x_candidate
        y_pred = self.W_out(x_new)

        return y_pred, x_new


# ---------------------------------------------------------------------------
# Reservoir state collection
# ---------------------------------------------------------------------------

def collect_states(
    model: EchoStateNetwork,
    sequence_dataset: OscillationSequenceDataset,
    warmup: int = 100,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sammelt Reservoir-Zustände x(t) und Targets y(t+1) aus Sequenzen."""
    model.eval()

    states: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(len(sequence_dataset)):
            sample = sequence_dataset[i]
            y = sample["y"].to(device)

            x = None
            for t in range(len(y) - 1):
                u = y[t].view(1, 1)
                _, x = model(u, x)

                if t >= warmup:
                    states.append(x.squeeze(0).detach().cpu())
                    targets.append(y[t + 1].view(1).detach().cpu())

    X = torch.stack(states).float()
    y_target = torch.stack(targets).float()

    return X, y_target


# ---------------------------------------------------------------------------
# Readout training
# ---------------------------------------------------------------------------

def train_readout(
    model: EchoStateNetwork,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device = DEVICE,
    print_every: int = 10,
) -> tuple[list[float], list[float]]:
    """Trainiert nur W_out auf gesammelten Reservoir-Zuständen. Gibt NRMSE-Kurven zurück."""
    train_dataset = ReservoirStateDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.W_out.parameters(), lr=lr)

    X_test_device = X_test.to(device)
    y_test_device = y_test.to(device)

    y_range = float((y_test.max() - y_test.min()).item())
    if y_range == 0:
        y_range = 1.0

    train_nrmse_list: list[float] = []
    test_nrmse_list: list[float] = []

    for epoch in range(n_epochs):
        model.train()
        epoch_losses: list[float] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model.W_out(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_pred = model.W_out(X_test_device)
            test_mse = criterion(test_pred, y_test_device).item()

        train_nrmse = float(np.sqrt(np.mean(epoch_losses))) / y_range
        test_nrmse = float(np.sqrt(test_mse)) / y_range

        train_nrmse_list.append(train_nrmse)
        test_nrmse_list.append(test_nrmse)

        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(
                f"Epoche {epoch + 1:3d}/{n_epochs} | "
                f"Train-NRMSE={train_nrmse:.6f} | "
                f"Test-NRMSE={test_nrmse:.6f}"
            )

    return train_nrmse_list, test_nrmse_list


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NRMSE = RMSE / (max(y) - min(y)), normiert über den Wertebereich des Signals."""
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    value_range = float(np.max(y_true) - np.min(y_true))
    if value_range == 0:
        return float("nan")
    return rmse_val / value_range


# ---------------------------------------------------------------------------
# Prediction & evaluation
# ---------------------------------------------------------------------------

def predict_sequence(
    model: EchoStateNetwork,
    osc: dict[str, Any],
    device: torch.device = DEVICE,
) -> dict[str, np.ndarray]:
    """Erzeugt sequenzielle One-Step-Ahead-Vorhersagen für eine Schwingung."""
    model.eval()

    y = torch.tensor(osc["y"], dtype=torch.float32, device=device)
    t_np = np.asarray(osc["t"])

    preds: list[float] = []
    x = None

    with torch.no_grad():
        for step in range(len(y) - 1):
            u = y[step].view(1, 1)
            pred, x = model(u, x)
            preds.append(pred.item())

    return {
        "t": t_np[1:],
        "target": np.asarray(osc["y"])[1:],
        "prediction": np.asarray(preds),
    }


def evaluate_model(
    model: EchoStateNetwork,
    data: np.ndarray,
    warmup: int = 100,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    """Evaluiert ein ESN sequenziell auf allen Schwingungen. Gibt NRMSE pro Sequenz zurück."""
    rows = []

    for idx, osc in enumerate(data):
        result = predict_sequence(model, osc, device=device)

        y_true = result["target"][warmup:]
        y_pred = result["prediction"][warmup:]

        params = osc["params"]
        rows.append(
            {
                "sequence_id": idx,
                "nrmse": nrmse(y_true, y_pred),
                "A": params["A"],
                "alpha": params["alpha"],
                "omega": params["omega"],
                "omega_pi": params["omega"] / np.pi,
                "phi": params["phi"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grid search helper
# ---------------------------------------------------------------------------

def train_and_evaluate_esn(
    n_reservoir: int,
    train_dataset: OscillationSequenceDataset,
    test_dataset: OscillationSequenceDataset,
    test_data_raw: np.ndarray,
    spectral_radius: float = 0.95,
    sparsity: float = 0.1,
    input_scaling: float = 1.0,
    leaking_rate: float = 1.0,
    warmup: int = 100,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
    device: torch.device = DEVICE,
    verbose: bool = False,
    print_every: int = 0,
) -> dict[str, Any]:
    """Trainiert und evaluiert ein ESN für eine gegebene Konfiguration."""
    set_seed(seed)

    model = EchoStateNetwork(
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        input_scaling=input_scaling,
        leaking_rate=leaking_rate,
        seed=seed,
        verbose=verbose,
    ).to(device)

    X_train_local, y_train_local = collect_states(
        model, train_dataset, warmup=warmup, device=device,
    )
    X_test_local, y_test_local = collect_states(
        model, test_dataset, warmup=warmup, device=device,
    )

    train_losses_local, test_losses_local = train_readout(
        model,
        X_train_local,
        y_train_local,
        X_test_local,
        y_test_local,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        print_every=print_every,
    )

    nrmse_df_local = evaluate_model(model, test_data_raw, warmup=warmup, device=device)

    return {
        "model": model,
        "train_losses": train_losses_local,
        "test_losses": test_losses_local,
        "nrmse_df": nrmse_df_local,
        "mean_nrmse": float(nrmse_df_local["nrmse"].mean()),
        "std_nrmse": float(nrmse_df_local["nrmse"].std()),
        "min_nrmse": float(nrmse_df_local["nrmse"].min()),
        "max_nrmse": float(nrmse_df_local["nrmse"].max()),
        "final_train_nrmse": float(train_losses_local[-1]),
        "final_test_nrmse": float(test_losses_local[-1]),
    }


# ---------------------------------------------------------------------------
# Noisy prediction & Monte-Carlo evaluation (for robustness analysis)
# ---------------------------------------------------------------------------

def predict_sequence_noisy(
    model: EchoStateNetwork,
    osc: dict[str, Any],
    sigma: float,
    rng: np.random.Generator,
    device: torch.device = DEVICE,
) -> dict[str, np.ndarray]:
    """One-Step-Ahead-Vorhersage mit verrauschtem Eingangssignal.

    Das verrauschte Signal dient als ESN-Eingang; der Zielwert bleibt das
    saubere Signal (Messrausch-Szenario).
    """
    model.eval()
    y_clean = np.asarray(osc["y"])
    noise = rng.normal(0.0, sigma, size=y_clean.shape) if sigma > 0 else np.zeros_like(y_clean)
    y_noisy = y_clean + noise

    y_tensor = torch.tensor(y_noisy, dtype=torch.float32, device=device)
    preds: list[float] = []
    x = None

    with torch.no_grad():
        for step in range(len(y_tensor) - 1):
            u = y_tensor[step].view(1, 1)
            pred, x = model(u, x)
            preds.append(pred.item())

    return {
        "t": np.asarray(osc["t"])[1:],
        "target": y_clean[1:],
        "noisy_input": y_noisy[:-1],
        "prediction": np.asarray(preds),
    }


def evaluate_model_noisy(
    model: EchoStateNetwork,
    data: np.ndarray,
    sigma_rel: float,
    n_mc: int = 10,
    warmup: int = 100,
    seed: int = 42,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    """Monte-Carlo-Evaluation bei einem relativen Rauschpegel sigma_rel.

    sigma_rel = sigma / (max(y) - min(y)), berechnet pro Sequenz.
    Gibt mean, std und alle per-MC-NRMSE-Werte zurück.
    """
    rng = np.random.default_rng(seed)
    mc_nrmses: list[float] = []

    for _ in range(n_mc):
        seq_nrmses: list[float] = []
        for osc in data:
            y_range = float(np.max(osc["y"]) - np.min(osc["y"]))
            sigma = sigma_rel * y_range if y_range > 0 else 0.0
            result = predict_sequence_noisy(model, osc, sigma, rng, device=device)
            y_true = result["target"][warmup:]
            y_pred = result["prediction"][warmup:]
            seq_nrmses.append(nrmse(y_true, y_pred))
        mc_nrmses.append(float(np.mean(seq_nrmses)))

    return {
        "mean": float(np.mean(mc_nrmses)),
        "std": float(np.std(mc_nrmses)),
        "all": mc_nrmses,
    }
