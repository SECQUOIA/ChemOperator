"""Explore PCA and POD modes for a continuously stirred tank reactor."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import chem_operator  # noqa: F401  # Registers the bundled Cantera data directory.
from chem_operator.example_paths import ExamplePaths


PATHS = ExamplePaths.from_script(__file__)
OUTPUT_DIR = PATHS.example / "results" / "pod_exploration"
MECHANISM = "n-heptane-NUIG-2016.yaml"
INLET_COMPOSITION = {"NC7H16": 0.005, "O2": 0.0275, "HE": 0.9675}
REACTOR_TEMPERATURE = 925.0  # K
REACTOR_PRESSURE = 1.046138 * ct.one_atm  # 1.06 bar
REACTOR_VOLUME = 30.5e-6  # m^3
RESIDENCE_TIME = 2.0  # s
MAX_SIMULATION_TIME = 50.0  # s
dpi = 150

def sample_inlet_composition(
    rng: np.random.Generator,
    phi_range: tuple[float, float] = (0.5, 2.0),
    reactive_fraction_range: tuple[float, float] = (0.02, 0.08),
) -> dict[str, float]:
    """Sample an n-heptane/O2/He inlet composition."""
    phi = rng.uniform(*phi_range)
    reactive_fraction = rng.uniform(*reactive_fraction_range)
    fuel_to_oxygen = phi / 11.0
    x_o2 = reactive_fraction / (1.0 + fuel_to_oxygen)
    return {
        "NC7H16": fuel_to_oxygen * x_o2,
        "O2": x_o2,
        "HE": 1.0 - reactive_fraction,
    }


def simulate_cstr(max_simulation_time: float) -> ct.SolutionArray:
    """Integrate the isothermal CSTR and return every accepted state."""
    gas = ct.Solution(MECHANISM)
    gas.TPX = REACTOR_TEMPERATURE, REACTOR_PRESSURE, INLET_COMPOSITION

    inlet = ct.Reservoir(gas, clone=False)
    exhaust = ct.Reservoir(gas, clone=False)
    reactor = ct.IdealGasMoleReactor(
        gas, clone=False, energy="off", volume=REACTOR_VOLUME
    )
    inlet_flow = ct.MassFlowController(
        upstream=inlet,
        downstream=reactor,
        mdot=lambda _: reactor.mass / RESIDENCE_TIME,
    )
    ct.PressureController(
        upstream=reactor,
        downstream=exhaust,
        primary=inlet_flow,
        K=1e-6,
    )
    network = ct.ReactorNet([reactor])
    states = ct.SolutionArray(gas, extra=["t"])

    started = time.perf_counter()
    t = 0.0
    while t < max_simulation_time:
        t = network.step()
        states.append(reactor.phase.state, t=t)
    elapsed = time.perf_counter() - started
    print(f"Simulation took {elapsed:.2f}s and {len(states)} steps")
    return states


def pod(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the feature mean and covariance eigenvectors in descending order."""
    mean = np.mean(values, axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (len(values) - 1)
    _, modes = np.linalg.eigh(covariance)
    return mean, np.fliplr(modes)


def plot_top_species_correlation(
    states: ct.SolutionArray,
    output_path: Path,
    *,
    n: int = 15,
    annotate: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    """Plot the correlation matrix for species with the largest mean abundance."""
    mole_fractions = pd.DataFrame(states.X, columns=states.species_names)
    top_species = mole_fractions.mean().nlargest(n).index
    correlation = mole_fractions[top_species].corr()

    fig, ax = plt.subplots(figsize=(0.4 * n + 3, 0.4 * n + 2))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(n), labels=top_species, rotation=90)
    ax.set_yticks(range(n), labels=top_species)
    if annotate:
        for row in range(n):
            for column in range(n):
                ax.text(
                    column,
                    row,
                    f"{correlation.iloc[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title(f"Top {n} Species Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return correlation


def plot_pca_pod_variance(
    pca: PCA,
    pod_eigenvalues: np.ndarray,
    output_path: Path,
    *,
    n_components: int = 10,
    dpi: int = 150,
) -> None:
    """Compare PCA and POD explained variance on one axis."""
    pca_ratio = np.asarray(pca.explained_variance_ratio_)
    pod_ratio = pod_eigenvalues / pod_eigenvalues.sum()
    n = min(len(pca_ratio), len(pod_ratio), n_components)
    components = np.arange(1, n + 1)
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        components - width / 2,
        pca_ratio[:n] * 100,
        width=width,
        color="C0",
        alpha=0.6,
        label="PCA individual",
    )
    ax.bar(
        components + width / 2,
        pod_ratio[:n] * 100,
        width=width,
        color="C1",
        alpha=0.6,
        label="POD individual",
    )
    ax.plot(
        components,
        np.cumsum(pca_ratio[:n]) * 100,
        color="C0",
        marker="o",
        linewidth=4,
        label="PCA cumulative",
    )
    ax.plot(
        components,
        np.cumsum(pod_ratio[:n]) * 100,
        color="C1",
        marker="o",
        markersize=3,
        label="POD cumulative",
    )
    ax.set(xlabel="Component", ylabel="Explained variance (%)", ylim=(0, 105))
    ax.legend()
    ax.set_title(
        f"{n} Components - PCA: {pca_ratio[:n].sum():.8%}, "
        f"POD: {pod_ratio[:n].sum():.8%}"
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_time_histories(
    states: ct.SolutionArray, output_dir: Path, *, dpi: int = 150
) -> None:
    """Save the CO mole fraction, temperature, and density histories."""
    histories = (
        (states("CO").X[:, 0], "CO mole fraction", "co_mole_fraction.png"),
        (states.T, "Temperature [K]", "temperature.png"),
        (states.density, "Density [kg/m³]", "density.png"),
    )
    for values, ylabel, filename in histories:
        fig, ax = plt.subplots()
        ax.semilogx(states.t, values, "-o", markersize=3)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def main() -> None:
    if MAX_SIMULATION_TIME <= 0:
        raise ValueError("--max-simulation-time must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running Cantera version: {ct.__version__}")
    rng = np.random.default_rng(seed=42)
    for _ in range(3):
        print("Sample inlet composition:", sample_inlet_composition(rng))

    states = simulate_cstr(MAX_SIMULATION_TIME)
    mole_fractions = pd.DataFrame(states.X, columns=states.species_names)
    print("Most abundant species:\n", mole_fractions.mean().nlargest(20))

    raw_states = np.column_stack((states.T, states.density, states.X))
    feature_names = ["temperature", "density", *states.species_names]
    print("State matrix:", raw_states.shape)
    print("Features:", len(feature_names))

    standardized = StandardScaler().fit_transform(raw_states)
    pca = PCA().fit(standardized)
    pod_mean, pod_modes = pod(standardized)
    pod_coordinates = (standardized - pod_mean) @ pod_modes
    pod_eigenvalues = np.var(pod_coordinates, axis=0, ddof=1)

    n_compare = min(len(pca.explained_variance_), len(pod_eigenvalues))
    differences = np.abs(
        pca.explained_variance_[:n_compare] - pod_eigenvalues[:n_compare]
    )
    print("Maximum PCA/POD eigenvalue difference:", differences.max())
    print("First 10 differences:", differences[:10])

    plot_top_species_correlation(
        states, OUTPUT_DIR / "species_correlation.png", dpi=dpi
    )
    plot_pca_pod_variance(
        pca,
        pod_eigenvalues,
        OUTPUT_DIR / "pca_pod_variance.png",
        dpi=dpi,
    )
    plot_time_histories(states, OUTPUT_DIR, dpi=dpi)
    print(f"Figures written to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
