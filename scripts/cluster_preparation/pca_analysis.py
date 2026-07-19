#!/usr/bin/env python3
"""Run exploratory PCA on the final PF or TF clustering variables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATHS: Dict[str, Path] = {
    "PF": PROJECT_ROOT / "outputs_correlation" / "pf" / "pf_for_PCA.csv",
    "TF": PROJECT_ROOT / "outputs_correlation" / "tf" / "tf_for_PCA.csv",
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output_pca"

PF_FEATURES = (
    "Per_extent",
    "N_PFlossR_2kiw",
    "N_PFlossR_2kmw",
    "PFResident_lossR",
    "PFDaynight_lossR",
    "PFAB5k_NOR",
    "PFABC_NOR",
)
TF_FEATURES = (
    "Tem_extent",
    "N_TFlossR_2kiw",
    "N_TFlossR_2kmw",
    "TFResident_lossR",
    "TFDaynight_lossR",
    "TFAB5k_NOR",
    "TFABC_NOR",
)
PF_METADATA_COLUMNS = ("fid", "MS_ID", "PF_Index_Risk_equal")
TF_METADATA_COLUMNS = ("fid", "MS_ID", "TF_Index_Risk_equal")

SCENARIO_CONFIG: Dict[str, Dict[str, Sequence[str]]] = {
    "PF": {"features": PF_FEATURES, "metadata": PF_METADATA_COLUMNS},
    "TF": {"features": TF_FEATURES, "metadata": TF_METADATA_COLUMNS},
}

@dataclass
class PreparedDataset:
    scenario: str
    metadata: pd.DataFrame
    features: pd.DataFrame
    fill_counts: pd.Series


@dataclass
class PcaResult:
    summary: pd.DataFrame
    loadings: pd.DataFrame
    full_scores: pd.DataFrame
    two_d_scores: pd.DataFrame
    correlation_coordinates: pd.DataFrame


def normalize_scenario(value: str) -> str:
    scenario = value.strip().upper()
    if scenario not in SCENARIO_CONFIG:
        raise ValueError("Scenario must be PF or TF.")
    return scenario


def prompt_scenario() -> str:
    while True:
        value = input("Select scenario (PF/TF): ").strip()
        try:
            return normalize_scenario(value)
        except ValueError:
            print("Please enter PF or TF.")


def _scenario_features(scenario: str) -> Sequence[str]:
    return SCENARIO_CONFIG[scenario]["features"]


def _scenario_metadata(scenario: str) -> Sequence[str]:
    return SCENARIO_CONFIG[scenario]["metadata"]


def load_data(input_path: Path, scenario: str) -> PreparedDataset:
    """Load one scenario and keep only metadata plus explicit PCA variables."""
    scenario = normalize_scenario(scenario)
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)
    features = list(_scenario_features(scenario))
    metadata_columns = list(_scenario_metadata(scenario))
    required_columns = metadata_columns + features
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns for "
            f"{scenario}: {', '.join(missing_columns)}"
        )

    for column in features:
        if not is_numeric_dtype(data[column]):
            raise ValueError(f"PCA feature must be numeric: {column}")
        values = data[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"PCA feature contains infinity: {column}")

    feature_values = data.loc[:, features].copy()
    fill_counts = feature_values.isna().sum().astype(int)
    feature_values = feature_values.fillna(0)
    if feature_values.isna().any().any():
        raise ValueError("Missing feature values remain after fill-zero imputation.")
    if len(feature_values) < 2:
        raise ValueError("PCA requires at least two observations.")

    return PreparedDataset(
        scenario=scenario,
        metadata=data.loc[:, metadata_columns].copy(),
        features=feature_values,
        fill_counts=fill_counts,
    )


def standardize(features: pd.DataFrame) -> np.ndarray:
    """Standardize PCA variables with StandardScaler."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for PCA standardization. "
            "Install/use the project environment from 2026master.yml."
        ) from error

    scaler = StandardScaler()
    return scaler.fit_transform(features.to_numpy(dtype=float))


def run_pca(values: np.ndarray, feature_names: Sequence[str]) -> PcaResult:
    """Run full PCA for tables and 2-component PCA for visual projection."""
    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for PCA analysis. "
            "Install/use the project environment from 2026master.yml."
        ) from error

    full_pca = PCA(n_components=None)
    full_scores_array = full_pca.fit_transform(values)

    two_d_pca = PCA(n_components=2)
    two_d_scores_array = two_d_pca.fit_transform(values)

    component_names = [
        f"PC{index}" for index in range(1, len(full_pca.explained_variance_) + 1)
    ]
    summary = pd.DataFrame(
        {
            "Component": component_names,
            "Eigenvalue": full_pca.explained_variance_,
            "Explained Variance Ratio": full_pca.explained_variance_ratio_,
            "Cumulative Explained Variance": np.cumsum(
                full_pca.explained_variance_ratio_
            ),
        }
    )

    loadings = pd.DataFrame(
        full_pca.components_.T,
        index=list(feature_names),
        columns=component_names,
    )
    loadings.insert(0, "Variable", loadings.index)
    loadings = loadings.reset_index(drop=True)

    full_scores = pd.DataFrame(full_scores_array, columns=component_names)
    two_d_scores = pd.DataFrame(two_d_scores_array, columns=["PC1", "PC2"])

    correlation_coordinates = pd.DataFrame(
        full_pca.components_.T[:, :2] * np.sqrt(full_pca.explained_variance_[:2]),
        index=list(feature_names),
        columns=["PC1", "PC2"],
    )
    correlation_coordinates.insert(0, "Variable", correlation_coordinates.index)
    correlation_coordinates = correlation_coordinates.reset_index(drop=True)

    return PcaResult(
        summary=summary,
        loadings=loadings,
        full_scores=full_scores,
        two_d_scores=two_d_scores,
        correlation_coordinates=correlation_coordinates,
    )


def save_tables(dataset: PreparedDataset, result: PcaResult, output_dir: Path) -> None:
    """Save PCA summary, loadings, and metadata-preserving scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "pca_summary.csv", index=False)
    result.loadings.to_csv(output_dir / "pca_loadings.csv", index=False)
    scores = pd.concat(
        [
            dataset.metadata.reset_index(drop=True),
            result.full_scores.reset_index(drop=True),
        ],
        axis=1,
    )
    scores.to_csv(output_dir / "pca_scores.csv", index=False)


def plot_scree(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        summary["Component"],
        summary["Explained Variance Ratio"],
        marker="o",
        color="black",
    )
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Explained Variance Ratio")
    axis.set_title("Explained Variance Ratio")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_cumulative_variance(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        summary["Component"],
        summary["Cumulative Explained Variance"],
        marker="o",
        color="black",
    )
    for threshold in (0.80, 0.90, 0.95):
        axis.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"{int(threshold * 100)}%",
        )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Cumulative Explained Variance")
    axis.set_title("Cumulative Explained Variance")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_scatter(result: PcaResult, output_path: Path) -> None:
    pc1_ratio = result.summary.loc[0, "Explained Variance Ratio"] * 100
    pc2_ratio = result.summary.loc[1, "Explained Variance Ratio"] * 100
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        result.two_d_scores["PC1"],
        result.two_d_scores["PC2"],
        color="gray",
        s=8,
        alpha=0.65,
        rasterized=True,
    )
    axis.set_xlabel(f"PC1 ({pc1_ratio:.1f}%)")
    axis.set_ylabel(f"PC2 ({pc2_ratio:.1f}%)")
    axis.set_title("PCA Projection")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _draw_arrow_labels(axis: plt.Axes, coordinates: pd.DataFrame) -> None:
    for row in coordinates.itertuples(index=False):
        axis.arrow(
            0,
            0,
            float(row.PC1),
            float(row.PC2),
            color="black",
            alpha=0.75,
            width=0.002,
            head_width=0.03,
            length_includes_head=True,
        )
        axis.text(
            float(row.PC1) * 1.06,
            float(row.PC2) * 1.06,
            str(row.Variable),
            fontsize=8,
            ha="center",
            va="center",
        )


def plot_loading(result: PcaResult, output_path: Path) -> None:
    coordinates = result.loadings.loc[:, ["Variable", "PC1", "PC2"]]
    figure, axis = plt.subplots(figsize=(8, 8))
    _draw_arrow_labels(axis, coordinates)
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.8)
    max_extent = max(
        1.0,
        float(np.abs(coordinates[["PC1", "PC2"]].to_numpy(dtype=float)).max())
        * 1.25,
    )
    axis.set_xlim(-max_extent, max_extent)
    axis.set_ylim(-max_extent, max_extent)
    axis.set_xlabel("PC1 Loading")
    axis.set_ylabel("PC2 Loading")
    axis.set_title("PCA Loading Plot")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_correlation_circle(result: PcaResult, output_path: Path) -> None:
    coordinates = result.correlation_coordinates
    figure, axis = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--")
    axis.add_patch(circle)
    _draw_arrow_labels(axis, coordinates)
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.axvline(0, color="gray", linewidth=0.8)
    axis.set_xlim(-1.1, 1.1)
    axis.set_ylim(-1.1, 1.1)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("PCA Correlation Circle")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def generate_report(
    dataset: PreparedDataset, result: PcaResult, output_path: Path
) -> None:
    top_count = min(5, len(result.summary))
    first_two_cumulative = float(
        result.summary.loc[1, "Cumulative Explained Variance"]
    )
    pc1_top = result.loadings.loc[
        result.loadings["PC1"].abs().idxmax(), "Variable"
    ]
    pc2_top = result.loadings.loc[
        result.loadings["PC2"].abs().idxmax(), "Variable"
    ]
    interpretation = (
        "The first two principal components capture most of the data variance."
        if first_two_cumulative > 0.70
        else "The first two principal components only capture part of the data variance."
    )

    lines = [
        f"PCA Report - {dataset.scenario}",
        "=" * (13 + len(dataset.scenario)),
        f"Total observations: {len(dataset.features)}",
        f"Variables: {len(dataset.features.columns)}",
        "",
        "First 5 component explained variance ratios:",
    ]
    for _, row in result.summary.head(top_count).iterrows():
        lines.append(
            f"  - {row['Component']}: "
            f"{float(row['Explained Variance Ratio']):.4f}"
        )

    lines.extend(
        [
            "",
            f"First two PCs cumulative variance: {first_two_cumulative:.4f}",
            f"Largest absolute loading on PC1: {pc1_top}",
            f"Largest absolute loading on PC2: {pc2_top}",
            "",
            "Fill-zero counts by PCA variable:",
        ]
    )
    for feature, count in dataset.fill_counts.items():
        lines.append(f"  - {feature}: {int(count)}")
    lines.extend(["", interpretation, ""])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(input_path: Path, scenario: str, output_root: Path) -> Path:
    scenario = normalize_scenario(scenario)
    print("Loading dataset...")
    dataset = load_data(input_path, scenario)
    print("Standardizing features...")
    values = standardize(dataset.features)
    print("Running PCA...")
    result = run_pca(values, dataset.features.columns)

    output_dir = Path(output_root) / scenario.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tables(dataset, result, output_dir)
    plot_scree(result.summary, output_dir / "explained_variance.png")
    plot_cumulative_variance(result.summary, output_dir / "cumulative_variance.png")
    plot_scatter(result, output_dir / "pca_scatter.png")
    plot_loading(result, output_dir / "loading_plot.png")
    plot_correlation_circle(result, output_dir / "correlation_circle.png")
    generate_report(dataset, result, output_dir / "pca_report.txt")

    print(f"Number of variables: {len(dataset.features.columns)}")
    print(f"Number of observations: {len(dataset.features)}")
    print(
        "PC1 explained variance: "
        f"{result.summary.loc[0, 'Explained Variance Ratio']:.4f}"
    )
    print(
        "PC2 explained variance: "
        f"{result.summary.loc[1, 'Explained Variance Ratio']:.4f}"
    )
    print(
        "First two PCs cumulative variance: "
        f"{result.summary.loc[1, 'Cumulative Explained Variance']:.4f}"
    )
    print(f"Output directory: {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exploratory PCA for one scenario after correlation-based "
            "variable screening."
        )
    )
    parser.add_argument(
        "-s",
        "--scenario",
        choices=sorted(SCENARIO_CONFIG),
        default=None,
        help="Scenario to analyze: PF or TF. Prompted when omitted.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input CSV. Defaults to the configured PF/TF for_PCA CSV.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = normalize_scenario(args.scenario) if args.scenario else prompt_scenario()
    input_path = args.input if args.input is not None else DEFAULT_INPUT_PATHS[scenario]
    run_analysis(input_path, scenario, args.output_root)


if __name__ == "__main__":
    main()
