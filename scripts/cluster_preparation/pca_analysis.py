#!/usr/bin/env python3
"""Run exploratory PCA on a verified prepared clustering artifact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.clustering_preprocessing import (
    PreparedArtifact,
    load_prepared_artifact,
    sha256_file,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output_pca"
PCA_MANAGED_FILENAMES = (
    "pca_summary.csv",
    "pca_loadings.csv",
    "pca_scores.csv",
    "explained_variance.png",
    "cumulative_variance.png",
    "pca_scatter.png",
    "loading_plot.png",
    "correlation_circle.png",
    "pca_report.txt",
    "preprocessing_reference.json",
)


@dataclass
class PcaResult:
    summary: pd.DataFrame
    loadings: pd.DataFrame
    full_scores: pd.DataFrame
    two_d_scores: pd.DataFrame
    correlation_coordinates: pd.DataFrame


def load_pca_input(path: Path) -> PreparedArtifact:
    """Load a prepared artifact from its directory or manifest path."""
    prepared_path = Path(path)
    if prepared_path.is_file():
        if prepared_path.name != "preprocessing_config.json":
            raise ValueError(
                "Prepared input file must be preprocessing_config.json"
            )
        prepared_path = prepared_path.parent
    return load_prepared_artifact(prepared_path)


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


def save_tables(
    artifact: PreparedArtifact, result: PcaResult, output_dir: Path
) -> None:
    """Save PCA summary, loadings, and metadata-preserving scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(output_dir / "pca_summary.csv", index=False)
    result.loadings.to_csv(output_dir / "pca_loadings.csv", index=False)
    scores = pd.concat(
        [
            artifact.metadata.reset_index(drop=True),
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
    artifact: PreparedArtifact, result: PcaResult, output_path: Path
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
        f"PCA Report - {artifact.config['scenario']}",
        "=" * (13 + len(str(artifact.config["scenario"]))),
        f"Total observations: {len(artifact.standardized_features)}",
        f"Variables: {len(artifact.feature_names)}",
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
    for feature, count in artifact.original_features.isna().sum().items():
        lines.append(f"  - {feature}: {int(count)}")
    lines.extend(
        [
            "",
            "PCA is diagnostic and summarizes the prepared clustering geometry.",
            "K-means does not consume PCA scores; it consumes the same prepared "
            "standardized matrix directly.",
            "",
            interpretation,
            "",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_preprocessing_reference(
    artifact: PreparedArtifact, output_path: Path
) -> None:
    """Copy the verified preprocessing identity into a PCA output."""
    manifest_path = (
        artifact.directory / "preprocessing_config.json"
    ).resolve()
    reference = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "matrix_sha256": artifact.config["matrix_sha256"],
        "scenario": artifact.config["scenario"],
        "row_count": len(artifact.standardized_features),
        "feature_names": list(artifact.feature_names),
    }
    Path(output_path).write_text(
        json.dumps(reference, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_pca_outputs(
    artifact: PreparedArtifact,
    result: PcaResult,
    output_dir: Path,
) -> None:
    """Generate the complete managed PCA output set in one directory."""
    save_tables(artifact, result, output_dir)
    plot_scree(result.summary, output_dir / "explained_variance.png")
    plot_cumulative_variance(
        result.summary,
        output_dir / "cumulative_variance.png",
    )
    plot_scatter(result, output_dir / "pca_scatter.png")
    plot_loading(result, output_dir / "loading_plot.png")
    plot_correlation_circle(
        result,
        output_dir / "correlation_circle.png",
    )
    generate_report(artifact, result, output_dir / "pca_report.txt")
    write_preprocessing_reference(
        artifact,
        output_dir / "preprocessing_reference.json",
    )


def _publish_pca_outputs(staging_dir: Path, output_dir: Path) -> None:
    """Replace only the complete PCA-managed set in the final directory."""
    missing = [
        filename
        for filename in PCA_MANAGED_FILENAMES
        if not (staging_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            "PCA staging did not produce every managed output: "
            + ", ".join(missing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in PCA_MANAGED_FILENAMES:
        (staging_dir / filename).replace(output_dir / filename)


def run_analysis(prepared_path: Path, output_root: Path) -> Path:
    print("Loading prepared artifact...")
    artifact = load_pca_input(prepared_path)
    scenario = str(artifact.config["scenario"])
    output_root = Path(output_root)
    output_dir = output_root / scenario.lower()
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            f"PCA output root is not a directory: {output_root}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"PCA final output path is not a directory: {output_dir}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    print("Running PCA...")
    result = run_pca(values, artifact.feature_names)

    with TemporaryDirectory(
        prefix=f".pca-{scenario.lower()}-",
        dir=output_root,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        _write_pca_outputs(artifact, result, staging_dir)
        _publish_pca_outputs(staging_dir, output_dir)

    print(f"Number of variables: {len(artifact.feature_names)}")
    print(f"Number of observations: {len(artifact.standardized_features)}")
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
            "Run diagnostic PCA on a verified prepared clustering artifact. "
            "K-means does not consume the PCA scores."
        )
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=None,
        help=(
            "Prepared scenario directory or preprocessing_config.json. "
            "Prompted when omitted."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_path = (
        args.prepared
        if args.prepared is not None
        else Path(
            input(
                "Prepared scenario directory or preprocessing_config.json: "
            ).strip()
        )
    )
    run_analysis(prepared_path, args.output_root)


if __name__ == "__main__":
    main()
