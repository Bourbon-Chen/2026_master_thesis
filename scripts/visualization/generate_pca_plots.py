#!/usr/bin/env python3
"""Generate PCA plots from saved labels without rerunning HDBSCAN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.config import INPUT_CSV, PIPELINE_ROOT, SCENARIOS  # noqa: E402
from utils.feature_utils import prepare_features  # noqa: E402
from utils.io_utils import (
    ensure_pipeline_dirs,
    prompt_input_path,
    resolve_dataset_output_root,
    resolve_input_path,
)  # noqa: E402
from utils.result_utils import discover_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PCA cluster plots from modular HDBSCAN outputs."
    )
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--version", choices=("version_a", "version_b"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    return parser.parse_args()


def plot_clusters(coordinates: np.ndarray, labels: pd.Series, path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = labels == -1
    clustered = ~noise
    plt.figure(figsize=(10, 8))
    if clustered.any():
        plt.scatter(
            coordinates[clustered, 0],
            coordinates[clustered, 1],
            c=labels[clustered],
            cmap="tab20",
            s=3,
            alpha=0.65,
            rasterized=True,
        )
    if noise.any():
        plt.scatter(
            coordinates[noise, 0],
            coordinates[noise, 1],
            c="lightgray",
            s=2,
            alpha=0.45,
            label="Noise (-1)",
            rasterized=True,
        )
        plt.legend()
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(f"ERROR: scikit-learn is required: {exc}")

    input_path = (
        resolve_input_path(args.input)
        if args.input is not None
        else prompt_input_path(INPUT_CSV)
    )
    output_root = resolve_dataset_output_root(input_path, args.pipeline_root.resolve())
    ensure_pipeline_dirs(output_root)
    metadata_files = discover_metadata(output_root, args.scenario, args.version)
    if not metadata_files:
        raise SystemExit("ERROR: no modular cluster metadata files were found.")
    data = pd.read_csv(input_path)
    coordinates_by_version = {}

    for metadata_path, metadata in metadata_files:
        key = (metadata["scenario"], metadata["version"])
        if key not in coordinates_by_version:
            prepared = prepare_features(data, metadata["scenario"])
            values = prepared.values.to_numpy(dtype=float)
            if metadata["version"] == "version_b":
                values = StandardScaler().fit_transform(values)
            coordinates = PCA(n_components=2).fit_transform(values)
            coordinates_by_version[key] = coordinates
            coordinate_output = (
                output_root
                / "visualizations"
                / "pca"
                / metadata["scenario"]
                / metadata["version"]
                / "pca_coordinates.csv"
            )
            coordinate_output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "street_id": data["MS_ID"] if "MS_ID" in data else data.index,
                    "pca_component_1": coordinates[:, 0],
                    "pca_component_2": coordinates[:, 1],
                }
            ).to_csv(coordinate_output, index=False)

        labels = pd.read_csv(metadata_path.parent / "cluster_labels.csv")
        if len(labels) != len(data):
            raise ValueError(f"Label row count mismatch: {metadata_path.parent}")
        plot_output = (
            output_root
            / "visualizations"
            / "pca"
            / metadata["scenario"]
            / metadata["version"]
            / metadata_path.parent.name
            / "pca_hdbscan_clusters.png"
        )
        plot_clusters(
            coordinates_by_version[key],
            labels["cluster_id"],
            plot_output,
            (
                f"{metadata['scenario']} {metadata['version']} "
                f"mcs={metadata['min_cluster_size']} ms={metadata['min_samples']}"
            ),
        )
        print(f"Saved: {plot_output}")


if __name__ == "__main__":
    main()
