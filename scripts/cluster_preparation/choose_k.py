#!/usr/bin/env python3
"""Evaluate K values for future K-means clustering."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")


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

OUTPUT_ROOT = PROJECT_ROOT / "outputs_choose_k"
K_MIN = 2
K_MAX = 10
SILHOUETTE_SAMPLE_SIZE = 10000

RANDOM_STATE = 42
N_INIT = 50
MAX_ITER = 500
ALGORITHM = "lloyd"

K_EVALUATION_COLUMNS = [
    "K",
    "Inertia",
    "Silhouette",
    "Calinski_Harabasz",
    "Davies_Bouldin",
    "Largest_Cluster",
    "Smallest_Cluster",
    "Cluster_Size_SD",
]


def _validate_k_range(values: np.ndarray, k_values: Iterable[int]) -> list[int]:
    validated = [int(k) for k in k_values]
    if not validated:
        raise ValueError("At least one K value is required.")
    if min(validated) < 2:
        raise ValueError("K values must be at least 2.")
    max_allowed = len(values) - 1
    invalid = [k for k in validated if k > max_allowed]
    if invalid:
        raise ValueError(
            "K must be smaller than the number of observations for silhouette "
            f"score. Invalid K values: {', '.join(map(str, invalid))}"
        )
    return validated


def _silhouette_sample_size(
    values: np.ndarray, sample_size: Optional[int]
) -> Optional[int]:
    if sample_size is None or sample_size <= 0 or sample_size >= len(values):
        return None
    return int(sample_size)


def evaluate_k(
    values: np.ndarray,
    k_values: Iterable[int],
    silhouette_sample_size: Optional[int] = SILHOUETTE_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Calculate K-means evaluation metrics for every configured K."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for K evaluation. "
            "Install/use the project environment from 2026master.yml."
        ) from error

    validated_k_values = _validate_k_range(values, k_values)
    records = []
    effective_sample_size = _silhouette_sample_size(values, silhouette_sample_size)
    for k in validated_k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            algorithm=ALGORITHM,
        )
        labels = model.fit_predict(values)
        cluster_sizes = np.array(
            [count for _, count in sorted(Counter(labels).items())],
            dtype=float,
        )
        records.append(
            {
                "K": k,
                "Inertia": float(model.inertia_),
                "Silhouette": float(
                    silhouette_score(
                        values,
                        labels,
                        sample_size=effective_sample_size,
                        random_state=RANDOM_STATE,
                    )
                ),
                "Calinski_Harabasz": float(calinski_harabasz_score(values, labels)),
                "Davies_Bouldin": float(davies_bouldin_score(values, labels)),
                "Largest_Cluster": int(cluster_sizes.max()),
                "Smallest_Cluster": int(cluster_sizes.min()),
                "Cluster_Size_SD": float(cluster_sizes.std(ddof=0)),
            }
        )
        print(f"K = {k} completed.")

    return pd.DataFrame(records, columns=K_EVALUATION_COLUMNS)


def save_results(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "k_evaluation.csv", index=False)


def _plot_metric(
    results: pd.DataFrame,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(results["K"], results[y_column], marker="o", color="black")
    axis.set_xlabel("K")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.set_xticks(results["K"].tolist())
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_elbow(results: pd.DataFrame, output_path: Path) -> None:
    _plot_metric(results, "Inertia", "Inertia", "Elbow Method", output_path)


def plot_silhouette(results: pd.DataFrame, output_path: Path) -> None:
    _plot_metric(
        results,
        "Silhouette",
        "Silhouette Score",
        "Silhouette Score",
        output_path,
    )


def plot_calinski(results: pd.DataFrame, output_path: Path) -> None:
    _plot_metric(
        results,
        "Calinski_Harabasz",
        "CH Score",
        "Calinski-Harabasz Index",
        output_path,
    )


def plot_davies(results: pd.DataFrame, output_path: Path) -> None:
    _plot_metric(
        results,
        "Davies_Bouldin",
        "DB Score",
        "Davies-Bouldin Index (lower is better)",
        output_path,
    )


def _best_elbow_k(results: pd.DataFrame) -> int:
    k = results["K"].to_numpy(dtype=float)
    inertia = results["Inertia"].to_numpy(dtype=float)
    if len(results) <= 2:
        return int(results.loc[results["Inertia"].idxmin(), "K"])

    start = np.array([k[0], inertia[0]])
    end = np.array([k[-1], inertia[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return int(results.iloc[0]["K"])

    points = np.column_stack([k, inertia])
    offsets = start - points
    distances = np.abs(line[0] * offsets[:, 1] - line[1] * offsets[:, 0]) / line_norm
    return int(results.iloc[int(np.argmax(distances))]["K"])


def recommend_best_k(
    results: pd.DataFrame,
    output_path: Path,
    silhouette_sample_size: Optional[int] = SILHOUETTE_SAMPLE_SIZE,
) -> int:
    """Write metric-specific best K values and return the modal recommendation."""
    best_by_metric = {
        "Elbow": _best_elbow_k(results),
        "Silhouette": int(results.loc[results["Silhouette"].idxmax(), "K"]),
        "Calinski-Harabasz": int(
            results.loc[results["Calinski_Harabasz"].idxmax(), "K"]
        ),
        "Davies-Bouldin": int(results.loc[results["Davies_Bouldin"].idxmin(), "K"]),
    }
    votes = Counter(best_by_metric.values())
    recommended_k, vote_count = sorted(
        votes.items(), key=lambda item: (-item[1], item[0])
    )[0]

    if vote_count == 1:
        reason = (
            "No single K is supported by multiple metrics; the smallest "
            "metric-best K was selected as a conservative recommendation."
        )
    else:
        reason = (
            f"{vote_count} out of four evaluation metrics suggest K={recommended_k}."
        )

    lines = [
        "K-means K Recommendation",
        "========================",
        (
            "Silhouette sample size: full data"
            if silhouette_sample_size is None or silhouette_sample_size <= 0
            else f"Silhouette sample size: {silhouette_sample_size}"
        ),
        "",
        "Metric-specific best K:",
    ]
    for metric, k in best_by_metric.items():
        lines.append(f"  - {metric}: K={k}")
    lines.extend(["", f"Recommended K: {recommended_k}", reason, ""])
    lines.extend(
        [
            "Candidate cluster-size diagnostics:",
            (
                "  - Minimum candidate cluster size: "
                f"{int(results['Smallest_Cluster'].min())}"
            ),
            (
                "  - Maximum candidate cluster size: "
                f"{int(results['Largest_Cluster'].max())}"
            ),
            "",
            (
                "The final K remains a documented research choice, informed by "
                "these diagnostics and substantive interpretation."
            ),
            "",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return int(recommended_k)


def write_preprocessing_reference(
    artifact: PreparedArtifact, output_path: Path
) -> None:
    """Copy the verified preprocessing identity into a choose-K output."""
    manifest_path = (artifact.directory / "preprocessing_config.json").resolve()
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


def run_analysis(
    prepared_path: Path,
    output_root: Path,
    k_values: Iterable[int],
) -> Path:
    print("Loading prepared artifact...")
    artifact = load_prepared_artifact(prepared_path)
    scenario = str(artifact.config["scenario"])
    output_dir = Path(output_root) / scenario
    if output_dir.exists():
        raise FileExistsError(
            f"choose-K final output directory already exists: {output_dir}"
        )
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    valid_k = _validate_k_range(values, k_values)
    print(f"Number of observations: {len(artifact.standardized_features)}")
    print(f"Number of variables: {len(artifact.feature_names)}")
    print("Running K evaluation...")
    results = evaluate_k(values, valid_k)

    save_results(results, output_dir)
    plot_elbow(results, output_dir / "elbow_plot.png")
    plot_silhouette(results, output_dir / "silhouette_score.png")
    plot_calinski(results, output_dir / "calinski_harabasz.png")
    plot_davies(results, output_dir / "davies_bouldin.png")
    recommended_k = recommend_best_k(results, output_dir / "recommended_k.txt")
    write_preprocessing_reference(
        artifact, output_dir / "preprocessing_reference.json"
    )

    print("Analysis completed.")
    print(f"Recommended K: {recommended_k}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate K values on a verified prepared clustering artifact."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=None,
        help="Prepared scenario artifact directory. Prompted when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output root directory (default: {OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=K_MIN,
        help=f"Minimum K to evaluate (default: {K_MIN}).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=K_MAX,
        help=f"Maximum K to evaluate (default: {K_MAX}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_max < args.k_min:
        raise SystemExit("ERROR: --k-max must be greater than or equal to --k-min.")
    prepared_path = (
        args.prepared
        if args.prepared is not None
        else Path(input("Prepared scenario artifact directory: ").strip())
    )
    run_analysis(
        prepared_path,
        args.output_root,
        range(args.k_min, args.k_max + 1),
    )


if __name__ == "__main__":
    main()
