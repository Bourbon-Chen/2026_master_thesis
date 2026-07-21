#!/usr/bin/env python3
"""Evaluate K values for future K-means clustering."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

os.environ.setdefault("OMP_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATHS: Dict[str, Path] = {
    "PF": PROJECT_ROOT / "outputs_correlation" / "pf" / "pf_for_PCA.csv",
    "TF": PROJECT_ROOT / "outputs_correlation" / "tf" / "tf_for_PCA.csv",
}
OUTPUT_ROOT = PROJECT_ROOT / "outputs_choose_k"
K_MIN = 2
K_MAX = 10
SILHOUETTE_SAMPLE_SIZE = 10000

RANDOM_STATE = 42
N_INIT = 50
MAX_ITER = 500
ALGORITHM = "lloyd"

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


@dataclass
class PreparedDataset:
    scenario: str
    metadata: pd.DataFrame
    features: pd.DataFrame
    fill_counts: pd.Series


def normalize_scenario(value: str) -> str:
    scenario = value.strip().upper()
    if scenario not in SCENARIO_CONFIG:
        raise ValueError("Scenario must be PF or TF.")
    return scenario


def prompt_scenario() -> str:
    while True:
        value = input("Select scenario (PF/TF): ")
        try:
            return normalize_scenario(value)
        except ValueError:
            print("Please enter PF or TF.")


def _scenario_features(scenario: str) -> Sequence[str]:
    return SCENARIO_CONFIG[scenario]["features"]


def _scenario_metadata(scenario: str) -> Sequence[str]:
    return SCENARIO_CONFIG[scenario]["metadata"]


def load_data(input_file: Path, scenario: str) -> PreparedDataset:
    """Load one scenario and keep only metadata plus configured features."""
    scenario = normalize_scenario(scenario)
    input_file = Path(input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")

    data = pd.read_csv(input_file, low_memory=False)
    feature_columns = list(_scenario_features(scenario))
    metadata_columns = list(_scenario_metadata(scenario))
    required_columns = metadata_columns + feature_columns
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns for "
            f"{scenario}: {', '.join(missing_columns)}"
        )

    for column in feature_columns:
        if not is_numeric_dtype(data[column]):
            raise ValueError(f"Clustering feature must be numeric: {column}")
        values = data[column].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Clustering feature contains infinity: {column}")

    features = data.loc[:, feature_columns].copy()
    fill_counts = features.isna().sum().astype(int)
    features = features.fillna(0)
    if features.isna().any().any():
        raise ValueError("Missing feature values remain after fill-zero imputation.")
    if len(features) < 3:
        raise ValueError("K evaluation requires at least three observations.")

    return PreparedDataset(
        scenario=scenario,
        metadata=data.loc[:, metadata_columns].copy(),
        features=features,
        fill_counts=fill_counts,
    )


def standardize_features(features: pd.DataFrame) -> np.ndarray:
    """Standardize clustering variables with StandardScaler."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required for K evaluation. "
            "Install/use the project environment from 2026master.yml."
        ) from error

    scaler = StandardScaler()
    return scaler.fit_transform(features.to_numpy(dtype=float))


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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return int(recommended_k)


def run_analysis(
    input_file: Path,
    scenario: str,
    output_root: Path,
    k_values: Iterable[int],
    silhouette_sample_size: Optional[int] = SILHOUETTE_SAMPLE_SIZE,
) -> Path:
    scenario = normalize_scenario(scenario)
    print("Loading dataset...")
    dataset = load_data(input_file, scenario)
    print(f"Number of observations: {len(dataset.features)}")
    print(f"Number of variables: {len(dataset.features.columns)}")
    print("Standardizing features...")
    values = standardize_features(dataset.features)
    print("Running K evaluation...")
    results = evaluate_k(values, k_values, silhouette_sample_size)

    output_dir = Path(output_root) / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, output_dir)
    plot_elbow(results, output_dir / "elbow_plot.png")
    plot_silhouette(results, output_dir / "silhouette_score.png")
    plot_calinski(results, output_dir / "calinski_harabasz.png")
    plot_davies(results, output_dir / "davies_bouldin.png")
    recommended_k = recommend_best_k(
        results, output_dir / "recommended_k.txt", silhouette_sample_size
    )

    print("Analysis completed.")
    print(f"Recommended K: {recommended_k}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate K values for future K-means clustering."
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=normalize_scenario,
        choices=sorted(SCENARIO_CONFIG),
        default=None,
        help="Scenario to analyze: PF or TF. Prompted when omitted.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Input CSV. Defaults to the configured PF/TF feature-selection CSV.",
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
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=SILHOUETTE_SAMPLE_SIZE,
        help=(
            "Sample size for silhouette_score. Use 0 for full data "
            f"(default: {SILHOUETTE_SAMPLE_SIZE})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = args.scenario if args.scenario is not None else prompt_scenario()
    if args.k_max < args.k_min:
        raise SystemExit("ERROR: --k-max must be greater than or equal to --k-min.")
    input_file = args.input if args.input is not None else DEFAULT_INPUT_PATHS[scenario]
    sample_size = (
        None if args.silhouette_sample_size <= 0 else args.silhouette_sample_size
    )
    run_analysis(
        input_file,
        scenario,
        args.output_root,
        range(args.k_min, args.k_max + 1),
        sample_size,
    )


if __name__ == "__main__":
    main()
