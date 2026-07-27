#!/usr/bin/env python3
"""Run one reproducible K-means experiment on prepared PF or TF data."""

from __future__ import annotations

import contextlib
import argparse
import inspect
import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.clustering_preprocessing import (
    PreparedArtifact,
    inverse_transform_frame,
    load_prepared_artifact,
    sha256_file,
)

OUTPUT_ROOT = PROJECT_ROOT / "output_kmeans"
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
KMEANS_RANDOM_STATE = 42
KMEANS_ALGORITHM = "lloyd"
KMEANS_TOL = 1e-4
SILHOUETTE_WORKING_MEMORY_MB = 256
PROVISIONAL_RUNTIME_SCOPE = (
    "start through all bulk CSV and PNG writes before the first metadata pass"
)
PERSISTED_RUNTIME_SCOPE = (
    "start through the complete first metadata pass inside staging; excludes "
    "the final metadata rewrite and atomic final-directory publication"
)
OUTPUT_DPI = 300
TABULAR_TEXT_OUTPUTS = (
    "clustered_data.csv",
    "cluster_summary.csv",
    "cluster_centroids_standardized.csv",
    "cluster_centroids_original_scale.csv",
    "cluster_feature_profile.csv",
    "sample_distances.csv",
    "model_metrics.csv",
    "cluster_label_mapping.csv",
    "preprocessing_audit.csv",
    "run_config.json",
    "run_report.txt",
)
PLOT_OUTPUTS = (
    "pca_cluster_scatter.png",
    "cluster_size_bar.png",
    "cluster_centroid_heatmap.png",
    "cluster_feature_profiles.png",
)
ALL_OUTPUTS = TABULAR_TEXT_OUTPUTS + PLOT_OUTPUTS

class KMeansAnalysisError(RuntimeError):
    """Base exception for expected user-facing workflow errors."""


class DataValidationError(KMeansAnalysisError):
    """Raised when prepared input cannot safely enter K-means."""


class CompatibilityError(KMeansAnalysisError):
    """Raised when installed scikit-learn cannot honor configured settings."""


class OutputDirectoryExistsError(KMeansAnalysisError):
    """Raised when a readable final experiment directory already exists."""


@dataclass(frozen=True)
class ValidatedData:
    artifact: PreparedArtifact
    values: np.ndarray
    feature_names: tuple[str, ...]
    metadata: pd.DataFrame
    row_count: int


@dataclass
class FittedKMeans:
    model: object
    original_labels: np.ndarray
    assigned_distances: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
    fit_seconds: float
    warnings: list[str]
    standardized_centroids: pd.DataFrame | None = None


@dataclass
class ClusterOrdering:
    reordered_labels: np.ndarray
    mapping: pd.DataFrame
    reordered_centroids: pd.DataFrame
    ordering_method: str
    pca_scores: np.ndarray | None
    explained_variance_ratio: np.ndarray | None
    warnings: list[str]


@dataclass
class AnalysisTables:
    clustered_data: pd.DataFrame
    cluster_summary: pd.DataFrame
    cluster_centroids_standardized: pd.DataFrame
    cluster_centroids_original_scale: pd.DataFrame
    cluster_feature_profile: pd.DataFrame
    sample_distances: pd.DataFrame
    label_mapping: pd.DataFrame
    data_quality_issues: pd.DataFrame
    preprocessing_audit_path: Path


@dataclass
class MetricResults:
    values: dict[str, float | None]
    warnings: list[str]


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _no_follow_identity(path: Path) -> tuple[int, int]:
    path_stat = os.lstat(path)
    return int(path_stat.st_dev), int(path_stat.st_ino)


def _is_reparse_point(path_stat: os.stat_result) -> bool:
    attributes = int(getattr(path_stat, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(attributes & reparse_flag)


def build_output_directory(
    output_root: Path, scenario: str, k: int, n_features: int
) -> Path:
    name = (
        f"{scenario.lower()}_k{k}_v{n_features}_"
        f"kmeanspp_n{KMEANS_N_INIT}_mi{KMEANS_MAX_ITER}_"
        f"rs{KMEANS_RANDOM_STATE}"
    )
    target = Path(output_root) / name
    if _path_lexists(target):
        raise OutputDirectoryExistsError(
            f"Output folder already exists: {target}. "
            "Please move or rename the existing result before running again."
        )
    return target


@contextlib.contextmanager
def staged_output_directory(final_path: Path):
    final_path = Path(final_path)
    output_root = final_path.parent
    staging_path: Path | None = None
    resolved_output_root: Path | None = None
    resolved_staging_path: Path | None = None
    output_root_identity: tuple[int, int] | None = None
    staging_identity: tuple[int, int] | None = None
    try:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise KMeansAnalysisError(
                f"Could not create output folder: {output_root}: {error}"
            ) from error
        try:
            resolved_output_root = output_root.resolve(strict=True)
            output_root_identity = _no_follow_identity(resolved_output_root)
        except OSError as error:
            raise KMeansAnalysisError(
                f"Could not identify output folder {output_root}: {error}"
            ) from error
        if _path_lexists(final_path):
            raise OutputDirectoryExistsError(
                f"Output folder already exists: {final_path}. "
                "Please move or rename the existing result before running again."
            )
        try:
            staging_path = Path(
                tempfile.mkdtemp(
                    prefix=".kmeans_staging_", dir=resolved_output_root
                )
            )
            resolved_staging_path = staging_path.resolve(strict=True)
            staging_identity = _no_follow_identity(resolved_staging_path)
        except OSError as error:
            raise KMeansAnalysisError(
                f"Could not create staging folder under "
                f"{resolved_output_root}: {error}"
            ) from error
        if (
            resolved_staging_path.parent != resolved_output_root
            or _no_follow_identity(resolved_output_root) != output_root_identity
        ):
            raise KMeansAnalysisError(
                f"Refusing staging folder outside the captured output root: "
                f"{resolved_staging_path}"
            )

        yield resolved_staging_path

        if _path_lexists(final_path):
            raise OutputDirectoryExistsError(
                f"Output folder already exists: {final_path}. "
                "Please move or rename the existing result before running again."
            )
        try:
            resolved_staging_path.rename(final_path)
        except OSError as error:
            raise KMeansAnalysisError(
                f"Could not rename staging folder {resolved_staging_path} "
                f"to final output {final_path}: {error}"
            ) from error
    finally:
        if (
            resolved_output_root is not None
            and resolved_staging_path is not None
            and output_root_identity is not None
            and staging_identity is not None
            and _path_lexists(resolved_staging_path)
        ):
            try:
                current_root_identity = _no_follow_identity(
                    resolved_output_root
                )
                current_staging_stat = os.lstat(resolved_staging_path)
                current_staging_identity = (
                    int(current_staging_stat.st_dev),
                    int(current_staging_stat.st_ino),
                )
            except OSError as error:
                raise KMeansAnalysisError(
                    f"Could not verify staging cleanup path "
                    f"{resolved_staging_path}: {error}"
                ) from error
            if resolved_staging_path.parent != resolved_output_root:
                raise KMeansAnalysisError(
                    f"Refusing unsafe staging cleanup outside "
                    f"{resolved_output_root}: {resolved_staging_path}"
                )
            if current_root_identity != output_root_identity:
                raise KMeansAnalysisError(
                    f"Output root identity changed; refusing staging cleanup: "
                    f"{resolved_output_root}"
                )
            if (
                current_staging_identity != staging_identity
                or not stat.S_ISDIR(current_staging_stat.st_mode)
                or _is_reparse_point(current_staging_stat)
            ):
                raise KMeansAnalysisError(
                    f"The staging identity changed; refusing recursive cleanup: "
                    f"{resolved_staging_path}"
                )
            try:
                shutil.rmtree(resolved_staging_path)
            except OSError as error:
                raise KMeansAnalysisError(
                    f"Could not remove staging folder "
                    f"{resolved_staging_path}: {error}"
                ) from error


def prompt_k(n_samples: int | None = None) -> int:
    while True:
        value = input("Enter the number of clusters K: ").strip()
        if re.fullmatch(r"[+-]?[0-9]+", value) is None:
            print("K must be an integer greater than or equal to 2.")
            continue
        k = int(value)
        if k < 2:
            print("K must be an integer greater than or equal to 2.")
            continue
        if n_samples is not None and k >= n_samples:
            print(f"K must be smaller than the valid sample count ({n_samples}).")
            continue
        return k


def _strip_matching_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1].strip()
    return stripped


def prompt_prepared_path() -> Path:
    """Prompt for the immutable prepared-artifact directory."""
    while True:
        candidate = Path(_strip_matching_quotes(
            input("Prepared artifact directory: ")
        ))
        if candidate.is_dir():
            return candidate.resolve()
        print(f"Prepared artifact directory not found: {candidate}")


def run_kmeans(values: np.ndarray, k: int) -> FittedKMeans:
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.cluster import KMeans
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as error:
        raise CompatibilityError(
            "scikit-learn is required. Activate the environment from 2026master.yml."
        ) from error

    configured = {
        "n_clusters": k,
        "init": KMEANS_INIT,
        "n_init": KMEANS_N_INIT,
        "max_iter": KMEANS_MAX_ITER,
        "random_state": KMEANS_RANDOM_STATE,
        "algorithm": KMEANS_ALGORITHM,
        "tol": KMEANS_TOL,
    }
    supported = set(inspect.signature(KMeans).parameters)
    unsupported = sorted(set(configured) - supported)
    if unsupported:
        raise CompatibilityError(
            f"scikit-learn {sklearn_version} does not support configured "
            f"KMeans arguments: {', '.join(unsupported)}"
        )

    try:
        model = KMeans(**configured)
    except (TypeError, ValueError) as error:
        settings = ", ".join(
            f"{name}={value!r}" for name, value in configured.items()
        )
        raise CompatibilityError(
            f"scikit-learn {sklearn_version} rejected the configured KMeans "
            f"settings ({settings}): {error}"
        ) from error
    captured_messages: list[str] = []
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            labels = model.fit_predict(values)
    except ValueError as error:
        message = str(error).lower()
        configurable = {
            name: value
            for name, value in configured.items()
            if name != "n_clusters"
        }
        offending = [
            f"{name}={value!r}"
            for name, value in configurable.items()
            if name.lower() in message and str(value).lower() in message
        ]
        if offending:
            raise CompatibilityError(
                f"scikit-learn {sklearn_version} rejected configured KMeans "
                f"parameter value(s) {', '.join(offending)}: {error}"
            ) from error
        raise DataValidationError(
            f"K-means rejected the validated input data: {error}"
        ) from error
    fit_seconds = time.perf_counter() - started
    captured_messages.extend(str(item.message) for item in caught)
    all_distances = model.transform(values)
    assigned = all_distances[np.arange(len(values)), labels]
    converged = int(model.n_iter_) < KMEANS_MAX_ITER
    if not converged:
        captured_messages.append(
            f"K-means reached max_iter={KMEANS_MAX_ITER}; convergence is not confirmed."
        )
    return FittedKMeans(
        model=model,
        original_labels=np.asarray(labels, dtype=int),
        assigned_distances=np.asarray(assigned, dtype=float),
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
        converged=converged,
        fit_seconds=float(fit_seconds),
        warnings=captured_messages,
    )


def reorder_cluster_labels(
    fitted: FittedKMeans,
    values: np.ndarray,
    feature_names: Sequence[str],
) -> ClusterOrdering:
    warnings_list: list[str] = []
    pca_scores: np.ndarray | None = None
    explained: np.ndarray | None = None
    centroids = np.asarray(fitted.model.cluster_centers_, dtype=float)

    if min(values.shape[0], values.shape[1]) >= 2:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            pca_scores = pca.fit_transform(values)
            ordering_values = pca.transform(centroids)[:, 0]
            explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
            method = "centroid PC1 projection"
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings_list.append(f"PCA unavailable; used centroid mean: {error}")
            ordering_values = centroids.mean(axis=1)
            method = "centroid mean"
    else:
        warnings_list.append(
            "PCA requires at least two samples and two features; used centroid mean."
        )
        ordering_values = centroids.mean(axis=1)
        method = "centroid mean"

    original_order = sorted(
        range(len(centroids)),
        key=lambda label: (float(ordering_values[label]), int(label)),
    )
    remap = {
        original_label: reordered_id
        for reordered_id, original_label in enumerate(original_order, start=1)
    }
    reordered_labels = np.array(
        [remap[int(label)] for label in fitted.original_labels], dtype=int
    )
    mapping = pd.DataFrame(
        {
            "sklearn_original_label": original_order,
            "reordered_cluster_id": range(1, len(original_order) + 1),
            "ordering_value": [
                float(ordering_values[label]) for label in original_order
            ],
        }
    )
    reordered_centroids = pd.DataFrame(
        centroids[original_order],
        columns=list(feature_names),
        index=pd.Index(
            np.arange(1, len(original_order) + 1), name="cluster"
        ),
    )
    fitted.standardized_centroids = reordered_centroids.copy()
    return ClusterOrdering(
        reordered_labels=reordered_labels,
        mapping=mapping,
        reordered_centroids=reordered_centroids,
        ordering_method=method,
        pca_scores=pca_scores,
        explained_variance_ratio=explained,
        warnings=warnings_list,
    )


def calculate_cluster_profiles(
    validated: ValidatedData,
    fitted: FittedKMeans,
    ordering: ClusterOrdering,
) -> AnalysisTables:
    if (
        len(ordering.reordered_labels) != validated.row_count
        or len(fitted.assigned_distances) != validated.row_count
    ):
        raise DataValidationError(
            "K-means output length does not match the prepared metadata."
        )
    assignments = pd.DataFrame(
        {
            "source_row_position": validated.metadata[
                "source_row_position"
            ].to_numpy(copy=True),
            "kmeans_cluster": ordering.reordered_labels,
            "distance_to_assigned_centroid": fitted.assigned_distances,
        }
    )
    if assignments["source_row_position"].duplicated().any():
        raise DataValidationError("K-means assignments are not one-to-one by row.")
    reporting_rows = pd.concat(
        [
            validated.metadata.reset_index(drop=True),
            validated.artifact.original_features.reset_index(drop=True),
        ],
        axis=1,
    )
    clustered = reporting_rows.merge(
        assignments,
        on="source_row_position",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(clustered) != validated.row_count or clustered["kmeans_cluster"].isna().any():
        raise DataValidationError("Every prepared row must receive a cluster label.")

    standardized_centroids = ordering.reordered_centroids.copy()
    original_centroids = inverse_transform_frame(
        standardized_centroids, validated.artifact.scaler_parameters
    )
    summaries = []
    profiles = []
    for cluster_id in range(1, len(ordering.mapping) + 1):
        member_mask = ordering.reordered_labels == cluster_id
        distances = fitted.assigned_distances[member_mask]
        count = int(member_mask.sum())
        summaries.append(
            {
                "cluster_id": cluster_id,
                "sample_count": count,
                "sample_percentage": count / len(ordering.reordered_labels) * 100.0,
                "distance_to_centroid_mean": float(np.mean(distances))
                if count
                else np.nan,
                "distance_to_centroid_median": float(np.median(distances))
                if count
                else np.nan,
                "distance_to_centroid_std": float(np.std(distances, ddof=0))
                if count
                else np.nan,
                "distance_to_centroid_max": float(np.max(distances))
                if count
                else np.nan,
            }
        )
        for feature_position, feature in enumerate(validated.feature_names):
            original_values = validated.artifact.original_features.loc[
                member_mask, feature
            ]
            filled_values = validated.artifact.filled_features.loc[
                member_mask, feature
            ]
            original_non_null = original_values.dropna()
            profiles.append(
                {
                    "cluster": cluster_id,
                    "feature": feature,
                    "original_non_null_count": int(original_non_null.count()),
                    "original_structural_null_count": int(original_values.isna().sum()),
                    "original_structural_null_percent": (
                        float(original_values.isna().mean() * 100.0)
                        if count
                        else np.nan
                    ),
                    "original_non_null_mean": (
                        float(original_non_null.mean())
                        if not original_non_null.empty
                        else np.nan
                    ),
                    "original_non_null_median": (
                        float(original_non_null.median())
                        if not original_non_null.empty
                        else np.nan
                    ),
                    "filled_mean": float(filled_values.mean()) if count else np.nan,
                    "standardized_centroid": float(
                        standardized_centroids.loc[cluster_id, feature]
                    ),
                    "inverse_transformed_centroid": float(
                        original_centroids.loc[cluster_id, feature]
                    ),
                }
            )

    distance_table = clustered.loc[
        :,
        [
            *validated.metadata.columns,
            "kmeans_cluster",
            "distance_to_assigned_centroid",
        ],
    ].copy()
    issues = pd.DataFrame(
        columns=["issue_type", "column", "count", "action", "details"]
    )
    return AnalysisTables(
        clustered_data=clustered,
        cluster_summary=pd.DataFrame(summaries),
        cluster_centroids_standardized=standardized_centroids,
        cluster_centroids_original_scale=original_centroids,
        cluster_feature_profile=pd.DataFrame(profiles),
        sample_distances=distance_table,
        label_mapping=ordering.mapping.copy(),
        data_quality_issues=issues,
        preprocessing_audit_path=(
            validated.artifact.directory / "preprocessing_audit.csv"
        ),
    )


def calculate_metrics(values: np.ndarray, labels: np.ndarray) -> MetricResults:
    try:
        import sklearn
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
    except ImportError as error:
        raise CompatibilityError(
            "scikit-learn metrics are unavailable in the active environment."
        ) from error

    def full_silhouette_score() -> float:
        with sklearn.config_context(
            working_memory=SILHOUETTE_WORKING_MEMORY_MB
        ):
            return float(silhouette_score(values, labels))

    calculators = {
        "silhouette_score": full_silhouette_score,
        "calinski_harabasz_score": lambda: calinski_harabasz_score(values, labels),
        "davies_bouldin_score": lambda: davies_bouldin_score(values, labels),
    }
    results: dict[str, float | None] = {}
    warning_messages: list[str] = []
    for name, calculator in calculators.items():
        try:
            results[name] = float(calculator())
        except ValueError as error:
            results[name] = None
            warning_messages.append(f"{name} unavailable: {error}")
    return MetricResults(results, warning_messages)


def build_run_config(
    scenario: str,
    input_path: Path,
    output_path: Path,
    k: int,
    validated: ValidatedData,
    fitted: FittedKMeans,
    ordering: ClusterOrdering,
    warnings_list: Sequence[str],
    started_at: str,
    ended_at: str,
    runtime_seconds: float,
    runtime_scope: str = PERSISTED_RUNTIME_SCOPE,
) -> dict[str, object]:
    from sklearn import __version__ as sklearn_version

    explained_variance: dict[str, float] | None = None
    if ordering.explained_variance_ratio is not None:
        percentages = (
            np.asarray(ordering.explained_variance_ratio, dtype=float) * 100.0
        )
        explained_variance = {
            "pc1": float(percentages[0]),
            "pc2": float(percentages[1]),
            "pc1_plus_pc2": float(percentages[0] + percentages[1]),
        }

    manifest_path = (
        validated.artifact.directory / "preprocessing_config.json"
    ).resolve()
    null_counts = validated.artifact.original_features.isna().sum()
    missing_counts = {
        str(name): int(count) for name, count in null_counts.items() if count
    }
    structural_null_row_count = int(
        validated.metadata["had_structural_null"].sum()
    )
    structural_null_value_count = int(
        validated.metadata["structural_null_count"].sum()
    )
    return {
        "scenario": str(scenario),
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        "k": int(k),
        "feature_list": [str(name) for name in validated.feature_names],
        "preprocessing_reference": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "matrix_sha256": validated.artifact.config["matrix_sha256"],
            "source_path": validated.artifact.config["source_path"],
            "source_sha256": validated.artifact.config["source_sha256"],
            "scenario": validated.artifact.config["scenario"],
            "row_count": validated.row_count,
            "feature_names": list(validated.feature_names),
            "feature_count": len(validated.feature_names),
            "structural_null_row_count": structural_null_row_count,
            "structural_null_value_count": structural_null_value_count,
            "rows_excluded_for_null": 0,
        },
        "excluded_columns": validated.artifact.excluded_features.loc[
            :, ["column", "reason"]
        ].to_dict(orient="records"),
        "id_columns": [str(name) for name in validated.metadata.columns],
        "data_quality_issues": [],
        "standard_scaler_applied": False,
        "scaling_transformations_applied": [],
        "kmeans_parameters": {
            "init": KMEANS_INIT,
            "n_init": int(KMEANS_N_INIT),
            "max_iter": int(KMEANS_MAX_ITER),
            "random_state": int(KMEANS_RANDOM_STATE),
            "algorithm": KMEANS_ALGORITHM,
            "tol": float(KMEANS_TOL),
        },
        "metric_parameters": {
            "silhouette_sample_size": None,
            "silhouette_working_memory_mb": int(
                SILHOUETTE_WORKING_MEMORY_MB
            ),
        },
        "sample_counts": {
            "original": validated.row_count,
            "valid": validated.row_count,
            "removed": 0,
        },
        "missing_counts": missing_counts,
        "missing_value_policy": "Prepared structural nulls retained.",
        "cluster_label_ordering": str(ordering.ordering_method),
        "pca_explained_variance_percent": explained_variance,
        "model_fit": {
            "inertia": float(fitted.inertia),
            "n_iter": int(fitted.n_iter),
            "converged": bool(fitted.converged),
            "fit_seconds": float(fitted.fit_seconds),
        },
        "runtime": {
            "runtime_seconds": float(runtime_seconds),
            "runtime_scope": str(runtime_scope),
        },
        "versions": {
            "python": str(platform.python_version()),
            "pandas": str(pd.__version__),
            "numpy": str(np.__version__),
            "scikit_learn": str(sklearn_version),
        },
        "timestamps": {
            "started_at": str(started_at),
            "ended_at": str(ended_at),
            "ended_at_scope": str(runtime_scope),
            "execution_timestamp": str(ended_at),
        },
        "warnings": [str(message) for message in warnings_list],
    }


def _report_value(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_run_report(
    config: dict[str, object],
    cluster_summary: pd.DataFrame,
    metric_values: dict[str, float | None],
) -> str:
    sample_counts = config["sample_counts"]
    settings = config["kmeans_parameters"]
    model_fit = config["model_fit"]
    timestamps = config["timestamps"]
    versions = config["versions"]
    runtime = config["runtime"]
    assert isinstance(sample_counts, dict)
    assert isinstance(settings, dict)
    assert isinstance(model_fit, dict)
    assert isinstance(timestamps, dict)
    assert isinstance(versions, dict)
    assert isinstance(runtime, dict)

    lines = [
        "K-means experiment report",
        "=========================",
        f"Scenario: {config['scenario']}",
        f"Prepared artifact: {config['input_path']}",
        f"Output directory: {config['output_path']}",
        f"K: {config['k']}",
        f"Features ({len(config['feature_list'])}): "
        f"{', '.join(config['feature_list'])}",
        "",
        "Sample counts",
        f"- Original: {sample_counts['original']}",
        f"- Valid: {sample_counts['valid']}",
        f"- Removed: {sample_counts['removed']}",
        "",
        "Excluded columns",
    ]
    exclusions = config["excluded_columns"]
    if exclusions:
        lines.extend(
            f"- {item['column']}: {item['reason']}" for item in exclusions
        )
    else:
        lines.append("- None")

    lines.extend(["", "Missing-data treatment"])
    missing_counts = config["missing_counts"]
    lines.append(f"- Policy: {config['missing_value_policy']}")
    if missing_counts:
        lines.extend(
            f"- {column}: {count} missing value(s)"
            for column, count in missing_counts.items()
        )
    else:
        lines.append("- No missing values in active features")

    lines.extend(["", "Data-quality issues"])
    issues = config["data_quality_issues"]
    if issues:
        lines.extend(
            f"- {item['issue_type']} | {item['column'] or '(all rows)'} | "
            f"count={item['count']} | {item['action']}: {item['details']}"
            for item in issues
        )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "Scaling",
            "The input data had already been standardized with StandardScaler.",
            "This script did not apply StandardScaler or any other scaling transformation.",
            "",
            "K-means settings",
            f"- init: {settings['init']}",
            f"- n_init: {settings['n_init']}",
            f"- max_iter: {settings['max_iter']}",
            f"- random_state: {settings['random_state']}",
            f"- algorithm: {settings['algorithm']}",
            f"- tolerance: {settings['tol']}",
            f"- iterations: {model_fit['n_iter']}",
            f"- converged: {model_fit['converged']}",
            f"- fit seconds: {_report_value(model_fit['fit_seconds'])}",
            "",
            "Cluster label ordering",
            f"- Rule: {config['cluster_label_ordering']}",
            "",
            "Cluster membership",
        ]
    )
    for row in cluster_summary.sort_values("cluster_id").itertuples(index=False):
        lines.append(
            f"- Cluster {int(row.cluster_id)}: {int(row.sample_count)} samples "
            f"({float(row.sample_percentage):.2f}%)"
        )

    lines.extend(
        [
            "",
            "Model metrics",
            f"- inertia: {_report_value(model_fit['inertia'])}",
            "- silhouette_score: "
            f"{_report_value(metric_values.get('silhouette_score'))}",
            "- calinski_harabasz_score: "
            f"{_report_value(metric_values.get('calinski_harabasz_score'))}",
            "- davies_bouldin_score: "
            f"{_report_value(metric_values.get('davies_bouldin_score'))}",
            "",
            "PCA explained variance",
        ]
    )
    pca_variance = config["pca_explained_variance_percent"]
    if pca_variance is None:
        lines.append("- Two-component PCA unavailable")
    else:
        lines.extend(
            [
                f"- PC1: {_report_value(pca_variance['pc1'])}%",
                f"- PC2: {_report_value(pca_variance['pc2'])}%",
                f"- PC1 + PC2: {_report_value(pca_variance['pc1_plus_pc2'])}%",
            ]
        )

    lines.extend(
        [
            "",
            "Persisted staging-complete runtime: "
            f"{float(runtime['runtime_seconds']):.3f} seconds",
            f"Runtime scope: {runtime['runtime_scope']}",
            (
                "End-to-end runtime, including the final metadata rewrite and "
                "atomic publication, is printed in the terminal."
            ),
            "",
            "Timestamps",
            f"- Started: {timestamps['started_at']}",
            f"- Artifact snapshot: {timestamps['ended_at']}",
            f"- Snapshot scope: {timestamps['ended_at_scope']}",
            f"- Execution timestamp: {timestamps['execution_timestamp']}",
            "",
            "Software versions",
            f"- Python: {versions['python']}",
            f"- pandas: {versions['pandas']}",
            f"- NumPy: {versions['numpy']}",
            f"- scikit-learn: {versions['scikit_learn']}",
            "",
            "Warnings",
        ]
    )
    report_warnings = config["warnings"]
    if report_warnings:
        lines.extend(f"- {message}" for message in report_warnings)
    else:
        lines.append("- None")

    lines.extend(["", "Output files"])
    lines.extend(f"- {filename}" for filename in ALL_OUTPUTS)
    return "\n".join(lines) + "\n"


def _write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    try:
        frame.to_csv(output_path, index=False, encoding="utf-8")
    except PermissionError as error:
        raise KMeansAnalysisError(
            f"Permission denied while writing output: {output_path}"
        ) from error
    except OSError as error:
        raise KMeansAnalysisError(
            f"Could not write output {output_path}: {error}"
        ) from error


def export_data_tables(
    output_directory: Path, tables: AnalysisTables
) -> None:
    """Write the bulk tabular outputs before timing the artifact snapshot."""
    output_directory = Path(output_directory)
    frame_outputs = (
        ("clustered_data.csv", tables.clustered_data),
        ("cluster_summary.csv", tables.cluster_summary),
        (
            "cluster_centroids_standardized.csv",
            tables.cluster_centroids_standardized.reset_index(),
        ),
        (
            "cluster_centroids_original_scale.csv",
            tables.cluster_centroids_original_scale.reset_index(),
        ),
        ("cluster_feature_profile.csv", tables.cluster_feature_profile),
        ("sample_distances.csv", tables.sample_distances),
        ("cluster_label_mapping.csv", tables.label_mapping),
    )
    for filename, frame in frame_outputs:
        _write_csv(frame, output_directory / filename)
    shutil.copyfile(
        tables.preprocessing_audit_path,
        output_directory / "preprocessing_audit.csv",
    )


def export_run_metadata(
    output_directory: Path,
    metrics: MetricResults,
    config: dict[str, object],
    report: str,
) -> None:
    """Write metrics and metadata from one internally consistent snapshot."""
    output_directory = Path(output_directory)
    sample_counts = config["sample_counts"]
    settings = config["kmeans_parameters"]
    model_fit = config["model_fit"]
    timestamps = config["timestamps"]
    runtime = config["runtime"]
    assert isinstance(sample_counts, dict)
    assert isinstance(settings, dict)
    assert isinstance(model_fit, dict)
    assert isinstance(timestamps, dict)
    assert isinstance(runtime, dict)
    model_metrics = pd.DataFrame(
        [
            {
                "scenario": config["scenario"],
                "input_csv": config["input_path"],
                "n_original_samples": sample_counts["original"],
                "n_valid_samples": sample_counts["valid"],
                "n_features": len(config["feature_list"]),
                "K": config["k"],
                "inertia": model_fit["inertia"],
                "silhouette_score": metrics.values.get("silhouette_score"),
                "calinski_harabasz_score": metrics.values.get(
                    "calinski_harabasz_score"
                ),
                "davies_bouldin_score": metrics.values.get(
                    "davies_bouldin_score"
                ),
                "n_iter": model_fit["n_iter"],
                "converged": model_fit["converged"],
                "init": settings["init"],
                "n_init": settings["n_init"],
                "max_iter": settings["max_iter"],
                "random_state": settings["random_state"],
                "runtime_seconds": float(runtime["runtime_seconds"]),
                "runtime_scope": runtime["runtime_scope"],
                "timestamp": timestamps["execution_timestamp"],
            }
        ]
    )
    _write_csv(model_metrics, output_directory / "model_metrics.csv")

    text_outputs = (
        ("run_config.json", json.dumps(config, ensure_ascii=False, indent=2) + "\n"),
        ("run_report.txt", report),
    )
    for filename, contents in text_outputs:
        output_path = output_directory / filename
        try:
            output_path.write_text(contents, encoding="utf-8")
        except PermissionError as error:
            raise KMeansAnalysisError(
                f"Permission denied while writing output: {output_path}"
            ) from error
        except OSError as error:
            raise KMeansAnalysisError(
                f"Could not write output {output_path}: {error}"
            ) from error


def export_results(
    output_directory: Path,
    tables: AnalysisTables,
    metrics: MetricResults,
    config: dict[str, object],
    report: str,
) -> None:
    """Write a complete result set when phase-level orchestration is unnecessary."""
    export_data_tables(output_directory, tables)
    export_run_metadata(output_directory, metrics, config, report)


def generate_pca_plot(
    ordering: ClusterOrdering,
    scenario: str,
    k: int,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    """Write a PCA projection of the reordered cluster assignments."""
    if ordering.pca_scores is None or ordering.explained_variance_ratio is None:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "Two-component PCA is unavailable.\n"
            "K-means was still fitted in the full standardized feature space.",
            ha="center",
            va="center",
        )
        figure.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        plt.close(figure)
        return

    figure, axis = plt.subplots(figsize=(8, 6))
    cluster_ids = np.unique(ordering.reordered_labels)
    colors = plt.get_cmap("tab10", len(cluster_ids))
    for color_index, cluster_id in enumerate(cluster_ids):
        member_mask = ordering.reordered_labels == cluster_id
        axis.scatter(
            ordering.pca_scores[member_mask, 0],
            ordering.pca_scores[member_mask, 1],
            color=colors(color_index),
            label=f"Cluster {cluster_id}",
            alpha=0.8,
        )
    explained = ordering.explained_variance_ratio * 100.0
    axis.set_xlabel(f"PC1 ({explained[0]:.1f}% explained variance)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1f}% explained variance)")
    axis.set_title(
        f"{scenario} scenario: K={k}, {len(feature_names)} clustering features"
    )
    axis.text(
        0.5,
        -0.18,
        "PCA projection only; K-means was fitted in the full standardized feature space.",
        transform=axis.transAxes,
        ha="center",
        va="top",
    )
    axis.legend(title="Reordered cluster")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)


def generate_cluster_size_plot(cluster_summary: pd.DataFrame, output_path: Path) -> None:
    """Write a bar chart of the member count and share for every cluster."""
    summary = cluster_summary.sort_values("cluster_id")
    cluster_ids = summary["cluster_id"].to_numpy(dtype=int)
    counts = summary["sample_count"].to_numpy(dtype=float)
    percentages = summary["sample_percentage"].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(max(7, len(summary) * 1.2), 5))
    bars = axis.bar(
        np.arange(len(summary)), counts, color=plt.get_cmap("tab10", len(summary)).colors
    )
    for bar, count, percentage in zip(bars, counts, percentages):
        axis.annotate(
            f"{int(count)}\n({percentage:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    axis.set_xticks(np.arange(len(summary)), [f"Cluster {item}" for item in cluster_ids])
    axis.set_ylabel("Sample count")
    axis.set_title("Cluster sizes")
    axis.set_ylim(top=max(float(np.nanmax(counts)) * 1.18, 1.0))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)


def generate_centroid_heatmap(
    centroids: pd.DataFrame,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    """Write an annotated, zero-centered heatmap of standardized centroids."""
    values = centroids.loc[:, list(feature_names)].to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(values))), 1e-12)
    row_labels = [f"Cluster {cluster_id}" for cluster_id in centroids.index]
    maximum_label_length = max(
        [len(label) for label in feature_names] + [len(label) for label in row_labels]
    )
    figure, axis = plt.subplots(
        figsize=(
            max(8, len(feature_names) * 1.1, maximum_label_length * 0.45),
            max(4.5, len(row_labels) * 0.8 + 2),
        )
    )
    image = axis.imshow(
        values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
    )
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            color = "white" if abs(value) > limit * 0.55 else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    axis.set_xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title("Cluster centroid heatmap")
    figure.colorbar(image, ax=axis, label="Standardized centroid")
    figure.tight_layout()
    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)


def generate_profile_plot(
    feature_profile: pd.DataFrame,
    feature_names: Sequence[str],
    output_path: Path,
) -> None:
    """Write cluster mean feature profiles in configured feature order."""
    ordered_features = list(feature_names)
    cluster_ids = sorted(feature_profile["cluster"].unique())
    figure, axis = plt.subplots(figsize=(max(8, len(ordered_features) * 1.15), 6))
    colors = plt.get_cmap("tab10", len(cluster_ids))
    positions = np.arange(len(ordered_features))
    for color_index, cluster_id in enumerate(cluster_ids):
        profile = feature_profile.loc[
            feature_profile["cluster"] == cluster_id,
            ["feature", "inverse_transformed_centroid"],
        ].set_index("feature")
        means = profile.reindex(ordered_features)[
            "inverse_transformed_centroid"
        ].to_numpy(dtype=float)
        axis.plot(
            positions,
            means,
            marker="o",
            color=colors(color_index),
            label=f"Cluster {cluster_id}",
        )
    axis.set_xticks(positions, ordered_features, rotation=45, ha="right")
    axis.set_ylabel("Inverse-transformed centroid (original scale)")
    axis.set_title("Cluster feature profiles (original scale)")
    axis.legend(title="Reordered cluster")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)


def prepare_data(prepared_path: Path) -> ValidatedData:
    """Load the single verified matrix boundary used by all clustering tools."""
    print("[1/7] Loading verified prepared artifact...")
    try:
        artifact = load_prepared_artifact(prepared_path)
    except (OSError, ValueError) as error:
        raise DataValidationError(f"Prepared artifact is invalid: {error}") from error
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise DataValidationError("Prepared standardized matrix is not finite.")
    return ValidatedData(
        artifact=artifact,
        values=values,
        feature_names=artifact.feature_names,
        metadata=artifact.metadata.copy(),
        row_count=len(artifact.metadata),
    )


def print_preflight_summary(
    scenario: str,
    prepared_path: Path,
    k: int,
    validated: ValidatedData,
) -> None:
    artifact = validated.artifact
    manifest_path = (artifact.directory / "preprocessing_config.json").resolve()
    print("K-means analysis preflight:")
    print(f"Scenario: {scenario}")
    print(f"Prepared artifact: {Path(prepared_path).resolve()}")
    print(f"Source path: {artifact.config['source_path']}")
    print(f"Source SHA-256: {artifact.config['source_sha256']}")
    print(f"Manifest path: {manifest_path}")
    print(f"Manifest SHA-256: {sha256_file(manifest_path)}")
    print(f"Standardized matrix SHA-256: {artifact.config['matrix_sha256']}")
    print(f"K: {k}")
    print(
        f"Features ({len(validated.feature_names)}): "
        f"{', '.join(validated.feature_names)}"
    )
    print(f"Prepared rows: {validated.row_count}")
    print(
        "Rows with structural null: "
        f"{int(artifact.metadata['had_structural_null'].sum())}"
    )
    print("Rows excluded for null = 0")
    print("No scaling will be applied; the verified standardized matrix is used exactly.")


def print_completion_summary(
    scenario: str,
    k: int,
    validated: ValidatedData,
    cluster_summary: pd.DataFrame,
    metrics: dict[str, float | None],
    inertia: float,
    runtime_seconds: float,
    output_path: Path,
) -> None:
    silhouette = metrics.get("silhouette_score")
    silhouette_text = (
        "not available" if silhouette is None else f"{silhouette:.6f}"
    )
    print("K-means analysis completed.")
    print(f"Scenario: {scenario}")
    print(f"K: {k}")
    print(
        f"Features ({len(validated.feature_names)}): "
        f"{', '.join(validated.feature_names)}"
    )
    print(f"Prepared samples: {validated.row_count}")
    print(f"Output folder: {output_path}")
    print("Cluster sizes:")
    for row in cluster_summary.itertuples(index=False):
        print(
            f"- Cluster {int(row.cluster_id)}: "
            f"{int(row.sample_count)} samples "
            f"({float(row.sample_percentage):.2f}%)"
        )
    print(f"Silhouette score: {silhouette_text}")
    print(f"Inertia: {inertia:.6f}")
    print(f"End-to-end runtime: {runtime_seconds:.2f} seconds")


def _run_validated_analysis(
    scenario: str,
    k: int,
    prepared_path: Path,
    output_root: Path,
    validated: ValidatedData,
    started_at_dt: datetime,
    started_perf: float,
) -> Path:
    if not 2 <= k < validated.row_count:
        raise DataValidationError(
            f"K must satisfy 2 <= K < {validated.row_count}; received {k}."
        )
    final_path = build_output_directory(
        output_root, scenario, k, len(validated.feature_names)
    )
    print_preflight_summary(scenario, prepared_path, k, validated)

    print("[4/7] Running K-means...")
    values = validated.values
    fitted = run_kmeans(values, k)
    ordering = reorder_cluster_labels(
        fitted, values, validated.feature_names
    )

    print("[5/7] Calculating metrics and profiles...")
    metric_results = calculate_metrics(values, ordering.reordered_labels)
    tables = calculate_cluster_profiles(validated, fitted, ordering)
    warnings_list = (
        list(fitted.warnings)
        + list(ordering.warnings)
        + list(metric_results.warnings)
    )

    with staged_output_directory(final_path) as staging:
        print("[6/7] Generating visualizations...")
        generate_pca_plot(
            ordering,
            scenario,
            k,
            validated.feature_names,
            staging / "pca_cluster_scatter.png",
        )
        generate_cluster_size_plot(
            tables.cluster_summary, staging / "cluster_size_bar.png"
        )
        generate_centroid_heatmap(
            tables.cluster_centroids_standardized,
            validated.feature_names,
            staging / "cluster_centroid_heatmap.png",
        )
        generate_profile_plot(
            tables.cluster_feature_profile,
            validated.feature_names,
            staging / "cluster_feature_profiles.png",
        )

        print("[7/7] Exporting results...")
        export_data_tables(staging, tables)
        provisional_snapshot_dt = datetime.now().astimezone()
        provisional_runtime_seconds = time.perf_counter() - started_perf
        provisional_config = build_run_config(
            scenario=scenario,
            input_path=prepared_path.resolve(),
            output_path=final_path.resolve(),
            k=k,
            validated=validated,
            fitted=fitted,
            ordering=ordering,
            warnings_list=warnings_list,
            started_at=started_at_dt.isoformat(timespec="seconds"),
            ended_at=provisional_snapshot_dt.isoformat(timespec="seconds"),
            runtime_seconds=provisional_runtime_seconds,
            runtime_scope=PROVISIONAL_RUNTIME_SCOPE,
        )
        provisional_report = build_run_report(
            provisional_config,
            tables.cluster_summary,
            metric_results.values,
        )
        export_run_metadata(
            staging,
            metric_results,
            provisional_config,
            provisional_report,
        )

        staging_complete_dt = datetime.now().astimezone()
        staging_complete_runtime_seconds = (
            time.perf_counter() - started_perf
        )
        final_config = build_run_config(
            scenario=scenario,
            input_path=prepared_path.resolve(),
            output_path=final_path.resolve(),
            k=k,
            validated=validated,
            fitted=fitted,
            ordering=ordering,
            warnings_list=warnings_list,
            started_at=started_at_dt.isoformat(timespec="seconds"),
            ended_at=staging_complete_dt.isoformat(timespec="seconds"),
            runtime_seconds=staging_complete_runtime_seconds,
        )
        final_report = build_run_report(
            final_config,
            tables.cluster_summary,
            metric_results.values,
        )
        export_run_metadata(
            staging,
            metric_results,
            final_config,
            final_report,
        )

    end_to_end_runtime_seconds = time.perf_counter() - started_perf
    print_completion_summary(
        scenario,
        k,
        validated,
        tables.cluster_summary,
        metric_results.values,
        fitted.inertia,
        end_to_end_runtime_seconds,
        final_path,
    )
    return final_path


def run_analysis(
    prepared_path: Path,
    k: int,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    started_at_dt = datetime.now().astimezone()
    started_perf = time.perf_counter()
    validated = prepare_data(prepared_path)
    return _run_validated_analysis(
        str(validated.artifact.config["scenario"]),
        k,
        prepared_path,
        output_root,
        validated,
        started_at_dt,
        started_perf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run K-means on a verified prepared clustering artifact."
    )
    parser.add_argument("--prepared", type=Path, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        prepared_path = args.prepared if args.prepared is not None else prompt_prepared_path()
        validated = prepare_data(prepared_path)
        k = args.k if args.k is not None else prompt_k(validated.row_count)
        _run_validated_analysis(
            str(validated.artifact.config["scenario"]),
            k,
            prepared_path,
            args.output_root,
            validated,
            datetime.now().astimezone(),
            time.perf_counter(),
        )
    except (
        KMeansAnalysisError,
        PermissionError,
        OSError,
    ) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
