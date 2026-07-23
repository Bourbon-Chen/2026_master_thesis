#!/usr/bin/env python3
"""Run one reproducible K-means experiment on prepared PF or TF data."""

from __future__ import annotations

import contextlib
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
from pandas.api.types import is_numeric_dtype

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.scenario_features import (
    DEFAULT_INPUT_PATHS,
    PF_FEATURES,
    SCENARIO_CONFIG,
    TF_FEATURES,
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
    "cluster_feature_profile.csv",
    "sample_distances.csv",
    "model_metrics.csv",
    "cluster_label_mapping.csv",
    "data_quality_issues.csv",
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

EXCLUSION_KEYWORDS = (
    "index",
    "risk",
    "cluster",
    "label",
    "type",
    "typology",
    "intervention",
    "score",
    "geometry",
)
EXACT_ID_NAMES = {
    "id",
    "fid",
    "gid",
    "objectid",
    "ogc_fid",
    "ms_id",
    "street_id",
    "segment_id",
}


class KMeansAnalysisError(RuntimeError):
    """Base exception for expected user-facing workflow errors."""


class DataValidationError(KMeansAnalysisError):
    """Raised when prepared input cannot safely enter K-means."""


class CompatibilityError(KMeansAnalysisError):
    """Raised when installed scikit-learn cannot honor configured settings."""


class OutputDirectoryExistsError(KMeansAnalysisError):
    """Raised when a readable final experiment directory already exists."""


@dataclass(frozen=True)
class ColumnExclusion:
    column: str
    reason: str


@dataclass(frozen=True)
class DataQualityIssue:
    issue_type: str
    column: str
    count: int
    action: str
    details: str


@dataclass
class FeatureSelection:
    feature_names: tuple[str, ...]
    exclusions: list[ColumnExclusion]
    id_columns: tuple[str, ...]
    issues: list[DataQualityIssue]


@dataclass
class ValidatedData:
    original_data: pd.DataFrame
    features: pd.DataFrame
    valid_mask: pd.Series
    feature_names: tuple[str, ...]
    exclusions: list[ColumnExclusion]
    id_columns: tuple[str, ...]
    missing_counts: dict[str, int]
    issues: list[DataQualityIssue]


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
    centroids: pd.DataFrame
    feature_profile: pd.DataFrame
    sample_distances: pd.DataFrame
    label_mapping: pd.DataFrame
    data_quality_issues: pd.DataFrame


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


def normalize_scenario(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"1": "PF", "pf": "PF", "2": "TF", "tf": "TF"}
    if normalized not in aliases:
        raise ValueError("Invalid selection. Enter 1/pf or 2/tf.")
    return aliases[normalized]


def prompt_scenario() -> str:
    while True:
        print("Select flood scenario:")
        print("1. PF")
        print("2. TF")
        try:
            return normalize_scenario(input().strip())
        except ValueError as error:
            print(str(error))


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


def _resolve_readable_csv_path(candidate: Path) -> Path:
    """Resolve a candidate and verify it can be opened before returning it."""
    resolved = Path(candidate).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(os.fspath(resolved))
    if not resolved.is_file():
        raise IsADirectoryError(os.fspath(resolved))
    with resolved.open("rb") as stream:
        stream.read(1)
    return resolved


def prompt_csv_path(scenario: str) -> Path:
    default_path = DEFAULT_INPUT_PATHS[scenario]
    default_name = default_path.name
    while True:
        entered = input(f"Enter CSV path [default: {default_name}]: ")
        candidate = default_path if not entered.strip() else Path(
            _strip_matching_quotes(entered)
        )
        try:
            resolved = _resolve_readable_csv_path(candidate)
        except FileNotFoundError:
            print(f"CSV file not found: {candidate}")
            continue
        except IsADirectoryError:
            print(f"CSV path is not a file: {candidate}")
            continue
        except PermissionError as error:
            print(f"CSV file is not readable: {candidate}: {error}")
            continue
        except (OSError, RuntimeError) as error:
            print(f"Could not access CSV path {candidate}: {error}")
            continue
        return resolved


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError as error:
        raise KMeansAnalysisError(f"CSV file not found: {path}") from error
    except PermissionError as error:
        raise KMeansAnalysisError(f"Permission denied while reading: {path}") from error
    except pd.errors.EmptyDataError as error:
        raise KMeansAnalysisError(f"CSV is empty: {path}") from error
    except pd.errors.ParserError as error:
        raise KMeansAnalysisError(f"CSV could not be parsed: {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise KMeansAnalysisError(f"CSV is not valid UTF-8 text: {path}") from error


def is_id_column(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in EXACT_ID_NAMES
        or normalized.endswith("_id")
        or normalized.startswith("id_")
    )


def _retained_exclusion_reason(column: str, series: pd.Series) -> str:
    if is_id_column(column):
        return "identifier field"
    if not is_numeric_dtype(series):
        return "non-numeric retained field"
    lowered = column.lower()
    for keyword in EXCLUSION_KEYWORDS:
        if keyword in lowered:
            return f"excluded keyword: {keyword}"
    return "not in scenario feature configuration"


def select_features(data: pd.DataFrame, scenario: str) -> FeatureSelection:
    if "kmeans_cluster" in data.columns:
        raise DataValidationError(
            "Reserved output column already exists: kmeans_cluster"
        )
    configured = tuple(SCENARIO_CONFIG[scenario]["features"])
    for column in configured:
        if column not in data.columns:
            raise DataValidationError(f"Missing configured feature: {column}")
        if not is_numeric_dtype(data[column]):
            raise DataValidationError(
                f"Configured feature is not numeric: {column}"
            )

    exclusions = [
        ColumnExclusion(column, _retained_exclusion_reason(column, data[column]))
        for column in data.columns
        if column not in configured
    ]
    id_columns = tuple(column for column in data.columns if is_id_column(column))
    return FeatureSelection(configured, exclusions, id_columns, [])


def validate_features(
    data: pd.DataFrame, selection: FeatureSelection
) -> ValidatedData:
    active = list(selection.feature_names)
    exclusions = list(selection.exclusions)
    issues = list(selection.issues)

    for column in list(active):
        series = data[column].astype(float)
        finite_nonmissing = series.dropna()
        if series.isna().all():
            reason = "all values are missing"
            issue_count = int(series.isna().sum())
        elif np.isinf(series.to_numpy(dtype=float)).any():
            reason = "contains infinity"
            issue_count = int(np.isinf(series.to_numpy(dtype=float)).sum())
        elif finite_nonmissing.nunique() <= 1:
            reason = "constant feature"
            issue_count = int(finite_nonmissing.size)
        else:
            continue
        active.remove(column)
        exclusions.append(ColumnExclusion(column, reason))
        issues.append(
            DataQualityIssue(reason, column, issue_count, "excluded feature", reason)
        )

    while active:
        frame = data.loc[:, active].astype(float)
        valid_mask = frame.notna().all(axis=1)
        valid_frame = frame.loc[valid_mask]
        post_drop_constants = [
            column for column in active if valid_frame[column].nunique() <= 1
        ]
        if not post_drop_constants:
            break
        for column in post_drop_constants:
            active.remove(column)
            exclusions.append(
                ColumnExclusion(column, "constant after missing-row removal")
            )
            issues.append(
                DataQualityIssue(
                    "constant_after_row_removal",
                    column,
                    int(len(valid_frame)),
                    "excluded feature",
                    "Only one valid value remained after complete-case filtering.",
                )
            )

    if not active:
        raise DataValidationError("No usable clustering features remain.")

    frame = data.loc[:, active].astype(float)
    missing_counts = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if int(count) > 0
    }
    valid_mask = frame.notna().all(axis=1)
    valid_frame = frame.loc[valid_mask].copy()
    if not np.isfinite(valid_frame.to_numpy(dtype=float)).all():
        raise DataValidationError("Non-finite values remain after feature validation.")
    removed_count = int((~valid_mask).sum())
    for column, count in missing_counts.items():
        issues.append(
            DataQualityIssue(
                "missing_values",
                column,
                count,
                "removed affected rows",
                "No imputation was applied.",
            )
        )
    if removed_count:
        issues.append(
            DataQualityIssue(
                "invalid_rows",
                "",
                removed_count,
                "excluded from K-means",
                "Rows remain in clustered_data.csv with an empty label.",
            )
        )
    if len(valid_frame) < 3:
        raise DataValidationError(
            "At least 3 valid samples are required because 2 <= K < n_samples."
        )
    return ValidatedData(
        original_data=data.copy(),
        features=valid_frame,
        valid_mask=valid_mask,
        feature_names=tuple(active),
        exclusions=exclusions,
        id_columns=selection.id_columns,
        missing_counts=missing_counts,
        issues=issues,
    )


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
        centroids[original_order], columns=list(feature_names)
    )
    reordered_centroids.insert(
        0, "cluster_id", np.arange(1, len(original_order) + 1)
    )
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
    valid_positions = np.flatnonzero(validated.valid_mask.to_numpy())
    full_labels = pd.array([pd.NA] * len(validated.original_data), dtype="Int64")
    full_labels[valid_positions] = ordering.reordered_labels
    clustered = validated.original_data.copy()
    clustered["kmeans_cluster"] = full_labels

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
        members = validated.features.iloc[np.flatnonzero(member_mask)]
        for feature in validated.feature_names:
            values = members[feature]
            profiles.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "mean": float(values.mean()) if count else np.nan,
                    "median": float(values.median()) if count else np.nan,
                    "std": float(values.std(ddof=0)) if count else np.nan,
                    "min": float(values.min()) if count else np.nan,
                    "max": float(values.max()) if count else np.nan,
                    "q25": float(values.quantile(0.25)) if count else np.nan,
                    "q75": float(values.quantile(0.75)) if count else np.nan,
                }
            )

    source_valid = validated.original_data.iloc[valid_positions]
    distance_table = source_valid.loc[:, list(validated.id_columns)].reset_index(
        drop=True
    )
    distance_table["kmeans_cluster"] = ordering.reordered_labels
    distance_table["distance_to_assigned_centroid"] = fitted.assigned_distances
    issues = pd.DataFrame(
        [
            {
                "issue_type": issue.issue_type,
                "column": issue.column,
                "count": issue.count,
                "action": issue.action,
                "details": issue.details,
            }
            for issue in validated.issues
        ],
        columns=["issue_type", "column", "count", "action", "details"],
    )
    return AnalysisTables(
        clustered_data=clustered,
        cluster_summary=pd.DataFrame(summaries),
        centroids=ordering.reordered_centroids.copy(),
        feature_profile=pd.DataFrame(profiles),
        sample_distances=distance_table,
        label_mapping=ordering.mapping.copy(),
        data_quality_issues=issues,
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

    original_count = int(len(validated.original_data))
    valid_count = int(len(validated.features))
    return {
        "scenario": str(scenario),
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        "k": int(k),
        "feature_list": [str(name) for name in validated.feature_names],
        "excluded_columns": [
            {"column": str(item.column), "reason": str(item.reason)}
            for item in validated.exclusions
        ],
        "id_columns": [str(name) for name in validated.id_columns],
        "data_quality_issues": [
            {
                "issue_type": str(item.issue_type),
                "column": str(item.column),
                "count": int(item.count),
                "action": str(item.action),
                "details": str(item.details),
            }
            for item in validated.issues
        ],
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
            "original": original_count,
            "valid": valid_count,
            "removed": original_count - valid_count,
        },
        "missing_counts": {
            str(name): int(count)
            for name, count in validated.missing_counts.items()
        },
        "missing_value_policy": (
            "Complete-case row removal for active clustering features; "
            "no imputation was applied."
        ),
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
        f"Input CSV: {config['input_path']}",
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
        ("cluster_centroids_standardized.csv", tables.centroids),
        ("cluster_feature_profile.csv", tables.feature_profile),
        ("sample_distances.csv", tables.sample_distances),
        ("cluster_label_mapping.csv", tables.label_mapping),
        ("data_quality_issues.csv", tables.data_quality_issues),
    )
    for filename, frame in frame_outputs:
        _write_csv(frame, output_directory / filename)


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
    row_labels = [f"Cluster {cluster_id}" for cluster_id in centroids["cluster_id"]]
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
    cluster_ids = sorted(feature_profile["cluster_id"].unique())
    figure, axis = plt.subplots(figsize=(max(8, len(ordered_features) * 1.15), 6))
    colors = plt.get_cmap("tab10", len(cluster_ids))
    positions = np.arange(len(ordered_features))
    for color_index, cluster_id in enumerate(cluster_ids):
        profile = feature_profile.loc[
            feature_profile["cluster_id"] == cluster_id, ["feature", "mean"]
        ].set_index("feature")
        means = profile.reindex(ordered_features)["mean"].to_numpy(dtype=float)
        axis.plot(
            positions,
            means,
            marker="o",
            color=colors(color_index),
            label=f"Cluster {cluster_id}",
        )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(positions, ordered_features, rotation=45, ha="right")
    axis.set_ylabel("Mean feature value")
    axis.set_title("Cluster feature profiles")
    axis.legend(title="Reordered cluster")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close(figure)


def prepare_data(scenario: str, input_path: Path) -> ValidatedData:
    print("[1/7] Reading input data...")
    data = load_dataset(input_path)
    print("[2/7] Selecting clustering features...")
    selection = select_features(data, scenario)
    print("[3/7] Validating data...")
    return validate_features(data, selection)


def print_preflight_summary(
    scenario: str,
    input_path: Path,
    k: int,
    validated: ValidatedData,
) -> None:
    original_count = len(validated.original_data)
    valid_count = len(validated.features)
    print("K-means analysis preflight:")
    print(f"Scenario: {scenario}")
    print(f"Input CSV: {Path(input_path).resolve()}")
    print(f"K: {k}")
    print(
        f"Features ({len(validated.feature_names)}): "
        f"{', '.join(validated.feature_names)}"
    )
    print(f"Original samples: {original_count}")
    print(f"Valid samples: {valid_count}")
    print(f"Rows excluded from K-means: {original_count - valid_count}")
    print("Excluded columns:")
    if validated.exclusions:
        for exclusion in validated.exclusions:
            print(f"- {exclusion.column}: {exclusion.reason}")
    else:
        print("- None")
    print("Missing values in active features:")
    if validated.missing_counts:
        for column, count in validated.missing_counts.items():
            print(f"- {column}: {count}")
    else:
        print("- None")
    print("No StandardScaler, normalization, or other scaling will be applied.")


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
    print(f"Valid samples: {len(validated.features)}")
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
    input_path: Path,
    output_root: Path,
    validated: ValidatedData,
    started_at_dt: datetime,
    started_perf: float,
) -> Path:
    if k < 2 or k >= len(validated.features):
        raise DataValidationError(
            f"K must satisfy 2 <= K < {len(validated.features)} valid samples."
        )
    final_path = build_output_directory(
        output_root, scenario, k, len(validated.feature_names)
    )
    print_preflight_summary(scenario, input_path, k, validated)

    print("[4/7] Running K-means...")
    values = validated.features.to_numpy(dtype=float, copy=True)
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
            tables.centroids,
            validated.feature_names,
            staging / "cluster_centroid_heatmap.png",
        )
        generate_profile_plot(
            tables.feature_profile,
            validated.feature_names,
            staging / "cluster_feature_profiles.png",
        )

        print("[7/7] Exporting results...")
        export_data_tables(staging, tables)
        provisional_snapshot_dt = datetime.now().astimezone()
        provisional_runtime_seconds = time.perf_counter() - started_perf
        provisional_config = build_run_config(
            scenario=scenario,
            input_path=input_path.resolve(),
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
            input_path=input_path.resolve(),
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
    scenario: str,
    k: int,
    input_path: Path,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    started_at_dt = datetime.now().astimezone()
    started_perf = time.perf_counter()
    validated = prepare_data(scenario, input_path)
    return _run_validated_analysis(
        scenario,
        k,
        input_path,
        output_root,
        validated,
        started_at_dt,
        started_perf,
    )


def main() -> None:
    try:
        scenario = prompt_scenario()
        k = prompt_k()
        input_path = prompt_csv_path(scenario)
        started_at_dt = datetime.now().astimezone()
        started_perf = time.perf_counter()
        validated = prepare_data(scenario, input_path)
        if k >= len(validated.features):
            print(
                f"K must be smaller than the valid sample count "
                f"({len(validated.features)})."
            )
            k = prompt_k(n_samples=len(validated.features))

        _run_validated_analysis(
            scenario,
            k,
            input_path,
            OUTPUT_ROOT,
            validated,
            started_at_dt,
            started_perf,
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
