"""Canonical field classification for PF/TF clustering inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn
from sklearn.preprocessing import StandardScaler


class FieldRole(str, Enum):
    PF_FEATURE = "PF_FEATURE"
    TF_FEATURE = "TF_FEATURE"
    SHARED_FEATURE = "SHARED_FEATURE"
    IDENTIFIER = "IDENTIFIER"
    COMPOSITE_INDEX = "COMPOSITE_INDEX"
    RESULT_OR_LABEL = "RESULT_OR_LABEL"
    NON_NUMERIC = "NON_NUMERIC"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True)
class FeatureDiscovery:
    scenario: str
    feature_names: tuple[str, ...]
    column_roles: pd.DataFrame


@dataclass(frozen=True)
class PreparedSemanticFeatures:
    """Validated semantic features before consumer-specific preprocessing."""

    discovery: FeatureDiscovery
    original_features: pd.DataFrame
    filled_features: pd.DataFrame
    feature_names: tuple[str, ...]
    missing_counts: pd.Series
    row_null_counts: pd.Series
    rows_with_structural_null: pd.Series


@dataclass(frozen=True)
class PreparedClusteringData:
    """The auditable, single-scaler input boundary for clustering consumers."""

    scenario: str
    source_data: pd.DataFrame
    metadata: pd.DataFrame
    original_features: pd.DataFrame
    filled_features: pd.DataFrame
    standardized_features: pd.DataFrame
    feature_names: tuple[str, ...]
    missing_counts: pd.Series
    rows_with_structural_null: pd.Series
    excluded_columns: pd.DataFrame
    audit: pd.DataFrame
    scaler: StandardScaler
    scaler_parameters: pd.DataFrame


@dataclass(frozen=True)
class PreparedArtifact:
    directory: Path
    config: dict[str, object]
    metadata: pd.DataFrame
    original_features: pd.DataFrame
    filled_features: pd.DataFrame
    standardized_features: pd.DataFrame
    feature_names: tuple[str, ...]
    excluded_features: pd.DataFrame
    scaler_parameters: pd.DataFrame
    audit: pd.DataFrame


PF_PATTERN = re.compile(r"^(?:PF|N_PF|Per_)", re.IGNORECASE)
TF_PATTERN = re.compile(r"^(?:TF|N_TF|Tem_)", re.IGNORECASE)
IDENTIFIER_NAMES = frozenset({"fid", "ms_id"})
SHARED_FEATURE_NAMES = frozenset(
    {"street_type_nor", "resident_nor", "daynight_nor", "prm_nor"}
)
RESULT_OR_LABEL_PATTERN = re.compile(
    r"(?:cluster|label|rule|typology|intervention|result)", re.IGNORECASE
)
ARTIFACT_FILENAMES = frozenset(
    {
        "metadata.csv",
        "features_original.csv",
        "features_filled.csv",
        "features_standardized.csv",
        "selected_features.csv",
        "excluded_features.csv",
        "scaler_parameters.csv",
        "preprocessing_audit.csv",
        "preprocessing_config.json",
    }
)
SERIALIZED_ARTIFACT_FILENAMES = frozenset(
    name for name in ARTIFACT_FILENAMES if name != "preprocessing_config.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_order_sha256(feature_names: tuple[str, ...]) -> str:
    payload = "".join(f"{name}\n" for name in feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_scenario(value: object) -> str:
    """Return the canonical scenario name for accepted PF/TF selectors."""
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in {"PF", "TF"}:
            return normalized
        if normalized in {"1", "2"}:
            return {"1": "PF", "2": "TF"}[normalized]
    elif type(value) is int and value in {1, 2}:
        return {1: "PF", 2: "TF"}[value]
    raise ValueError("Scenario must be one of PF, TF, 1, or 2")


def classify_column(name: str, series: pd.Series) -> FieldRole:
    """Classify a source column using canonical, precedence-ordered rules."""
    normalized_name = name.casefold()
    if normalized_name in IDENTIFIER_NAMES:
        return FieldRole.IDENTIFIER
    if "index" in normalized_name or "risk" in normalized_name:
        return FieldRole.COMPOSITE_INDEX
    if RESULT_OR_LABEL_PATTERN.search(name):
        return FieldRole.RESULT_OR_LABEL
    if normalized_name in SHARED_FEATURE_NAMES:
        return FieldRole.SHARED_FEATURE
    if PF_PATTERN.match(name):
        return (
            FieldRole.PF_FEATURE
            if is_numeric_dtype(series)
            else FieldRole.NON_NUMERIC
        )
    if TF_PATTERN.match(name):
        return (
            FieldRole.TF_FEATURE
            if is_numeric_dtype(series)
            else FieldRole.NON_NUMERIC
        )
    return FieldRole.EXCLUDED


def discover_scenario_features(
    data: pd.DataFrame, scenario: object, exclusions: tuple[str, ...] = ()
) -> FeatureDiscovery:
    """Discover usable, numeric clustering features for one scenario."""
    canonical_scenario = normalize_scenario(scenario)
    active_role = (
        FieldRole.PF_FEATURE
        if canonical_scenario == "PF"
        else FieldRole.TF_FEATURE
    )
    source_columns = tuple(data.columns)
    absent_exclusions = [name for name in exclusions if name not in data.columns]
    if absent_exclusions:
        raise ValueError(
            "Explicit exclusions are absent from the table: "
            + ", ".join(absent_exclusions)
        )

    base_roles = {
        name: classify_column(name, data[name])
        for name in source_columns
    }
    invalid_exclusions = [
        name
        for name in exclusions
        if base_roles[name] is not active_role
        and not (
            base_roles[name] is FieldRole.NON_NUMERIC
            and (
                (canonical_scenario == "PF" and PF_PATTERN.match(name))
                or (canonical_scenario == "TF" and TF_PATTERN.match(name))
            )
        )
    ]
    if invalid_exclusions:
        raise ValueError(
            "Explicit exclusions must be active scenario features: "
            + ", ".join(invalid_exclusions)
        )

    excluded_names = frozenset(exclusions)
    records = []
    non_numeric_names = []
    for name in source_columns:
        role = base_roles[name]
        reason = ""
        is_active_marked_column = (
            canonical_scenario == "PF" and PF_PATTERN.match(name)
        ) or (canonical_scenario == "TF" and TF_PATTERN.match(name))
        if (
            is_active_marked_column
            and role is FieldRole.NON_NUMERIC
            and name not in excluded_names
        ):
            non_numeric_names.append(name)
            reason = "active_scenario_feature_is_non_numeric"
        if name in excluded_names:
            role = FieldRole.EXCLUDED
            reason = "explicit_research_exclusion"
        records.append({"column": name, "role": role.value, "reason": reason})

    if non_numeric_names:
        raise ValueError(
            "Active scenario features contain non-numeric columns: "
            + ", ".join(non_numeric_names)
        )

    column_roles = pd.DataFrame.from_records(records, columns=["column", "role", "reason"])
    feature_names = tuple(
        record["column"]
        for record in records
        if record["role"] == active_role.value
    )
    if not feature_names:
        raise ValueError(f"No usable {canonical_scenario} features discovered")
    return FeatureDiscovery(canonical_scenario, feature_names, column_roles)


def _validate_semantic_bounds(semantic_bounds: tuple[float, float]) -> tuple[float, float]:
    """Return finite, ordered inclusive semantic bounds."""
    if len(semantic_bounds) != 2:
        raise ValueError("semantic_bounds must contain exactly two values")
    lower, upper = semantic_bounds
    if not all(isinstance(value, (int, float, np.number)) for value in (lower, upper)):
        raise ValueError("semantic_bounds must be numeric")
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("semantic_bounds must be finite and ordered")
    return lower, upper


def _validate_semantic_values(
    original: pd.DataFrame, semantic_bounds: tuple[float, float]
) -> None:
    """Reject malformed or out-of-domain feature values before null filling."""
    lower, upper = _validate_semantic_bounds(semantic_bounds)
    for name in original.columns:
        series = original[name]
        if not is_numeric_dtype(series):
            raise ValueError(f"non-numeric active feature: {name}")
        finite_values = series.dropna().to_numpy(dtype="float64")
        if not np.isfinite(finite_values).all():
            raise ValueError(f"infinity in active feature: {name}")
        if ((finite_values < lower) | (finite_values > upper)).any():
            raise ValueError(
                f"outside semantic bounds [{lower:g}, {upper:g}] in active feature: {name}"
            )


def _scaler_parameters(
    original: pd.DataFrame, filled: pd.DataFrame, scaler: StandardScaler
) -> pd.DataFrame:
    """Persist all quantities needed to reproduce StandardScaler inversion."""
    return pd.DataFrame(
        {
            "feature": filled.columns,
            "mean": scaler.mean_,
            "variance": scaler.var_,
            "scale": scaler.scale_,
            "original_null_count": original.isna().sum().astype("int64").to_numpy(),
            "original_min": original.min(skipna=True).to_numpy(),
            "original_max": original.max(skipna=True).to_numpy(),
            "filled_min": filled.min().to_numpy(),
            "filled_max": filled.max().to_numpy(),
        }
    )


def prepare_semantic_features(
    data: pd.DataFrame,
    scenario: object,
    exclusions: tuple[str, ...] = (),
    semantic_min: float = -1.0,
    semantic_max: float = 1.0,
) -> PreparedSemanticFeatures:
    """Discover, validate, and structurally zero-fill scenario features."""
    if data.empty:
        raise ValueError("Input table must be non-empty")

    discovery = discover_scenario_features(data, scenario, exclusions)
    original = data.loc[:, discovery.feature_names].copy()
    _validate_semantic_values(original, (semantic_min, semantic_max))
    missing_counts = original.isna().sum().astype("int64")
    row_null_counts = original.isna().sum(axis=1).astype("int64")
    rows_with_structural_null = row_null_counts.gt(0)
    rows_with_structural_null.name = "had_structural_null"
    filled = original.fillna(0.0)
    if not np.isfinite(filled.to_numpy(dtype="float64")).all():
        raise ValueError("Filled semantic features contain non-finite values")

    return PreparedSemanticFeatures(
        discovery=discovery,
        original_features=original,
        filled_features=filled,
        feature_names=discovery.feature_names,
        missing_counts=missing_counts,
        row_null_counts=row_null_counts,
        rows_with_structural_null=rows_with_structural_null,
    )


def prepare_clustering_data(
    data: pd.DataFrame,
    scenario: object,
    exclusions: tuple[str, ...] = (),
    semantic_bounds: tuple[float, float] = (-1.0, 1.0),
) -> PreparedClusteringData:
    """Validate, structurally zero-fill, and standardize active scenario features."""
    if data.empty:
        raise ValueError("Input high-risk table must be non-empty")

    canonical_scenario = normalize_scenario(scenario)
    risk_column = f"{canonical_scenario}_Index_Risk_equal"
    required_columns = ("fid", "MS_ID", risk_column)
    missing_required = [name for name in required_columns if name not in data.columns]
    if missing_required:
        raise ValueError("Required metadata columns are missing: " + ", ".join(missing_required))

    semantic_min, semantic_max = _validate_semantic_bounds(semantic_bounds)
    semantic = prepare_semantic_features(
        data,
        canonical_scenario,
        exclusions,
        semantic_min=semantic_min,
        semantic_max=semantic_max,
    )
    discovery = semantic.discovery
    original = semantic.original_features
    filled = semantic.filled_features
    missing_counts = semantic.missing_counts
    constant_names = tuple(
        name for name in filled.columns if filled[name].nunique(dropna=False) <= 1
    )
    filled = filled.drop(columns=list(constant_names))
    original = original.drop(columns=list(constant_names))
    missing_counts = missing_counts.drop(index=list(constant_names))

    if filled.empty:
        raise ValueError(f"No usable {canonical_scenario} features remain after constant removal")

    # Only retained modelling features contribute to the per-row null audit.
    row_null_counts = original.isna().sum(axis=1).astype("int64")
    rows_with_structural_null = row_null_counts.gt(0)
    rows_with_structural_null.name = "had_structural_null"

    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(filled),
        columns=filled.columns,
        index=filled.index,
    )
    standardized_values = standardized.to_numpy(dtype="float64")
    if not np.isfinite(standardized_values).all():
        raise ValueError("Standardized features contain non-finite values")
    means = standardized.mean(axis=0).to_numpy(dtype="float64")
    deviations = standardized.std(axis=0, ddof=0).to_numpy(dtype="float64")
    if not np.allclose(means, np.zeros(len(means)), atol=1e-10, rtol=1e-10):
        raise ValueError("Standardized features do not have zero mean")
    if not np.allclose(deviations, np.ones(len(deviations)), atol=1e-10, rtol=1e-10):
        raise ValueError("Standardized features do not have unit population standard deviation")

    metadata = data.loc[:, required_columns].copy()
    metadata["source_row_position"] = np.arange(len(data), dtype=np.int64)
    metadata["structural_null_count"] = row_null_counts
    metadata["had_structural_null"] = rows_with_structural_null

    active_role = (
        FieldRole.PF_FEATURE.value
        if canonical_scenario == "PF"
        else FieldRole.TF_FEATURE.value
    )
    excluded_columns = discovery.column_roles.loc[
        discovery.column_roles["role"].ne(active_role)
    ].copy()
    if constant_names:
        constants = pd.DataFrame(
            {
                "column": constant_names,
                "role": FieldRole.EXCLUDED.value,
                "reason": "constant_after_structural_zero",
            }
        )
        excluded_columns = pd.concat([excluded_columns, constants], ignore_index=True)
    excluded_columns = excluded_columns.reset_index(drop=True)

    audit_records = [
        {
            "event": "constant_feature_removed",
            "severity": "warning",
            "column": name,
            "detail": "constant_after_structural_zero",
        }
        for name in constant_names
    ]
    if metadata["fid"].duplicated().any() or metadata["MS_ID"].duplicated().any():
        audit_records.append(
            {
                "event": "duplicate_identifier",
                "severity": "warning",
                "column": "fid,MS_ID",
                "detail": "source_row_position preserves row alignment",
            }
        )
    audit = pd.DataFrame.from_records(
        audit_records, columns=["event", "severity", "column", "detail"]
    )

    return PreparedClusteringData(
        scenario=canonical_scenario,
        source_data=data.copy(deep=True),
        metadata=metadata,
        original_features=original,
        filled_features=filled,
        standardized_features=standardized,
        feature_names=tuple(filled.columns),
        missing_counts=missing_counts,
        rows_with_structural_null=rows_with_structural_null,
        excluded_columns=excluded_columns,
        audit=audit,
        scaler=scaler,
        scaler_parameters=_scaler_parameters(original, filled, scaler),
    )


def inverse_transform_frame(
    standardized_features: pd.DataFrame, scaler_parameters: pd.DataFrame
) -> pd.DataFrame:
    """Reconstruct filled semantic values from persisted scaler parameters."""
    expected_columns = (
        "feature",
        "mean",
        "variance",
        "scale",
        "original_null_count",
        "original_min",
        "original_max",
        "filled_min",
        "filled_max",
    )
    if tuple(scaler_parameters.columns) != expected_columns:
        raise ValueError("scaler_parameters columns do not match the required schema")
    expected_features = tuple(scaler_parameters["feature"])
    if tuple(standardized_features.columns) != expected_features:
        raise ValueError("standardized feature order does not match scaler parameters")
    if scaler_parameters["feature"].duplicated().any():
        raise ValueError("scaler_parameters feature order contains duplicates")
    values = standardized_features.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("standardized features contain non-finite values")
    means = scaler_parameters["mean"].to_numpy(dtype="float64")
    scales = scaler_parameters["scale"].to_numpy(dtype="float64")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("scaler parameters contain invalid mean or scale values")
    return pd.DataFrame(
        values * scales + means,
        columns=standardized_features.columns,
        index=standardized_features.index,
    )


def _validate_prepared_artifact_input(prepared: PreparedClusteringData) -> None:
    """Reject internally inconsistent prepared data before writing any artifact."""
    scenario = normalize_scenario(prepared.scenario)
    if scenario != prepared.scenario:
        raise ValueError("Prepared scenario must use canonical PF or TF form")
    feature_names = prepared.feature_names
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Prepared feature order must be non-empty and unique")
    for matrix_name, matrix in (
        ("original", prepared.original_features),
        ("filled", prepared.filled_features),
        ("standardized", prepared.standardized_features),
    ):
        if tuple(matrix.columns) != feature_names:
            raise ValueError(
                f"Prepared {matrix_name} feature order does not match feature_names"
            )
    row_count = len(prepared.metadata)
    if len(prepared.source_data) != row_count or any(
        len(matrix) != row_count
        for matrix in (
            prepared.original_features,
            prepared.filled_features,
            prepared.standardized_features,
        )
    ):
        raise ValueError("Prepared metadata and feature matrix row counts differ")
    if "source_row_position" not in prepared.metadata.columns:
        raise ValueError("Prepared metadata is missing source_row_position")
    if not np.array_equal(
        prepared.metadata["source_row_position"].to_numpy(),
        np.arange(row_count, dtype=np.int64),
    ):
        raise ValueError("Prepared source-row positions are not contiguous")
    if tuple(prepared.scaler_parameters["feature"]) != feature_names:
        raise ValueError("Prepared scaler feature order does not match feature_names")
    try:
        standardized_values = prepared.standardized_features.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Prepared standardized features are not numeric") from exc
    if not np.isfinite(standardized_values).all():
        raise ValueError("Prepared standardized features contain non-finite values")


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dataframe_records_for_json(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(name): _json_scalar(value) for name, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def save_prepared_artifact(
    prepared: PreparedClusteringData,
    source_path: Path,
    output_root: Path,
    risk_filter: dict[str, object],
    exclusion_path: Path | None = None,
) -> Path:
    """Atomically publish one immutable, provenance-complete prepared artifact."""
    _validate_prepared_artifact_input(prepared)
    resolved_source = Path(source_path).resolve(strict=True)
    resolved_exclusion = (
        Path(exclusion_path).resolve(strict=True)
        if exclusion_path is not None
        else None
    )
    dataset_directory = Path(output_root).resolve() / resolved_source.stem
    final_directory = dataset_directory / prepared.scenario
    if final_directory.exists():
        raise FileExistsError(
            f"Prepared artifact is immutable and already exists: {final_directory}"
        )

    dataset_directory.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{prepared.scenario}.preparing-",
            dir=dataset_directory,
        )
    )
    try:
        prepared.metadata.to_csv(temporary_directory / "metadata.csv", index=False)
        prepared.original_features.to_csv(
            temporary_directory / "features_original.csv", index=False
        )
        prepared.filled_features.to_csv(
            temporary_directory / "features_filled.csv", index=False
        )
        prepared.standardized_features.to_csv(
            temporary_directory / "features_standardized.csv", index=False
        )
        pd.DataFrame({"feature": prepared.feature_names}).to_csv(
            temporary_directory / "selected_features.csv", index=False
        )
        prepared.excluded_columns.to_csv(
            temporary_directory / "excluded_features.csv", index=False
        )
        prepared.scaler_parameters.to_csv(
            temporary_directory / "scaler_parameters.csv", index=False
        )
        prepared.audit.to_csv(
            temporary_directory / "preprocessing_audit.csv", index=False
        )

        config: dict[str, object] = {
            "source_path": str(resolved_source),
            "source_sha256": sha256_file(resolved_source),
            "source_row_count": len(prepared.source_data),
            "scenario": prepared.scenario,
            "risk_filter": dict(risk_filter),
            "marker_rules": {
                "case_sensitive": False,
                "anchored": True,
                "PF": ["^PF", "^N_PF", "^Per_"],
                "TF": ["^TF", "^N_TF", "^Tem_"],
            },
            "exclusion_path": (
                str(resolved_exclusion) if resolved_exclusion is not None else None
            ),
            "exclusion_sha256": (
                sha256_file(resolved_exclusion)
                if resolved_exclusion is not None
                else None
            ),
            "feature_order": list(prepared.feature_names),
            "feature_order_sha256": _feature_order_sha256(prepared.feature_names),
            "structural_null": {
                "strategy": "fill_with_constant",
                "fill_value": 0.0,
                "semantic_meaning": "no measurable change",
            },
            "scaler": {
                "class": "sklearn.preprocessing.StandardScaler",
                "parameters": {
                    name: _json_scalar(value)
                    for name, value in prepared.scaler.get_params(deep=False).items()
                },
                "statistics": _dataframe_records_for_json(
                    prepared.scaler_parameters
                ),
            },
            "matrix_sha256": sha256_file(
                temporary_directory / "features_standardized.csv"
            ),
            "artifact_sha256": {
                name: sha256_file(temporary_directory / name)
                for name in sorted(SERIALIZED_ARTIFACT_FILENAMES)
            },
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }
        (temporary_directory / "preprocessing_config.json").write_text(
            json.dumps(config, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if final_directory.exists():
            raise FileExistsError(
                f"Prepared artifact is immutable and already exists: {final_directory}"
            )
        try:
            temporary_directory.rename(final_directory)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Prepared artifact is immutable and already exists: {final_directory}"
            ) from exc
        temporary_directory = None
        return final_directory
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory)


def _load_artifact_config(directory: Path) -> dict[str, object]:
    try:
        config = json.loads(
            (directory / "preprocessing_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Prepared artifact config is unreadable") from exc
    if not isinstance(config, dict):
        raise ValueError("Prepared artifact config must be a JSON object")
    required_keys = {
        "source_path",
        "source_sha256",
        "source_row_count",
        "scenario",
        "risk_filter",
        "marker_rules",
        "exclusion_path",
        "exclusion_sha256",
        "feature_order",
        "feature_order_sha256",
        "structural_null",
        "scaler",
        "matrix_sha256",
        "artifact_sha256",
        "versions",
        "created_at_utc",
    }
    missing_keys = required_keys.difference(config)
    if missing_keys:
        raise ValueError(
            "Prepared artifact config is missing keys: "
            + ", ".join(sorted(missing_keys))
        )
    return config


def _validate_serialized_artifact_hashes(
    directory: Path, config: dict[str, object]
) -> None:
    """Verify every serialized CSV before trusting any of its parsed contents."""
    recorded_hashes = config["artifact_sha256"]
    expected_names = set(SERIALIZED_ARTIFACT_FILENAMES)
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != expected_names:
        raise ValueError("Prepared artifact serialized-file SHA-256 manifest is invalid")
    for filename in sorted(SERIALIZED_ARTIFACT_FILENAMES):
        recorded_hash = recorded_hashes[filename]
        if not isinstance(recorded_hash, str) or sha256_file(directory / filename) != recorded_hash:
            raise ValueError(
                f"Prepared artifact SHA-256 differs for serialized file: {filename}"
            )
    if config["matrix_sha256"] != recorded_hashes["features_standardized.csv"]:
        raise ValueError("Prepared artifact matrix SHA-256 differs from serialized-file hash")


def _validate_structural_zero_contract(
    original_features: pd.DataFrame,
    filled_features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    """Check persisted fill-zero values and their row-level semantic audit."""
    try:
        expected_filled = original_features.fillna(0.0).to_numpy(dtype="float64")
        actual_filled = filled_features.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Prepared artifact original and filled features must be numeric") from exc
    if not np.isfinite(actual_filled).all() or not np.allclose(
        actual_filled, expected_filled, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Prepared artifact filled features differ from zero-fill contract")
    expected_null_counts = original_features.isna().sum(axis=1).astype("int64")
    expected_had_null = expected_null_counts.gt(0)
    if (
        "structural_null_count" not in metadata.columns
        or "had_structural_null" not in metadata.columns
    ):
        raise ValueError("Prepared artifact metadata is missing structural-null audit")
    if not np.array_equal(
        metadata["structural_null_count"].to_numpy(),
        expected_null_counts.to_numpy(),
    ) or not np.array_equal(
        metadata["had_structural_null"].to_numpy(dtype=bool),
        expected_had_null.to_numpy(dtype=bool),
    ):
        raise ValueError("Prepared artifact structural-null audit differs from original features")


def _validate_standardization_contract(
    original_features: pd.DataFrame,
    filled_features: pd.DataFrame,
    standardized_features: pd.DataFrame,
    scaler_parameters: pd.DataFrame,
) -> None:
    """Check the persisted StandardScaler relationship without refitting it."""
    expected_columns = {
        "feature",
        "mean",
        "variance",
        "scale",
        "original_null_count",
        "original_min",
        "original_max",
        "filled_min",
        "filled_max",
    }
    if set(scaler_parameters.columns) != expected_columns:
        raise ValueError("Prepared artifact scaler parameters schema is invalid")
    try:
        filled_values = filled_features.to_numpy(dtype="float64")
        original_values = original_features.to_numpy(dtype="float64")
        standardized_values = standardized_features.to_numpy(dtype="float64")
        means = scaler_parameters["mean"].to_numpy(dtype="float64")
        scales = scaler_parameters["scale"].to_numpy(dtype="float64")
        variances = scaler_parameters["variance"].to_numpy(dtype="float64")
        original_null_counts = scaler_parameters["original_null_count"].to_numpy(
            dtype="float64"
        )
        original_mins = scaler_parameters["original_min"].to_numpy(dtype="float64")
        original_maxs = scaler_parameters["original_max"].to_numpy(dtype="float64")
        filled_mins = scaler_parameters["filled_min"].to_numpy(dtype="float64")
        filled_maxs = scaler_parameters["filled_max"].to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Prepared artifact scaler parameters are non-numeric") from exc
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(variances).all()
        or not np.isfinite(original_null_counts).all()
        or not np.isfinite(original_mins).all()
        or not np.isfinite(original_maxs).all()
        or not np.isfinite(filled_mins).all()
        or not np.isfinite(filled_maxs).all()
        or np.any(scales <= 0)
        or np.any(variances < 0)
    ):
        raise ValueError("Prepared artifact scaler parameters are invalid")
    calculated_means = filled_values.mean(axis=0)
    calculated_variances = filled_values.var(axis=0, ddof=0)
    # Match StandardScaler's numerical-constant rule. A fixed ``isclose``
    # threshold would incorrectly collapse legitimate very-small-variance
    # features to scale 1.
    sample_count = len(filled_values)
    epsilon = np.finfo(np.float64).eps
    constant_upper_bounds = (
        sample_count * epsilon * calculated_variances
        + (sample_count * calculated_means * epsilon) ** 2
    )
    constant_mask = calculated_variances <= constant_upper_bounds
    calculated_scales = np.sqrt(calculated_variances)
    calculated_scales[constant_mask] = 1.0
    if not (
        np.allclose(means, calculated_means, atol=1e-12, rtol=1e-12)
        and np.allclose(variances, calculated_variances, atol=1e-12, rtol=1e-12)
        and np.allclose(scales, calculated_scales, atol=1e-12, rtol=1e-12)
        and np.allclose(
            original_null_counts,
            np.isnan(original_values).sum(axis=0),
            atol=0.0,
            rtol=0.0,
        )
        and np.allclose(
            original_mins,
            np.nanmin(original_values, axis=0),
            atol=1e-12,
            rtol=1e-12,
        )
        and np.allclose(
            original_maxs,
            np.nanmax(original_values, axis=0),
            atol=1e-12,
            rtol=1e-12,
        )
        and np.allclose(
            filled_mins, filled_values.min(axis=0), atol=1e-12, rtol=1e-12
        )
        and np.allclose(
            filled_maxs, filled_values.max(axis=0), atol=1e-12, rtol=1e-12
        )
    ):
        raise ValueError("Prepared artifact scaler parameters do not match feature values")
    expected_standardized = (filled_values - means) / scales
    if not np.allclose(
        standardized_values, expected_standardized, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Prepared artifact standardized features differ from scaler parameters")


def _validate_source_alignment(
    config: dict[str, object],
    feature_names: tuple[str, ...],
    metadata: pd.DataFrame,
    original_features: pd.DataFrame,
) -> None:
    """Prove serialized row keys and original values match the recorded source."""
    scenario = str(config["scenario"])
    source_path = Path(str(config["source_path"]))
    try:
        source = pd.read_csv(source_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError("Prepared artifact recorded source is unreadable") from exc

    risk_column = f"{scenario}_Index_Risk_equal"
    source_metadata_columns = ("fid", "MS_ID", risk_column)
    serialized_metadata_columns = source_metadata_columns + (
        "source_row_position",
        "structural_null_count",
        "had_structural_null",
    )
    if tuple(metadata.columns) != serialized_metadata_columns:
        raise ValueError("Prepared artifact metadata schema differs from recorded source")
    required_source_columns = set(source_metadata_columns).union(feature_names)
    missing_source_columns = required_source_columns.difference(source.columns)
    if missing_source_columns:
        raise ValueError(
            "Prepared artifact recorded source is missing columns: "
            + ", ".join(sorted(missing_source_columns))
        )
    if len(source) != len(metadata):
        raise ValueError("Prepared artifact row count differs from recorded source")

    expected_metadata = source.loc[:, source_metadata_columns].reset_index(drop=True)
    actual_metadata = metadata.loc[:, source_metadata_columns].reset_index(drop=True)
    expected_original = source.loc[:, feature_names].reset_index(drop=True)
    actual_original = original_features.reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual_metadata,
            expected_metadata,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        pd.testing.assert_frame_equal(
            actual_original,
            expected_original,
            check_dtype=False,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "Prepared artifact rows or original features differ from recorded source"
        ) from exc


def _validate_recorded_file_hash(
    path_value: object,
    hash_value: object,
    label: str,
) -> None:
    if not isinstance(path_value, str) or not isinstance(hash_value, str):
        raise ValueError(f"Prepared artifact {label} provenance is invalid")
    provenance_path = Path(path_value)
    if not provenance_path.is_file():
        raise ValueError(f"Prepared artifact {label} provenance file is missing")
    if sha256_file(provenance_path) != hash_value:
        raise ValueError(f"Prepared artifact {label} SHA-256 differs")


def load_prepared_artifact(path: Path) -> PreparedArtifact:
    """Load a prepared artifact only after strict order and integrity checks."""
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise ValueError(f"Prepared artifact directory is missing: {directory}")
    missing_files = sorted(
        name for name in ARTIFACT_FILENAMES if not (directory / name).is_file()
    )
    if missing_files:
        raise ValueError(
            "Prepared artifact is missing required files: "
            + ", ".join(missing_files)
        )

    config = _load_artifact_config(directory)
    _validate_serialized_artifact_hashes(directory, config)
    scenario = config["scenario"]
    if scenario not in {"PF", "TF"}:
        raise ValueError("Prepared artifact scenario must be PF or TF")
    configured_order = config["feature_order"]
    if (
        not isinstance(configured_order, list)
        or not configured_order
        or not all(isinstance(name, str) for name in configured_order)
        or len(configured_order) != len(set(configured_order))
    ):
        raise ValueError("Prepared artifact feature order is invalid")
    feature_names = tuple(configured_order)
    if _feature_order_sha256(feature_names) != config["feature_order_sha256"]:
        raise ValueError("Prepared artifact feature-order SHA-256 differs")

    selected_features = pd.read_csv(directory / "selected_features.csv")
    if tuple(selected_features.columns) != ("feature",):
        raise ValueError("Prepared artifact selected-feature schema is invalid")
    if tuple(selected_features["feature"]) != feature_names:
        raise ValueError(
            "Prepared artifact selected-feature order differs from config"
        )

    matrix_path = directory / "features_standardized.csv"

    metadata = pd.read_csv(directory / "metadata.csv")
    original_features = pd.read_csv(directory / "features_original.csv")
    filled_features = pd.read_csv(directory / "features_filled.csv")
    standardized_features = pd.read_csv(matrix_path)
    for matrix in (
        original_features,
        filled_features,
        standardized_features,
    ):
        if tuple(matrix.columns) != feature_names:
            raise ValueError(
                "Prepared artifact feature CSV column order differs from config"
            )
    row_count = len(metadata)
    if any(
        len(matrix) != row_count
        for matrix in (
            original_features,
            filled_features,
            standardized_features,
        )
    ):
        raise ValueError(
            "Prepared artifact metadata and feature matrix row counts differ"
        )
    if config["source_row_count"] != row_count:
        raise ValueError(
            "Prepared artifact source and serialized row counts differ"
        )
    if "source_row_position" not in metadata.columns or not np.array_equal(
        metadata.get("source_row_position", pd.Series(dtype="int64")).to_numpy(),
        np.arange(row_count, dtype=np.int64),
    ):
        raise ValueError(
            "Prepared artifact source-row positions are not contiguous 0..n-1"
        )
    _validate_structural_zero_contract(original_features, filled_features, metadata)
    try:
        standardized_values = standardized_features.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Prepared artifact standardized values are non-finite or non-numeric"
        ) from exc
    if not np.isfinite(standardized_values).all():
        raise ValueError("Prepared artifact standardized values are non-finite")

    excluded_features = pd.read_csv(directory / "excluded_features.csv")
    scaler_parameters = pd.read_csv(directory / "scaler_parameters.csv")
    audit = pd.read_csv(directory / "preprocessing_audit.csv")
    if "feature" not in scaler_parameters.columns or tuple(
        scaler_parameters["feature"]
    ) != feature_names:
        raise ValueError(
            "Prepared artifact scaler feature order differs from config"
        )
    _validate_standardization_contract(
        original_features, filled_features, standardized_features, scaler_parameters
    )

    _validate_recorded_file_hash(
        config["source_path"], config["source_sha256"], "source"
    )
    _validate_source_alignment(
        config, feature_names, metadata, original_features
    )
    exclusion_path = config["exclusion_path"]
    exclusion_hash = config["exclusion_sha256"]
    if exclusion_path is None and exclusion_hash is None:
        pass
    elif exclusion_path is None or exclusion_hash is None:
        raise ValueError("Prepared artifact exclusion provenance is invalid")
    else:
        _validate_recorded_file_hash(
            exclusion_path, exclusion_hash, "exclusion"
        )

    return PreparedArtifact(
        directory=directory,
        config=config,
        metadata=metadata,
        original_features=original_features,
        filled_features=filled_features,
        standardized_features=standardized_features,
        feature_names=feature_names,
        excluded_features=excluded_features,
        scaler_parameters=scaler_parameters,
        audit=audit,
    )
