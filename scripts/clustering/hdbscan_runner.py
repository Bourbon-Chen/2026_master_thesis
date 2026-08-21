"""Run HDBSCAN against one verified prepared clustering artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.clustering_preprocessing import (  # noqa: E402
    PreparedArtifact,
    load_prepared_artifact,
)
from utils.config import (  # noqa: E402
    CLUSTER_SELECTION_METHOD,
    CONDA_ENVIRONMENT,
    METRIC,
    PIPELINE_ROOT,
    STRUCTURAL_NULL_POLICY,
    dataset_name_from_input,
    tables_root,
)
from utils.io_utils import (  # noqa: E402
    ensure_pipeline_dirs,
    python_runtime_metadata,
    resolve_dataset_output_root,
    sha256_file,
    utc_now_iso,
    write_json,
)
from utils.logging_utils import (  # noqa: E402
    append_runtime_log,
    format_duration,
    print_progress,
)


@dataclass(frozen=True)
class HdbscanInput:
    """The immutable matrix and provenance selected for one HDBSCAN version."""

    artifact: PreparedArtifact
    values: np.ndarray
    scale_label: str


def load_hdbscan_input(path: Path, version: str) -> HdbscanInput:
    """Load a prepared matrix without feature discovery or transformation."""
    artifact = load_prepared_artifact(path)
    if version == "version_b":
        frame = artifact.standardized_features
        label = "standardized_primary"
    elif version == "version_a":
        frame = artifact.filled_features
        label = "filled_raw_scale_sensitivity"
    else:
        raise ValueError(f"Unsupported HDBSCAN version: {version}")
    return HdbscanInput(artifact, frame.to_numpy(dtype=float, copy=True), label)


def parse_args(versions: tuple[str, ...]) -> argparse.Namespace:
    version_text = ", ".join(versions)
    parser = argparse.ArgumentParser(
        description=(
            "Run modular HDBSCAN on a verified prepared artifact for "
            f"{version_text}."
        )
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Directory containing the immutable prepared clustering artifact.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Pipeline dataset output root (default: derived from artifact source).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=positive_int,
        help="HDBSCAN min_cluster_size. Prompted interactively when omitted.",
    )
    parser.add_argument(
        "--min-samples",
        type=positive_int,
        help="HDBSCAN min_samples. Prompted interactively when omitted.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write prepared-artifact audits without importing or fitting HDBSCAN.",
    )
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def prompt_positive_int(label: str) -> int:
    while True:
        try:
            return positive_int(input(f"Enter {label}: ").strip())
        except (ValueError, argparse.ArgumentTypeError):
            print(f"{label} must be a positive integer.")


def resolve_hdbscan_parameters(args: argparse.Namespace) -> tuple[int, int]:
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    if min_cluster_size is None:
        min_cluster_size = prompt_positive_int("min_cluster_size")
    if min_samples is None:
        min_samples = prompt_positive_int("min_samples")
    return min_cluster_size, min_samples


def check_conda_environment() -> None:
    active_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if active_environment != CONDA_ENVIRONMENT:
        print(
            f"WARNING: expected Conda environment '{CONDA_ENVIRONMENT}', "
            f"active='{active_environment or '<not detected>'}'."
        )


def import_sklearn():
    try:
        import sklearn
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn with sklearn.cluster.HDBSCAN is required. "
            f"Use the configured Conda environment '{CONDA_ENVIRONMENT}'."
        ) from exc
    return sklearn, HDBSCAN


def matrix_path_for_input(hdbscan_input: HdbscanInput) -> Path:
    filename = (
        "features_standardized.csv"
        if hdbscan_input.scale_label == "standardized_primary"
        else "features_filled.csv"
    )
    return hdbscan_input.artifact.directory / filename


def preprocessing_reference(hdbscan_input: HdbscanInput) -> dict[str, object]:
    artifact = hdbscan_input.artifact
    manifest_path = artifact.directory / "preprocessing_config.json"
    matrix_path = matrix_path_for_input(hdbscan_input)
    return {
        "prepared_manifest_path": str(manifest_path.resolve()),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": sha256_file(matrix_path),
        "scenario": artifact.config["scenario"],
        "feature_order": list(artifact.feature_names),
        "scale_label": hdbscan_input.scale_label,
        "structural_null_policy": STRUCTURAL_NULL_POLICY,
        "structural_null_row_count": int(
            artifact.metadata["had_structural_null"].sum()
        )
        if "had_structural_null" in artifact.metadata
        else 0,
    }


def save_audit(
    hdbscan_input: HdbscanInput,
    version: str,
    output_root: Path,
    algorithm_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    """Copy the immutable preprocessing audit with the HDBSCAN matrix identity."""
    reference = preprocessing_reference(hdbscan_input)
    if algorithm_parameters is not None:
        reference["algorithm_parameters"] = algorithm_parameters
        parameter_directory = (
            f"min_cluster_size_{algorithm_parameters['min_cluster_size']}"
            f"_min_samples_{algorithm_parameters['min_samples']}"
        )
    else:
        parameter_directory = "audit_only"
    audit_dir = (
        tables_root(output_root)
        / "audits"
        / str(reference["scenario"])
        / version
        / parameter_directory
    )
    if audit_dir.exists():
        raise FileExistsError(
            f"HDBSCAN audit directory already exists: {audit_dir}"
        )
    audit_dir.mkdir(parents=True, exist_ok=False)
    audit = hdbscan_input.artifact.audit.copy()
    if audit.empty:
        audit = pd.DataFrame([{}])
    for key, value in reference.items():
        if isinstance(value, list):
            audit[key] = ",".join(value)
        elif isinstance(value, dict):
            audit[key] = json.dumps(value, sort_keys=True)
        else:
            audit[key] = value
    audit.to_csv(audit_dir / "preprocessing_audit.csv", index=False)
    write_json(audit_dir / "preprocessing_identity.json", reference)
    return reference


def cluster_statistics(labels: np.ndarray) -> dict[str, object]:
    cluster_sizes = (
        pd.Series(labels, name="cluster_id")
        .value_counts()
        .rename_axis("cluster_id")
        .reset_index(name="count")
        .sort_values("cluster_id")
    )
    non_noise = cluster_sizes[cluster_sizes["cluster_id"] != -1]
    noise_count = int((labels == -1).sum())
    return {
        "cluster_sizes": cluster_sizes,
        "n_clusters": int(len(set(labels)) - (-1 in labels)),
        "noise_count": noise_count,
        "noise_ratio": noise_count / len(labels),
        "largest_cluster_size": int(non_noise["count"].max())
        if not non_noise.empty
        else 0,
    }


def build_prepared_label_table(
    artifact: PreparedArtifact, labels: np.ndarray, scenario: str, version: str
) -> pd.DataFrame:
    """Join labels to the preserved metadata row order, never a re-read source CSV."""
    if len(labels) != len(artifact.metadata):
        raise ValueError("HDBSCAN label count does not match prepared metadata rows.")
    metadata = artifact.metadata.reset_index(drop=True)
    if "MS_ID" in metadata.columns:
        street_id = metadata["MS_ID"]
    elif "fid" in metadata.columns:
        street_id = metadata["fid"]
    else:
        street_id = metadata["source_row_position"]
    output = pd.DataFrame({"street_id": street_id})
    for column in ("source_row_position", "fid", "MS_ID"):
        if column in metadata.columns:
            output[column] = metadata[column]
    output["scenario"] = scenario
    output["version"] = version
    output["cluster_id"] = labels
    return output


def default_output_root(artifact: PreparedArtifact) -> Path:
    source_path = Path(str(artifact.config["source_path"]))
    return resolve_dataset_output_root(source_path, PIPELINE_ROOT)


def run_destination(
    output_root: Path,
    scenario: str,
    version: str,
    min_cluster_size: int,
    min_samples: int,
) -> Path:
    """Return the immutable final directory for one concrete HDBSCAN run."""
    return (
        output_root
        / "clustering_results"
        / scenario
        / version
        / f"min_cluster_size_{min_cluster_size}_min_samples_{min_samples}"
    )


def run_experiments(versions: tuple[str, ...]) -> None:
    args = parse_args(versions)
    check_conda_environment()
    if not versions:
        raise ValueError("At least one HDBSCAN version is required.")
    inputs = {version: load_hdbscan_input(args.prepared, version) for version in versions}
    scenarios = {str(item.artifact.config["scenario"]) for item in inputs.values()}
    if len(scenarios) != 1:
        raise ValueError("All HDBSCAN versions must use the same prepared scenario.")
    scenario = scenarios.pop()
    artifact = next(iter(inputs.values())).artifact
    output_root = (
        args.output_root.resolve() if args.output_root is not None else default_output_root(artifact)
    )
    ensure_pipeline_dirs(output_root)
    if args.audit_only:
        for version, hdbscan_input in inputs.items():
            save_audit(hdbscan_input, version, output_root)
        print("Audit-only mode completed; HDBSCAN was not run.")
        return

    min_cluster_size, min_samples = resolve_hdbscan_parameters(args)
    algorithm_parameters = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": METRIC,
        "cluster_selection_method": CLUSTER_SELECTION_METHOD,
    }
    destinations = {
        version: run_destination(
            output_root, scenario, version, min_cluster_size, min_samples
        )
        for version in versions
    }
    existing_destinations = [
        destination for destination in destinations.values() if destination.exists()
    ]
    if existing_destinations:
        raise FileExistsError(
            "HDBSCAN final output directory already exists: "
            + ", ".join(str(destination) for destination in existing_destinations)
        )
    references = {
        version: save_audit(
            hdbscan_input, version, output_root, algorithm_parameters
        )
        for version, hdbscan_input in inputs.items()
    }
    sklearn, HDBSCAN = import_sklearn()
    total_runs = len(versions)
    print(f"Prepared artifact: {artifact.directory}")
    print(f"Scenario: {scenario}")
    print(f"Total Runs: {total_runs}")
    print(f"min_cluster_size: {min_cluster_size}")
    print(f"min_samples: {min_samples}")
    print(f"metric: {METRIC}")
    print(f"cluster_selection_method: {CLUSTER_SELECTION_METHOD}")
    completed_durations: list[float] = []
    experiment_start = time.perf_counter()
    runtime_log = output_root / "logs" / "runtime_log.csv"

    for current, version in enumerate(versions, start=1):
        hdbscan_input = inputs[version]
        reference = references[version]
        experiment_id = f"{scenario}_{version}_mcs_{min_cluster_size}_ms_{min_samples}"
        destination = destinations[version]
        labels_path = destination / "cluster_labels.csv"
        metadata_path = destination / "cluster_metadata.json"
        print_progress(
            current,
            total_runs,
            scenario,
            version,
            min_cluster_size,
            min_samples,
            time.perf_counter() - experiment_start,
            completed_durations,
        )
        start_iso = utc_now_iso()
        run_start = time.perf_counter()
        log_row = {
            "scenario": scenario,
            "version": version,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "scale_label": hdbscan_input.scale_label,
            "start_time": start_iso,
        }
        try:
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=METRIC,
                cluster_selection_method=CLUSTER_SELECTION_METHOD,
            ).fit_predict(hdbscan_input.values)
            duration = time.perf_counter() - run_start
            end_iso = utc_now_iso()
            statistics = cluster_statistics(labels)
            destination.mkdir(parents=True, exist_ok=True)
            build_prepared_label_table(artifact, labels, scenario, version).to_csv(
                labels_path, index=False
            )
            statistics["cluster_sizes"].to_csv(
                destination / "cluster_sizes.csv", index=False
            )
            audit = artifact.audit.copy()
            if audit.empty:
                audit = pd.DataFrame([{}])
            for key, value in reference.items():
                if isinstance(value, list):
                    audit[key] = ",".join(value)
                elif isinstance(value, dict):
                    audit[key] = json.dumps(value, sort_keys=True)
                else:
                    audit[key] = value
            audit.to_csv(destination / "preprocessing_audit.csv", index=False)
            metadata = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name_from_input(Path(str(artifact.config["source_path"]))),
                "pipeline_dataset_root": str(output_root),
                "version": version,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": METRIC,
                "cluster_selection_method": CLUSTER_SELECTION_METHOD,
                "algorithm_parameters": algorithm_parameters,
                **reference,
                "n_clusters": statistics["n_clusters"],
                "noise_count": statistics["noise_count"],
                "noise_ratio": statistics["noise_ratio"],
                "largest_cluster_size": statistics["largest_cluster_size"],
                "runtime_seconds": duration,
                "start_time": start_iso,
                "end_time": end_iso,
                "scikit_learn_version": sklearn.__version__,
                **python_runtime_metadata(),
            }
            write_json(metadata_path, metadata)
            log_row.update(
                {
                    "end_time": end_iso,
                    "duration_seconds": duration,
                    "n_clusters": statistics["n_clusters"],
                    "noise_count": statistics["noise_count"],
                    "noise_ratio": statistics["noise_ratio"],
                    "status": "completed",
                }
            )
            append_runtime_log(runtime_log, log_row)
            completed_durations.append(duration)
            print("Completed:")
            print(f"{scenario} {version}")
            print(f"mcs={min_cluster_size} ms={min_samples}")
            print(f"Runtime: {format_duration(duration)}")
            print(f"Clusters: {statistics['n_clusters']}")
            print(f"Noise Ratio: {statistics['noise_ratio'] * 100:.1f}%")
        except Exception as exc:
            duration = time.perf_counter() - run_start
            log_row.update(
                {
                    "end_time": utc_now_iso(),
                    "duration_seconds": duration,
                    "status": "failed",
                    "error_message": str(exc),
                }
            )
            append_runtime_log(runtime_log, log_row)
            raise

    print(f"Total Elapsed Time: {format_duration(time.perf_counter() - experiment_start)}")


def main(*versions: str) -> None:
    selected_versions = tuple(versions) or ("version_a", "version_b")
    try:
        run_experiments(selected_versions)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
