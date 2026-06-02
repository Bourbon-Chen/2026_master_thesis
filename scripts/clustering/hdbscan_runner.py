"""Shared execution logic for independent Version A and Version B runners."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.config import (  # noqa: E402
    CLUSTER_SELECTION_METHOD,
    CONDA_ENVIRONMENT,
    INPUT_CSV,
    METRIC,
    MISSING_VALUE_STRATEGY,
    PIPELINE_ROOT,
    SCENARIOS,
    dataset_name_from_input,
    tables_root,
)
from utils.feature_utils import prepare_features, save_feature_audit  # noqa: E402
from utils.io_utils import (  # noqa: E402
    build_label_table,
    ensure_pipeline_dirs,
    python_runtime_metadata,
    prompt_input_path,
    read_last_version_a_config,
    resolve_dataset_output_root,
    resolve_input_path,
    sha256_file,
    utc_now_iso,
    write_last_version_a_config,
    write_json,
)
from utils.logging_utils import (  # noqa: E402
    append_runtime_log,
    format_duration,
    print_progress,
)


def parse_args(versions: tuple[str, ...]) -> argparse.Namespace:
    version_text = ", ".join(versions)
    parser = argparse.ArgumentParser(
        description=f"Run modular unsupervised HDBSCAN clustering for {version_text}."
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help="Scenarios to run (default: PF TF).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV. Prompted interactively when omitted.",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=PIPELINE_ROOT,
        help=f"Pipeline root directory (default: {PIPELINE_ROOT}).",
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
        help="Save feature audits without importing sklearn or running HDBSCAN.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute completed runs in pipeline_v1. Historical legacy outputs remain untouched.",
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


def resolve_run_config(
    args: argparse.Namespace, versions: tuple[str, ...]
) -> tuple[Path, Path, list[str], int, int, dict | None]:
    input_path = (
        resolve_input_path(args.input)
        if args.input is not None
        else prompt_input_path(INPUT_CSV)
    )
    pipeline_root = args.pipeline_root.resolve()
    output_root = resolve_dataset_output_root(input_path, pipeline_root)
    if versions == ("version_b",):
        config = read_last_version_a_config(output_root)
        input_path = resolve_input_path(Path(config["input_file"]))
        if resolve_dataset_output_root(input_path, pipeline_root) != output_root:
            raise ValueError("Last Version A config points to a different dataset root.")
        scenarios = config["scenarios"]
        min_cluster_size = int(config["min_cluster_size"])
        min_samples = int(config["min_samples"])
        print("Using last Version A configuration:")
        print(f"dataset_name: {config['dataset_name']}")
        print(f"input_file: {config['input_file']}")
        print(f"scenarios: {' '.join(scenarios)}")
        print(f"min_cluster_size: {min_cluster_size}")
        print(f"min_samples: {min_samples}")
        print(f"metric: {config['metric']}")
        print(f"selection: {config['cluster_selection_method']}")
        return input_path, output_root, scenarios, min_cluster_size, min_samples, config

    scenarios = list(args.scenario)
    if args.audit_only:
        return input_path, output_root, scenarios, 0, 0, None
    min_cluster_size, min_samples = resolve_hdbscan_parameters(args)
    return input_path, output_root, scenarios, min_cluster_size, min_samples, None


def check_conda_environment() -> None:
    active_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if active_environment != CONDA_ENVIRONMENT:
        print(
            f"WARNING: expected Conda environment '{CONDA_ENVIRONMENT}', "
            f"active='{active_environment or '<not detected>'}'."
        )


def import_sklearn(versions: tuple[str, ...]):
    try:
        import sklearn
        from sklearn.cluster import HDBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn with sklearn.cluster.HDBSCAN is required. "
            f"Use the configured Conda environment '{CONDA_ENVIRONMENT}'."
        ) from exc
    return sklearn, HDBSCAN, StandardScaler if "version_b" in versions else None


def save_audit(prepared, scenario: str, version: str, output_root: Path) -> None:
    audit_dir = tables_root(output_root) / "audits" / scenario / version
    save_feature_audit(prepared, audit_dir)
    print(f"{scenario} {version}: selected {len(prepared.feature_names)} features")
    for feature in prepared.feature_names:
        print(f"  - {feature}")
    if prepared.extreme_columns.empty:
        print("Extreme-value audit passed: selected features are within [-1, 1].")
    else:
        print("WARNING: selected features outside [-1, 1] were detected:")
        for row in prepared.extreme_columns.itertuples(index=False):
            print(f"  - {row.feature}: min={row.min:g}, max={row.max:g}")


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


def run_experiments(versions: tuple[str, ...]) -> None:
    args = parse_args(versions)
    check_conda_environment()
    (
        input_path,
        output_root,
        scenarios,
        min_cluster_size,
        min_samples,
        linked_version_a_config,
    ) = resolve_run_config(args, versions)
    print(f"Reading: {input_path}")
    data = pd.read_csv(input_path)
    input_sha256 = sha256_file(input_path)
    ensure_pipeline_dirs(output_root)

    prepared_by_scenario = {}
    for scenario in scenarios:
        prepared = prepare_features(data, scenario)
        prepared_by_scenario[scenario] = prepared
        for version in versions:
            save_audit(prepared, scenario, version, output_root)
    if args.audit_only:
        print("Audit-only mode completed; HDBSCAN was not run.")
        return

    sklearn, HDBSCAN, StandardScaler = import_sklearn(versions)

    if "version_a" in versions:
        config = {
            "dataset_name": dataset_name_from_input(input_path),
            "input_file": str(input_path),
            "input_sha256": input_sha256,
            "scenarios": scenarios,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": METRIC,
            "cluster_selection_method": CLUSTER_SELECTION_METHOD,
            "missing_value_strategy": MISSING_VALUE_STRATEGY,
            "timestamp": utc_now_iso(),
        }
        config_path = write_last_version_a_config(output_root, config)
        linked_version_a_config = config
        print(f"Saved Version A configuration: {config_path}")

    total_runs = len(scenarios) * len(versions)
    print(f"Total Runs: {total_runs}")
    print(f"min_cluster_size: {min_cluster_size}")
    print(f"min_samples: {min_samples}")
    print(f"metric: {METRIC}")
    print(f"cluster_selection_method: {CLUSTER_SELECTION_METHOD}")
    completed_durations: list[float] = []
    experiment_start = time.perf_counter()
    runtime_log = output_root / "logs" / "runtime_log.csv"

    for current, (scenario, version) in enumerate(
        ((scenario, version) for scenario in scenarios for version in versions),
        start=1,
    ):
        experiment_id = (
            f"{scenario}_{version}_mcs_{min_cluster_size}_ms_{min_samples}"
        )
        destination = (
            output_root
            / "clustering_results"
            / scenario
            / version
            / f"min_cluster_size_{min_cluster_size}_min_samples_{min_samples}"
        )
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
        if labels_path.exists() and metadata_path.exists() and not args.overwrite:
            print("Completed output already exists; skipping. Use --overwrite to recompute.")
            continue

        prepared = prepared_by_scenario[scenario]
        clustering_input = prepared.values.to_numpy(dtype=float)
        if version == "version_b":
            clustering_input = StandardScaler().fit_transform(clustering_input)

        start_iso = utc_now_iso()
        run_start = time.perf_counter()
        log_row = {
            "scenario": scenario,
            "version": version,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "start_time": start_iso,
        }
        try:
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=METRIC,
                cluster_selection_method=CLUSTER_SELECTION_METHOD,
            ).fit_predict(clustering_input)
            duration = time.perf_counter() - run_start
            end_iso = utc_now_iso()
            statistics = cluster_statistics(labels)
            destination.mkdir(parents=True, exist_ok=True)
            build_label_table(data, labels, scenario, version).to_csv(
                labels_path, index=False
            )
            statistics["cluster_sizes"].to_csv(
                destination / "cluster_sizes.csv", index=False
            )
            metadata = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_name_from_input(input_path),
                "pipeline_dataset_root": str(output_root),
                "linked_version_a_config": linked_version_a_config,
                "scenario": scenario,
                "version": version,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": METRIC,
                "cluster_selection_method": CLUSTER_SELECTION_METHOD,
                "n_clusters": statistics["n_clusters"],
                "noise_count": statistics["noise_count"],
                "noise_ratio": statistics["noise_ratio"],
                "largest_cluster_size": statistics["largest_cluster_size"],
                "runtime_seconds": duration,
                "start_time": start_iso,
                "end_time": end_iso,
                "input_file": str(input_path),
                "input_sha256": input_sha256,
                "selected_features": prepared.feature_names,
                "missing_value_strategy": MISSING_VALUE_STRATEGY,
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
    try:
        run_experiments(tuple(versions))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
