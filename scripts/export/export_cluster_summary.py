#!/usr/bin/env python3
"""Export cluster summaries and noise analysis from saved labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.config import INPUT_CSV, PIPELINE_ROOT  # noqa: E402
from utils.io_utils import (
    ensure_pipeline_dirs,
    prompt_input_path,
    resolve_dataset_output_root,
    resolve_input_path,
)  # noqa: E402
from utils.result_utils import discover_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export modular HDBSCAN summaries.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = (
        resolve_input_path(args.input)
        if args.input is not None
        else prompt_input_path(INPUT_CSV)
    )
    output_root = resolve_dataset_output_root(input_path, args.pipeline_root.resolve())
    ensure_pipeline_dirs(output_root)
    metadata_files = discover_metadata(output_root)
    if not metadata_files:
        raise SystemExit("ERROR: no modular cluster metadata files were found.")
    cluster_rows = []
    noise_rows = []
    parameter_rows = []
    for metadata_path, metadata in metadata_files:
        labels = pd.read_csv(metadata_path.parent / "cluster_labels.csv")
        counts = labels["cluster_id"].value_counts().sort_index()
        for cluster_id, count in counts.items():
            cluster_rows.append(
                {
                    "scenario": metadata["scenario"],
                    "version": metadata["version"],
                    "min_cluster_size": metadata["min_cluster_size"],
                    "min_samples": metadata["min_samples"],
                    "cluster_id": cluster_id,
                    "count": count,
                    "percentage": count / len(labels) * 100,
                }
            )
        noise_rows.append(
            {
                "scenario": metadata["scenario"],
                "version": metadata["version"],
                "min_cluster_size": metadata["min_cluster_size"],
                "min_samples": metadata["min_samples"],
                "noise_count": metadata["noise_count"],
                "noise_ratio": metadata["noise_ratio"],
            }
        )
        parameter_rows.append(
            {
                "scenario": metadata["scenario"],
                "version": metadata["version"],
                "min_cluster_size": metadata["min_cluster_size"],
                "min_samples": metadata["min_samples"],
                "n_clusters": metadata["n_clusters"],
                "noise_count": metadata["noise_count"],
                "noise_ratio": metadata["noise_ratio"],
                "largest_cluster_size": metadata["largest_cluster_size"],
                "runtime_seconds": metadata["runtime_seconds"],
            }
        )
    summaries = output_root / "tables" / "summaries"
    noise = output_root / "tables" / "noise_analysis"
    pd.DataFrame(cluster_rows).to_csv(summaries / "cluster_summary.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        summaries / "parameter_summary.csv", index=False
    )
    pd.DataFrame(noise_rows).to_csv(noise / "noise_analysis.csv", index=False)
    print(f"Saved summaries: {summaries}")
    print(f"Saved noise analysis: {noise}")


if __name__ == "__main__":
    main()
