#!/usr/bin/env python3
"""Export label comparisons and conflict samples from saved clustering results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.config import INPUT_CSV, PIPELINE_ROOT, PROJECT_ROOT  # noqa: E402
from utils.io_utils import (
    ensure_pipeline_dirs,
    prompt_input_path,
    resolve_dataset_output_root,
    resolve_input_path,
)  # noqa: E402
from utils.result_utils import discover_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PF/TF and Version A/B label comparisons."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--pipeline-root", type=Path, default=PIPELINE_ROOT)
    parser.add_argument(
        "--rule-label-column",
        default="cause_label",
        help="Optional rule-based label used only for post-clustering comparison.",
    )
    parser.add_argument(
        "--pf-rule-label-input",
        type=Path,
        default=PROJECT_ROOT / "data" / "toGIS_pf_ms.csv",
    )
    parser.add_argument(
        "--tf-rule-label-input",
        type=Path,
        default=PROJECT_ROOT / "data" / "toGIS_tf_ms.csv",
    )
    return parser.parse_args()


def load_runs(output_root: Path) -> dict[tuple[str, str, int, int], pd.DataFrame]:
    runs = {}
    for metadata_path, metadata in discover_metadata(output_root):
        key = (
            metadata["scenario"],
            metadata["version"],
            metadata["min_cluster_size"],
            metadata["min_samples"],
        )
        runs[key] = pd.read_csv(metadata_path.parent / "cluster_labels.csv")
    return runs


def compare_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    output_path: Path,
) -> None:
    merged = left[["street_id", "cluster_id"]].merge(
        right[["street_id", "cluster_id"]],
        on="street_id",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        validate="one_to_one",
    )
    left_cluster = f"cluster_id_{left_name}"
    right_cluster = f"cluster_id_{right_name}"
    merged["is_conflict"] = merged[left_cluster] != merged[right_cluster]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    merged[merged["is_conflict"]].to_csv(
        output_path.with_name(f"{output_path.stem}_conflicts.csv"), index=False
    )


def load_rule_labels(path: Path, label_column: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Rule-label comparison skipped; file not found: {path}")
        return None
    labels = pd.read_csv(path)
    required = {"MS_ID", label_column}
    if not required.issubset(labels.columns):
        print(
            "Rule-label comparison skipped; missing columns "
            f"{sorted(required - set(labels.columns))} in {path}"
        )
        return None
    return labels[["MS_ID", label_column]].drop_duplicates(subset=["MS_ID"])


def export_rule_label_comparison(
    cluster_labels: pd.DataFrame,
    rule_labels: pd.DataFrame,
    label_column: str,
    output_dir: Path,
) -> None:
    merged = cluster_labels.merge(rule_labels, on="MS_ID", how="left")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.crosstab(
        merged[label_column], merged["cluster_id"], dropna=False
    ).to_csv(output_dir / "rule_label_cluster_crosstab.csv")
    non_noise = merged[merged["cluster_id"] != -1].copy()
    majority = (
        non_noise.dropna(subset=[label_column])
        .groupby("cluster_id")[label_column]
        .agg(lambda values: values.value_counts().index[0])
    )
    non_noise["cluster_majority_rule_type"] = non_noise["cluster_id"].map(majority)
    conflicts = non_noise[
        non_noise[label_column].notna()
        & non_noise["cluster_majority_rule_type"].notna()
        & (non_noise[label_column] != non_noise["cluster_majority_rule_type"])
    ]
    conflicts.to_csv(output_dir / "rule_label_conflict_samples.csv", index=False)


def main() -> None:
    args = parse_args()
    input_path = (
        resolve_input_path(args.input)
        if args.input is not None
        else prompt_input_path(INPUT_CSV)
    )
    output_root = resolve_dataset_output_root(input_path, args.pipeline_root.resolve())
    ensure_pipeline_dirs(output_root)
    runs = load_runs(output_root)
    if not runs:
        raise SystemExit("ERROR: no modular cluster label files were found.")
    comparisons = output_root / "tables" / "comparisons"
    rule_labels_by_scenario = {
        "PF": load_rule_labels(args.pf_rule_label_input.resolve(), args.rule_label_column),
        "TF": load_rule_labels(args.tf_rule_label_input.resolve(), args.rule_label_column),
    }

    for (scenario, version, mcs, samples), labels in runs.items():
        rule_labels = rule_labels_by_scenario[scenario]
        if rule_labels is not None:
            export_rule_label_comparison(
                labels,
                rule_labels,
                args.rule_label_column,
                comparisons
                / "rule_based"
                / scenario
                / version
                / f"mcs_{mcs}_ms_{samples}",
            )

    parameter_sets = sorted({(key[2], key[3]) for key in runs})
    for scenario in ("PF", "TF"):
        for mcs, samples in parameter_sets:
            a = runs.get((scenario, "version_a", mcs, samples))
            b = runs.get((scenario, "version_b", mcs, samples))
            if a is not None and b is not None:
                compare_pair(
                    a,
                    b,
                    "version_a",
                    "version_b",
                    comparisons
                    / "version_a_vs_b"
                    / scenario
                    / f"mcs_{mcs}_ms_{samples}.csv",
                )

    for version in ("version_a", "version_b"):
        for mcs, samples in parameter_sets:
            pf = runs.get(("PF", version, mcs, samples))
            tf = runs.get(("TF", version, mcs, samples))
            if pf is not None and tf is not None:
                compare_pair(
                    pf,
                    tf,
                    "PF",
                    "TF",
                    comparisons
                    / "pf_vs_tf"
                    / version
                    / f"mcs_{mcs}_ms_{samples}.csv",
                )
    print(f"Saved comparisons and conflict samples: {comparisons}")
    print(
        "NOTE: cluster IDs are run-specific identifiers. Conflicts indicate "
        "different assigned IDs and should be interpreted with cluster profiles."
    )


if __name__ == "__main__":
    main()
