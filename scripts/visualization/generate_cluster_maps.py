#!/usr/bin/env python3
"""Create GIS join tables from saved labels without rerunning HDBSCAN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.config import INPUT_CSV, PIPELINE_ROOT, SCENARIOS  # noqa: E402
from utils.io_utils import (
    ensure_pipeline_dirs,
    prompt_input_path,
    resolve_dataset_output_root,
    resolve_input_path,
)  # noqa: E402
from utils.result_utils import discover_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export saved HDBSCAN labels as GIS join tables."
    )
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--version", choices=("version_a", "version_b"))
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
    metadata_files = discover_metadata(output_root, args.scenario, args.version)
    if not metadata_files:
        raise SystemExit("ERROR: no modular cluster metadata files were found.")
    for metadata_path, metadata in metadata_files:
        labels = pd.read_csv(metadata_path.parent / "cluster_labels.csv")
        output = (
            output_root
            / "visualizations"
            / "maps"
            / metadata["scenario"]
            / metadata["version"]
            / metadata_path.parent.name
            / "cluster_labels_gis_join.csv"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(output, index=False)
        print(f"Saved GIS join table: {output}")
    print(
        "NOTE: the current CSV data has no geometry. Join these tables to the "
        "street geometry layer in QGIS using MS_ID or fid."
    )


if __name__ == "__main__":
    main()
