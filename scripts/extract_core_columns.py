#!/usr/bin/env python3
"""Extract core PF/TF indicator columns from a dataset in the data directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

CORE_COLUMNS = (
    "fid",
    "MS_ID",
    "PF_Index_population",
    "PF_Index_acc_loss",
    "PF_Index_centrality",
    "Per_extent",
    "TF_Index_population",
    "TF_Index_acc_loss",
    "TF_Index_centrality",
    "Tem_extent",
    "PF_Index_Risk_equal",
    "TF_Index_Risk_equal",
)


def resolve_data_path(value: str | Path) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        input_path = raw_path
    else:
        parts = raw_path.parts
        input_path = PROJECT_ROOT / raw_path if parts and parts[0] == "data" else DATA_DIR / raw_path

    resolved = input_path.resolve()
    data_root = DATA_DIR.resolve()
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"Input file must be inside the data directory: {DATA_DIR}")
    if not resolved.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Input path is not a file: {resolved}")
    return resolved


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_core_columns{input_path.suffix}")


def prompt_input_path() -> Path:
    while True:
        value = input("Enter dataset file name/path under data/: ").strip()
        if not value:
            print("Dataset file name/path cannot be empty.")
            continue
        try:
            return resolve_data_path(value)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)


def extract_core_columns(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_path}")

        missing_columns = [
            column for column in CORE_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "Missing required core columns: " + ", ".join(missing_columns)
            )

        with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=CORE_COLUMNS)
            writer.writeheader()

            row_count = 0
            for row in reader:
                writer.writerow({column: row[column] for column in CORE_COLUMNS})
                row_count += 1

    print(f"Input CSV: {input_path}")
    print(f"Output CSV: {output_path}")
    print(f"Rows written: {row_count}")
    print(f"Columns written: {len(CORE_COLUMNS)}")
    print("Core columns:")
    for column in CORE_COLUMNS:
        print(f"  - {column}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract core PF/TF indicator columns from a CSV in data/."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input CSV file name/path under data/. Prompted when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input stem>_core_columns.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_data_path(args.input) if args.input else prompt_input_path()
    output_path = (
        args.output.resolve() if args.output is not None else default_output_path(input_path)
    )
    extract_core_columns(input_path, output_path.resolve())


if __name__ == "__main__":
    main()
