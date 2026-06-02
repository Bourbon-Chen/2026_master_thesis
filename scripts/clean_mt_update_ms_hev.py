#!/usr/bin/env python3
"""Keep the requested indicator columns from the MT_UPDATE_MS_HEV dataset."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "../data/MT_UPDATE_MS_HEV.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "../data/MT_UPDATE_MS_HEV_cleaned.csv"
BASE_COLUMNS = ("fid", "MS_ID", "Per_extent", "Tem_extent")


def is_selected(column: str) -> bool:
    """Return whether a source column belongs in the cleaned indicator dataset."""
    return (
        column in BASE_COLUMNS
        or column.endswith("_NOR")
        or column.startswith("N_")
        or column.endswith("_lossR")
        or "Index" in column
    )


def clean_csv(
    input_path: Path, output_path: Path, normalized_output_path: Path
) -> None:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_path}")

        missing_base_columns = [
            column for column in BASE_COLUMNS if column not in reader.fieldnames
        ]
        if missing_base_columns:
            raise ValueError(
                "Missing required base columns: " + ", ".join(missing_base_columns)
            )

        selected_columns = [
            column for column in reader.fieldnames if is_selected(column)
        ]
        indicator_columns = [
            column for column in selected_columns if column not in BASE_COLUMNS
        ]
        value_ranges: dict[str, list[float | None]] = {
            column: [None, None] for column in indicator_columns
        }
        normalized_value_ranges: dict[str, list[float | None]] = {
            column: [None, None] for column in indicator_columns
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as destination:
            with normalized_output_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as normalized_destination:
                writer = csv.DictWriter(destination, fieldnames=selected_columns)
                normalized_writer = csv.DictWriter(
                    normalized_destination, fieldnames=selected_columns
                )
                writer.writeheader()
                normalized_writer.writeheader()
                row_count = 0
                replacement_count = 0
                for row in reader:
                    writer.writerow(
                        {column: row[column] for column in selected_columns}
                    )
                    update_value_ranges(row, value_ranges)
                    normalized_row, replacements = normalize_row(
                        row, indicator_columns
                    )
                    update_value_ranges(normalized_row, normalized_value_ranges)
                    normalized_writer.writerow(
                        {
                            column: normalized_row[column]
                            for column in selected_columns
                        }
                    )
                    replacement_count += replacements
                    row_count += 1

    outside_unit_interval = [
        (column, minimum, maximum)
        for column, (minimum, maximum) in value_ranges.items()
        if minimum is not None
        and maximum is not None
        and (minimum < 0 or maximum > 1)
    ]
    normalized_outside_supported_interval = [
        (column, minimum, maximum)
        for column, (minimum, maximum) in normalized_value_ranges.items()
        if minimum is not None
        and maximum is not None
        and (minimum < -1 or maximum > 1)
    ]

    print(f"Cleaned CSV written to: {output_path}")
    print(f"Normalized cleaned CSV written to: {normalized_output_path}")
    print(f"Rows written: {row_count}")
    print(f"Normalized indicator values adjusted: {replacement_count}")
    print(
        f"Columns retained: {len(selected_columns)} "
        f"({len(BASE_COLUMNS)} base columns and {len(indicator_columns)} indicators)"
    )
    print("Retained indicator columns:")
    for column in indicator_columns:
        print(f"  - {column}")
    if outside_unit_interval:
        print("WARNING: the original cleaned indicators contain values outside [0, 1]:")
        for column, minimum, maximum in outside_unit_interval:
            print(f"  - {column}: min={minimum:g}, max={maximum:g}")
    else:
        print("Original cleaned audit passed: all indicators are within [0, 1].")
    if normalized_outside_supported_interval:
        raise AssertionError(
            "Normalized output still contains values outside [-1, 1]: "
            + ", ".join(
                column for column, _, _ in normalized_outside_supported_interval
            )
        )
    print("Normalized cleaned audit passed: all indicators are within [-1, 1].")


def normalize_row(
    row: dict[str, str],
    indicator_columns: list[str],
) -> tuple[dict[str, str], int]:
    """Preserve signed values and cap _lossR indicator values above 1."""
    normalized_row = row.copy()
    replacement_count = 0
    for column in indicator_columns:
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if column.endswith("_lossR") and value > 1:
            normalized_row[column] = "1"
            replacement_count += 1
    return normalized_row, replacement_count


def update_value_ranges(
    row: dict[str, str], value_ranges: dict[str, list[float | None]]
) -> None:
    """Track numeric minima and maxima without loading the full CSV into memory."""
    for column, value_range in value_ranges.items():
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        minimum, maximum = value_range
        value_range[0] = value if minimum is None else min(minimum, value)
        value_range[1] = value if maximum is None else max(maximum, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep requested indicator columns from MT_UPDATE_MS_HEV.csv."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=None,
        help="Normalized output CSV path (default: <input stem>_NORcleaned.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    normalized_output_path = args.normalized_output
    if normalized_output_path is None:
        normalized_output_path = input_path.with_name(
            f"{input_path.stem}_NORcleaned{input_path.suffix}"
        )
    clean_csv(input_path, args.output.resolve(), normalized_output_path.resolve())


if __name__ == "__main__":
    main()
