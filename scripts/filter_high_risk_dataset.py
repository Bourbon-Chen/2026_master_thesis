#!/usr/bin/env python3
"""Filter a normalized or standardized HEV dataset by PF/TF risk index."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "MT_UPDATE_MS_HEV_v2_NORcleaned.csv"
RISK_COLUMNS = {
    "pf": "PF_Index_Risk_equal",
    "tf": "TF_Index_Risk_equal",
}


def parse_decimal(value: str) -> float:
    """Parse decimal values typed as either 0.25 or 0,25."""
    try:
        parsed = float(value.strip().replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "risk index threshold must be a number, for example 0.25"
        ) from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("risk index threshold must be finite")
    return parsed


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace("-", "minus_").replace(".", "_")


def default_output_path(input_path: Path, scenario: str, threshold: float) -> Path:
    return (
        input_path.parent
        / f"{input_path.stem}_{scenario}_risk_gt_{threshold_tag(threshold)}{input_path.suffix}"
    )


def prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path:
    entered_path = input(f"Enter input CSV path [{default_path}]: ").strip()
    return Path(entered_path) if entered_path else default_path


def prompt_scenario() -> str:
    while True:
        scenario = input("Enter risk type (pf/tf): ").strip().lower()
        if scenario in RISK_COLUMNS:
            return scenario
        print("Risk type must be 'pf' or 'tf'.")


def prompt_threshold() -> float:
    while True:
        try:
            return parse_decimal(input("Enter risk index threshold: "))
        except argparse.ArgumentTypeError as exc:
            print(exc)


def filter_dataset(
    input_path: Path,
    output_path: Path,
    scenario: str,
    threshold: float,
) -> None:
    risk_column = RISK_COLUMNS[scenario]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {input_path}")
        if risk_column not in reader.fieldnames:
            raise ValueError(
                f"Missing required risk column '{risk_column}' in {input_path}"
            )

        with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()

            total_rows = 0
            matched_rows = 0
            non_numeric_rows = 0
            for row in reader:
                total_rows += 1
                try:
                    risk_value = float(row[risk_column])
                except (TypeError, ValueError):
                    non_numeric_rows += 1
                    continue
                if not math.isfinite(risk_value):
                    non_numeric_rows += 1
                    continue
                if risk_value > threshold:
                    writer.writerow(row)
                    matched_rows += 1

    print(f"Input CSV: {input_path}")
    print(f"Risk type: {scenario.upper()}")
    print(f"Risk column: {risk_column}")
    print(f"Filter: {risk_column} > {threshold:g}")
    print(f"Output CSV: {output_path}")
    print(f"Rows read: {total_rows}")
    print(f"Rows written: {matched_rows}")
    if non_numeric_rows:
        print(f"Rows skipped because risk index is non-numeric: {non_numeric_rows}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a normalized or standardized HEV CSV by PF/TF risk index. "
            "When --scenario or --threshold is omitted, the script prompts for it."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=(
            "Input CSV path "
            f"(prompted when omitted; blank uses: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input stem>_<pf|tf>_risk_gt_<threshold>.csv)",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        choices=sorted(RISK_COLUMNS),
        default=None,
        help="Risk type to filter by: pf or tf. Prompted when omitted.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=parse_decimal,
        default=None,
        help="Risk index threshold, for example 0.25. Prompted when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = args.scenario if args.scenario is not None else prompt_scenario()
    threshold = args.threshold if args.threshold is not None else prompt_threshold()

    input_path = (
        args.input if args.input is not None else prompt_input_path()
    ).resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else default_output_path(input_path, scenario, threshold).resolve()
    )
    filter_dataset(input_path, output_path, scenario, threshold)


if __name__ == "__main__":
    main()
