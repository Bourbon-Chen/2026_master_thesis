#!/usr/bin/env python3
"""Create PF- or TF-specific datasets for PCA and Elbow analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "MT_UPDATE_MS_HEV_v2_NORcleaned.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs_correlation"
IDENTIFIER_COLUMNS = ("fid", "MS_ID")

SCENARIO_CONFIG: Dict[str, Dict[str, object]] = {
    "pf": {
        "default_removals": (
            "Per_extent",
            "PFAB2k_NOR",
            "PFPRM_lossR",
            "N_PFlossR_2ktw",
        ),
        "opposite_token": "TF",
        "opposite_exposure": "Tem_extent",
        "output_name": "pf_for_PCA.csv",
    },
    "tf": {
        "default_removals": (
            "Tem_extent",
            "TFAB2k_NOR",
            "TFPRM_lossR",
            "N_TFlossR_2ktw",
        ),
        "opposite_token": "PF",
        "opposite_exposure": "Per_extent",
        "output_name": "tf_for_PCA.csv",
    },
}


def prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path:
    """Prompt for an input CSV path, using the configured path when blank."""
    entered_path = input(f"Enter input CSV path [{default_path}]: ").strip()
    return Path(entered_path) if entered_path else default_path


def _scenario_config(scenario: str) -> Dict[str, object]:
    normalized = scenario.strip().lower()
    if normalized not in SCENARIO_CONFIG:
        raise ValueError("Scenario must be 'pf' or 'tf'.")
    return SCENARIO_CONFIG[normalized]


def parse_removal_input(value: str) -> List[str]:
    """Parse a comma-separated column list while preserving its order."""
    parsed: List[str] = []
    for item in value.replace("，", ",").split(","):
        column = item.strip()
        if column and column not in parsed:
            parsed.append(column)
    return parsed


def _is_index_column(column: str) -> bool:
    return "index" in column.lower()


def _is_opposite_scenario_column(column: str, scenario: str) -> bool:
    config = _scenario_config(scenario)
    opposite_token = str(config["opposite_token"])
    opposite_exposure = str(config["opposite_exposure"])
    uppercase = column.upper()
    return uppercase.startswith(
        (opposite_token, f"N_{opposite_token}")
    ) or column == opposite_exposure


def validate_removal_columns(
    available_columns: Iterable[str],
    scenario: str,
    removal_columns: Sequence[str],
) -> None:
    """Reject missing, protected, or opposite-scenario manual removals."""
    _scenario_config(scenario)
    available = set(available_columns)
    protected = [
        column
        for column in removal_columns
        if column in IDENTIFIER_COLUMNS or _is_index_column(column)
    ]
    if protected:
        raise ValueError(
            "The following protected identifier/index columns cannot be removed: "
            + ", ".join(protected)
        )

    not_removable = [
        column
        for column in removal_columns
        if column not in available
        or _is_opposite_scenario_column(column, scenario)
    ]
    if not_removable:
        raise ValueError(
            "The following columns are not removable for "
            f"{scenario.upper()}: "
            + ", ".join(not_removable)
        )


def filter_dataset(
    data: pd.DataFrame,
    scenario: str,
    removal_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove selected high-correlation and all opposite-scenario columns."""
    normalized = scenario.strip().lower()
    missing_identifiers = [
        column for column in IDENTIFIER_COLUMNS if column not in data.columns
    ]
    if missing_identifiers:
        raise ValueError(
            "Missing required identifier columns: "
            + ", ".join(missing_identifiers)
        )
    validate_removal_columns(data.columns, normalized, removal_columns)

    requested = set(removal_columns)
    removed_columns = [
        column
        for column in data.columns
        if column in requested
        or _is_opposite_scenario_column(column, normalized)
    ]
    filtered = data.drop(columns=removed_columns).copy()
    return filtered, removed_columns


def prompt_scenario() -> str:
    """Prompt until the user selects PF or TF."""
    while True:
        scenario = input("Select scenario (pf/tf): ").strip().lower()
        if scenario in SCENARIO_CONFIG:
            return scenario
        print("Please enter 'pf' or 'tf'.")


def prompt_removal_columns(
    available_columns: Iterable[str], scenario: str
) -> List[str]:
    """Prompt for manual removals, using scenario defaults on empty input."""
    config = _scenario_config(scenario)
    defaults = list(config["default_removals"])
    print("Default high-correlation columns to remove:")
    for column in defaults:
        print(f"  - {column}")
    print("Press Enter to use the defaults, or enter comma-separated column names.")

    while True:
        value = input("Columns to remove: ").strip()
        removal_columns = parse_removal_input(value) if value else defaults
        try:
            validate_removal_columns(
                available_columns, scenario, removal_columns
            )
        except ValueError as error:
            print(error)
            continue
        return removal_columns


def resolve_input_path(value: str | Path) -> Path:
    """Resolve a CLI input path and require an existing CSV file."""
    raw_path = Path(value)
    if raw_path.is_absolute():
        input_path = raw_path
    elif raw_path.parts and raw_path.parts[0].lower() == "data":
        input_path = PROJECT_ROOT / raw_path
    else:
        input_path = DATA_DIR / raw_path

    resolved = input_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {resolved}")
    if resolved.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV: {resolved}")
    return resolved


def extract_dataset(
    input_path: Path,
    scenario: str,
    removal_columns: Sequence[str],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Filter one CSV and save it below the scenario output directory."""
    normalized = scenario.strip().lower()
    config = _scenario_config(normalized)
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    data = pd.read_csv(input_path, low_memory=False)
    filtered, removed_columns = filter_dataset(
        data, normalized, removal_columns
    )

    output_directory = Path(output_root) / normalized
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / str(config["output_name"])
    filtered.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input CSV: {input_path}")
    print(f"Scenario: {normalized.upper()}")
    print(f"Removed columns ({len(removed_columns)}):")
    for column in removed_columns:
        print(f"  - {column}")
    print(f"Saved dataset: {output_path}")
    print(
        f"Output shape: {len(filtered)} rows x {len(filtered.columns)} columns"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove selected high-correlation columns and all columns from the "
            "opposite PF/TF scenario, then create a PCA-ready CSV."
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
        "-s",
        "--scenario",
        type=str.lower,
        choices=sorted(SCENARIO_CONFIG),
        default=None,
        help="Scenario to retain: pf or tf. Prompted when omitted.",
    )
    parser.add_argument(
        "-r",
        "--remove",
        default=None,
        help=(
            "Comma-separated high-correlation columns to remove. "
            "Prompted when omitted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_input_path = (
        args.input if args.input is not None else prompt_input_path()
    )
    input_path = resolve_input_path(raw_input_path)
    scenario = args.scenario or prompt_scenario()
    available_columns = pd.read_csv(input_path, nrows=0).columns

    if args.remove is None:
        removal_columns = prompt_removal_columns(available_columns, scenario)
    else:
        removal_columns = parse_removal_input(args.remove)
        if not removal_columns:
            removal_columns = list(
                _scenario_config(scenario)["default_removals"]
            )
        validate_removal_columns(
            available_columns, scenario, removal_columns
        )

    extract_dataset(
        input_path,
        scenario,
        removal_columns,
        args.output_root,
    )


if __name__ == "__main__":
    main()
