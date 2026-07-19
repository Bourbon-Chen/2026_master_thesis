
#!/usr/bin/env python3
"""Interactively filter highly correlated PF/TF core variables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "MT_UPDATE_MS_HEV_v2.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs_correlation"
DEFAULT_THRESHOLD = 0.6

SCENARIO_CONFIG = {
    "pf": {
        "features": [
            "Per_extent",
            "PF_Index_population",
            "PF_Index_acc_loss",
            "PF_Index_centrality",
        ],
        "risk_index": "PF_Index_Risk_equal",
    },
    "tf": {
        "features": [
            "Tem_extent",
            "TF_Index_population",
            "TF_Index_acc_loss",
            "TF_Index_centrality",
        ],
        "risk_index": "TF_Index_Risk_equal",
    },
}


def parse_decimal(value: str) -> float:
    try:
        parsed = float(value.strip().replace(",", "."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threshold must be a number, for example 0.6"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return parsed


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


def prompt_input_path(default: Path) -> Path:
    while True:
        value = input(f"Enter dataset path under data/ [{default.name}]: ").strip()
        try:
            return resolve_data_path(value or default)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)


def prompt_scenario() -> str:
    while True:
        scenario = input("Enter scenario (pf/tf): ").strip().lower()
        if scenario in SCENARIO_CONFIG:
            return scenario
        print("Scenario must be 'pf' or 'tf'.")


def prompt_threshold(default: float) -> float:
    while True:
        value = input(f"Enter correlation threshold [{default:g}]: ").strip()
        if not value:
            return default
        try:
            return parse_decimal(value)
        except argparse.ArgumentTypeError as exc:
            print(exc)


def find_high_correlations(
    correlation_matrix: pd.DataFrame, threshold: float
) -> list[dict[str, object]]:
    high_correlations: list[dict[str, object]] = []
    columns = list(correlation_matrix.columns)
    for left_index, left_column in enumerate(columns):
        for right_column in columns[left_index + 1 :]:
            correlation = correlation_matrix.loc[left_column, right_column]
            if pd.isna(correlation):
                continue
            if abs(float(correlation)) > threshold:
                high_correlations.append(
                    {
                        "variable_1": left_column,
                        "variable_2": right_column,
                        "correlation": float(correlation),
                        "abs_correlation": abs(float(correlation)),
                    }
                )
    return sorted(
        high_correlations,
        key=lambda row: float(row["abs_correlation"]),
        reverse=True,
    )


def save_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path,
    scenario: str,
    threshold: float,
) -> None:
    variable_count = len(correlation_matrix.columns)
    figure_size = max(8.0, variable_count * 1.8 + 3.0)

    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(
        correlation_matrix,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(variable_count))
    ax.set_yticks(range(variable_count))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlation_matrix.index)
    ax.set_title(
        f"{scenario.upper()} Pearson Correlation Heatmap "
        f"(threshold = {threshold:g})"
    )

    for row_index, row_name in enumerate(correlation_matrix.index):
        for column_index, column_name in enumerate(correlation_matrix.columns):
            value = correlation_matrix.loc[row_name, column_name]
            label = "nan" if pd.isna(value) else f"{float(value):.2f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prompt_removal_decision(
    variable_1: str, variable_2: str, correlation: float
) -> tuple[str, str | None]:
    print()
    print("High correlation detected:")
    print(f"  {variable_1} vs {variable_2}: correlation = {correlation:.4f}")
    print("Which variable do you want to remove?")
    print(f"  [1] {variable_1}")
    print(f"  [2] {variable_2}")
    print("  [3] keep both")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            return "remove_variable_1", variable_1
        if choice == "2":
            return "remove_variable_2", variable_2
        if choice == "3":
            return "keep_both", None
        print("Please enter 1, 2, or 3.")


def run_filter(
    input_path: Path,
    scenario: str,
    threshold: float,
    output_root: Path,
) -> None:
    config = SCENARIO_CONFIG[scenario]
    features = list(config["features"])
    risk_index = str(config["risk_index"])
    required_columns = features + [risk_index]

    data = pd.read_csv(input_path)
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    numeric_features = data[features].apply(pd.to_numeric, errors="coerce")
    correlation_matrix = numeric_features.corr(method="pearson")
    high_correlations = find_high_correlations(correlation_matrix, threshold)

    output_dir = output_root.resolve() / scenario.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_matrix.to_csv(output_dir / "correlation_matrix.csv")
    heatmap_path = output_dir / "pearson_correlation_heatmap.png"
    save_correlation_heatmap(correlation_matrix, heatmap_path, scenario, threshold)

    removed_variables: set[str] = set()
    decision_rows: list[dict[str, object]] = []

    print(f"Input CSV: {input_path}")
    print(f"Scenario: {scenario.upper()}")
    print(f"Threshold: abs(correlation) > {threshold:g}")
    print(f"Output directory: {output_dir}")
    print("Variables used for correlation:")
    for feature in features:
        print(f"  - {feature}")
    print(f"Risk index retained but excluded from correlation: {risk_index}")

    if not high_correlations:
        print("No variable pairs exceed the correlation threshold.")

    for row in high_correlations:
        variable_1 = str(row["variable_1"])
        variable_2 = str(row["variable_2"])
        correlation = float(row["correlation"])

        if variable_1 in removed_variables or variable_2 in removed_variables:
            decision_rows.append(
                {
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "threshold": threshold,
                    "decision": "skipped_already_removed",
                    "removed_variable": "",
                    "kept_variable": "",
                }
            )
            continue

        decision, removed_variable = prompt_removal_decision(
            variable_1, variable_2, correlation
        )
        if removed_variable is None:
            kept_variable = "both"
        else:
            removed_variables.add(removed_variable)
            kept_variable = variable_2 if removed_variable == variable_1 else variable_1

        decision_rows.append(
            {
                "variable_1": variable_1,
                "variable_2": variable_2,
                "correlation": correlation,
                "abs_correlation": abs(correlation),
                "threshold": threshold,
                "decision": decision,
                "removed_variable": removed_variable or "",
                "kept_variable": kept_variable,
            }
        )

    final_features = [
        feature for feature in features if feature not in removed_variables
    ] + [risk_index]

    removed_table = pd.DataFrame(
        decision_rows,
        columns=[
            "variable_1",
            "variable_2",
            "correlation",
            "abs_correlation",
            "threshold",
            "decision",
            "removed_variable",
            "kept_variable",
        ],
    )
    removed_table.to_csv(output_dir / "removed_collinear_variables.csv", index=False)

    final_table = pd.DataFrame(
        {
            "feature": final_features,
            "used_in_correlation": [
                feature != risk_index for feature in final_features
            ],
        }
    )
    final_table.to_csv(output_dir / "final_feature_list.csv", index=False)

    print()
    print(f"Saved: {output_dir / 'correlation_matrix.csv'}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {output_dir / 'removed_collinear_variables.csv'}")
    print(f"Saved: {output_dir / 'final_feature_list.csv'}")
    print("Final feature list:")
    for feature in final_features:
        print(f"  - {feature}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PF/TF core-variable correlations and interactively choose "
            "which highly correlated variables to remove."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help=f"Input CSV file name/path under data/ (default prompt: {DEFAULT_INPUT.name}).",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        choices=sorted(SCENARIO_CONFIG),
        default=None,
        help="Scenario to filter: pf or tf. Prompted when omitted.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=parse_decimal,
        default=None,
        help=f"Correlation threshold between 0 and 1 (default prompt: {DEFAULT_THRESHOLD}).",
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
    input_path = (
        resolve_data_path(args.input)
        if args.input is not None
        else prompt_input_path(DEFAULT_INPUT)
    )
    scenario = args.scenario if args.scenario is not None else prompt_scenario()
    threshold = (
        args.threshold
        if args.threshold is not None
        else prompt_threshold(DEFAULT_THRESHOLD)
    )
    run_filter(input_path, scenario, threshold, args.output_root)


if __name__ == "__main__":
    main()
