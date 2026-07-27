"""Create one immutable, audited clustering-input artifact from a high-risk CSV."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import pandas as pd

from scripts.utils import clustering_preprocessing as preprocessing


DEFAULT_OUTPUT_ROOT = Path("outputs_preprocessing")
SUPPORTED_RISK_OPERATORS = frozenset({">", ">="})


def load_exclusions(path: Path | None, scenario: str) -> tuple[str, ...]:
    """Load documented active-scenario feature exclusions, if supplied."""
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Exclusion artifact root must be a JSON object.")
    required_keys = ("scenario", "excluded_features", "reason")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(
            "Exclusion artifact is missing required keys: " + ", ".join(missing_keys)
        )

    canonical_scenario = preprocessing.normalize_scenario(scenario)
    artifact_scenario = payload["scenario"]
    if not isinstance(artifact_scenario, str):
        raise ValueError("Exclusion artifact scenario must be a string.")
    try:
        normalized_artifact_scenario = preprocessing.normalize_scenario(
            artifact_scenario
        )
    except ValueError as exc:
        raise ValueError("Exclusion artifact scenario must be canonical PF or TF.") from exc
    if artifact_scenario != normalized_artifact_scenario:
        raise ValueError("Exclusion artifact scenario must be canonical PF or TF.")
    if normalized_artifact_scenario != canonical_scenario:
        raise ValueError("Exclusion artifact scenario does not match active scenario.")
    names = payload["excluded_features"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("excluded_features must be a JSON list of strings.")
    if len(names) != len(set(names)):
        raise ValueError("excluded_features contains duplicate names.")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string.")
    return tuple(names)


def verify_recorded_risk_filter(
    data: pd.DataFrame,
    column: str,
    operator: str,
    threshold: float,
) -> None:
    """Ensure this input already satisfies its recorded high-risk predicate."""
    if operator not in SUPPORTED_RISK_OPERATORS:
        raise ValueError("risk operator must be one of >, >=")
    if column not in data.columns:
        raise ValueError(f"Recorded risk column is missing: {column}")

    values = pd.to_numeric(data[column], errors="coerce")
    matches = values.gt(threshold) if operator == ">" else values.ge(threshold)
    if not matches.all():
        raise ValueError(
            "Input contains rows that fail the recorded risk comparison: "
            f"{column} {operator} {threshold}"
        )


def _print_audit(
    source: pd.DataFrame,
    prepared: preprocessing.PreparedClusteringData,
    artifact_dir: Path,
) -> None:
    """Print a compact, human-readable summary of the prepared artifact."""
    removed_constants = prepared.audit.loc[
        prepared.audit["event"].eq("constant_feature_removed"), "column"
    ].tolist()

    print(f"Input rows: {len(source)}")
    print(f"Retained rows: {len(prepared.metadata)}")
    print(f"Feature count: {len(prepared.feature_names)}")
    print(f"Rows containing structural nulls: {int(prepared.rows_with_structural_null.sum())}")
    print("Per-feature null counts:")
    for name, count in prepared.missing_counts.items():
        print(f"  {name}: {int(count)}")
    print(
        "Removed constants: "
        + (", ".join(removed_constants) if removed_constants else "none")
    )
    print(f"Final artifact path: {artifact_dir}")


def run_preparation(
    input_path: Path,
    scenario: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    exclusion_path: Path | None = None,
    risk_column: str | None = None,
    risk_operator: str = ">",
    risk_threshold: float = 0.25,
) -> Path:
    """Prepare and persist an immutable artifact without applying a risk filter."""
    canonical_scenario = preprocessing.normalize_scenario(scenario)
    resolved_input = Path(input_path)
    resolved_exclusion = Path(exclusion_path) if exclusion_path is not None else None
    active_risk_column = risk_column or f"{canonical_scenario}_Index_Risk_equal"
    source = pd.read_csv(resolved_input)
    verify_recorded_risk_filter(
        source, active_risk_column, risk_operator, risk_threshold
    )
    exclusions = load_exclusions(resolved_exclusion, canonical_scenario)
    prepared = preprocessing.prepare_clustering_data(
        source, canonical_scenario, exclusions=exclusions
    )
    artifact_dir = preprocessing.save_prepared_artifact(
        prepared,
        source_path=resolved_input,
        output_root=Path(output_root),
        risk_filter={
            "column": active_risk_column,
            "operator": risk_operator,
            "threshold": risk_threshold,
        },
        exclusion_path=resolved_exclusion,
    )
    _print_audit(source, prepared, artifact_dir)
    return artifact_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for modelling-input preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scenario", choices=("PF", "TF"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--risk-column")
    parser.add_argument("--risk-operator", choices=(">", ">="), default=">")
    parser.add_argument("--risk-threshold", type=float, default=0.25)
    return parser


def main() -> None:
    """Run the preparation command with parsed arguments."""
    args = build_parser().parse_args()
    run_preparation(
        input_path=args.input,
        scenario=args.scenario,
        output_root=args.output_root,
        exclusion_path=args.exclusions,
        risk_column=args.risk_column,
        risk_operator=args.risk_operator,
        risk_threshold=args.risk_threshold,
    )


if __name__ == "__main__":
    main()
