"""Build a validated feature-exclusions JSON for clustering preparation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import clustering_preprocessing as preprocessing


REQUIRED_ROLE_COLUMNS = ("column", "role", "reason")
EXCLUSION_REASON = "User-selected exclusions after Pearson correlation review"
RAW_AB_REASON = preprocessing.RAW_AB_LOSSR_EXCLUSION_REASON
ACTIVE_ROLES = {"PF": "PF_FEATURE", "TF": "TF_FEATURE"}
OPPOSITE_ROLES = {"PF": "TF_FEATURE", "TF": "PF_FEATURE"}
EXCLUSION_PROMPT = "Columns to exclude, separated by commas (blank keeps all):"
DEFAULT_CORRELATION_ROOT = PROJECT_ROOT / "outputs_correlation"


class ExclusionValidationError(ValueError):
    """Report every invalid exclusion candidate in one correction pass."""


def parse_exclusion_text(text: str) -> tuple[str, ...]:
    normalized = text.replace("，", ",")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in normalized.split(","):
        name = item.strip()
        if name and name not in seen:
            ordered.append(name)
            seen.add(name)
    return tuple(ordered)


def load_feature_roles(path: Path) -> pd.DataFrame:
    roles_path = Path(path)
    if not roles_path.is_file():
        raise FileNotFoundError(f"Feature roles CSV not found: {roles_path}")
    roles = pd.read_csv(roles_path, keep_default_na=False)
    missing = [name for name in REQUIRED_ROLE_COLUMNS if name not in roles.columns]
    if missing:
        raise ValueError(
            "Feature roles CSV is missing required columns: " + ", ".join(missing)
        )
    roles = roles.loc[:, list(REQUIRED_ROLE_COLUMNS)].copy()
    if roles["column"].duplicated().any():
        duplicates = roles.loc[roles["column"].duplicated(), "column"].tolist()
        raise ValueError(
            "Feature roles CSV contains duplicate columns: " + ", ".join(duplicates)
        )
    return roles


def _is_non_numeric_candidate_for_scenario(
    name: str,
    scenario: str,
    role: str,
) -> bool:
    if role != preprocessing.FieldRole.NON_NUMERIC.value:
        return False
    pattern = (
        preprocessing.PF_PATTERN
        if scenario == "PF"
        else preprocessing.TF_PATTERN
    )
    return bool(pattern.match(name))


def validate_exclusions(
    roles: pd.DataFrame,
    scenario: object,
    names: Sequence[str],
) -> tuple[str, ...]:
    canonical = preprocessing.normalize_scenario(scenario)
    indexed = roles.set_index("column", drop=False)
    categories: dict[str, list[str]] = {
        "absent from feature_roles.csv": [],
        "opposite scenario": [],
        "protected or non-feature": [],
        "already excluded by a built-in rule": [],
        "non-numeric active-scenario candidate": [],
    }
    raw_ab_names: list[str] = []

    for name in names:
        if name not in indexed.index:
            categories["absent from feature_roles.csv"].append(name)
            continue
        record = indexed.loc[name]
        role = str(record["role"])
        reason = str(record["reason"])
        if role == ACTIVE_ROLES[canonical]:
            continue
        if role == OPPOSITE_ROLES[canonical]:
            categories["opposite scenario"].append(name)
        elif reason == RAW_AB_REASON:
            categories["already excluded by a built-in rule"].append(name)
            raw_ab_names.append(name)
        elif _is_non_numeric_candidate_for_scenario(
            name,
            "TF" if canonical == "PF" else "PF",
            role,
        ):
            categories["opposite scenario"].append(name)
        elif _is_non_numeric_candidate_for_scenario(name, canonical, role):
            categories["non-numeric active-scenario candidate"].append(name)
        else:
            categories["protected or non-feature"].append(name)

    populated = [(label, values) for label, values in categories.items() if values]
    if populated:
        lines = ["Invalid exclusion columns:"]
        lines.extend(f"- {label}: {', '.join(values)}" for label, values in populated)
        if raw_ab_names:
            lines.append(
                "- Raw AB lossR columns are already excluded and should not be added; "
                "use the normalized AB2k_NOR/AB5k_NOR columns when relevant."
            )
        raise ExclusionValidationError("\n".join(lines))
    return tuple(names)


def default_paths(scenario: object) -> tuple[Path, Path]:
    canonical = preprocessing.normalize_scenario(scenario)
    scenario_directory = DEFAULT_CORRELATION_ROOT / canonical.lower()
    return (
        scenario_directory / "feature_roles.csv",
        scenario_directory / "feature_exclusions.json",
    )


def build_payload(
    scenario: object,
    names: Sequence[str],
) -> dict[str, object]:
    return {
        "scenario": preprocessing.normalize_scenario(scenario),
        "excluded_features": list(names),
        "reason": EXCLUSION_REASON,
    }


def ensure_output_available(output_path: Path, overwrite: bool) -> None:
    if Path(output_path).exists() and not overwrite:
        raise FileExistsError(
            f"Exclusions JSON already exists: {output_path}. "
            "Use --overwrite to replace it."
        )


def write_exclusions_json(
    output_path: Path,
    payload: dict[str, object],
    overwrite: bool = False,
) -> Path:
    destination = Path(output_path)
    ensure_output_available(destination, overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def generate_exclusions(
    scenario: object,
    exclusion_text: str,
    roles_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    canonical = preprocessing.normalize_scenario(scenario)
    names = parse_exclusion_text(exclusion_text)
    roles = load_feature_roles(roles_path)
    validated = validate_exclusions(roles, canonical, names)
    payload = build_payload(canonical, validated)
    return write_exclusions_json(output_path, payload, overwrite=overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=str.upper,
        choices=("PF", "TF"),
        required=True,
    )
    parser.add_argument("--exclude")
    parser.add_argument("--roles", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_roles, default_output = default_paths(args.scenario)
    roles_path = args.roles if args.roles is not None else default_roles
    output_path = args.output if args.output is not None else default_output
    try:
        ensure_output_available(output_path, args.overwrite)
        exclusion_text = (
            args.exclude
            if args.exclude is not None
            else input(EXCLUSION_PROMPT)
        )
        written_path = generate_exclusions(
            scenario=args.scenario,
            exclusion_text=exclusion_text,
            roles_path=roles_path,
            output_path=output_path,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    selected = parse_exclusion_text(exclusion_text)
    print(f"Exclusions JSON: {written_path}")
    print(
        "Selected exclusions: "
        + (", ".join(selected) if selected else "none")
    )
    print(
        "Next: pass this file to "
        f"prepare_clustering_inputs.py --exclusions {written_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
