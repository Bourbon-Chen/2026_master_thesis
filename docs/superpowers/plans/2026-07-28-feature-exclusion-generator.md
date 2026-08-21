# Feature Exclusion Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-time command-line generator that validates a researcher-entered list of PF or TF feature names and writes the minimal exclusions JSON already consumed by clustering preparation.

**Architecture:** Keep the active modelling boundary unchanged: Pearson correlation continues to produce `feature_roles.csv`, the new standalone generator converts one manual list into `feature_exclusions.json`, and `prepare_clustering_inputs.py` remains the sole creator of standardized prepared artifacts. The generator separates pure parsing and validation from atomic file publication and exposes a small `main(argv)` entry point so all interactive, non-interactive, safety, and compatibility behavior can be tested without running PCA or clustering.

**Tech Stack:** Python 3 in Conda environment `2026master`, standard library (`argparse`, `json`, `pathlib`, `tempfile`), pandas, and the existing `scripts.utils.clustering_preprocessing` and `scripts.cluster_preparation.prepare_clustering_inputs` modules.

## Global Constraints

- The generator is `scripts/pearson_correlation/build_feature_exclusions.py`.
- It never inspects, ranks, displays, or decides from individual correlation pairs.
- When `--exclude` is omitted, prompt exactly once with `Columns to exclude, separated by commas (blank keeps all):`.
- Accept ASCII `,` and Chinese `，`; trim entries, discard empty entries, and deduplicate while preserving first occurrence and user order.
- A blank prompt response or blank `--exclude` value produces an empty exclusion list.
- Column matching is exact and case-sensitive.
- `--scenario` is required, accepts PF/TF case-insensitively, and is stored as canonical uppercase.
- Validate against a roles CSV containing the required columns `column`, `role`, and `reason`; only `PF_FEATURE` is selectable for PF and only `TF_FEATURE` is selectable for TF.
- Report all invalid entered names together, categorized as absent, opposite-scenario, protected/non-feature, already excluded by a built-in rule, or non-numeric active-scenario candidate.
- For reason `raw_ab_lossR_replaced_by_normalized_NOR`, explicitly say the raw AB lossR column is already excluded and should not be added.
- The JSON field order is `scenario`, `excluded_features`, `reason`, with the exact reason `User-selected exclusions after Pearson correlation review`.
- Default roles and output paths are `outputs_correlation/<pf|tf>/feature_roles.csv` and `outputs_correlation/<pf|tf>/feature_exclusions.json`; `--roles` and `--output` override them.
- Refuse an existing output before prompting unless `--overwrite` is supplied.
- Publish through a same-directory temporary file followed by `Path.replace`; failures must leave any existing decision file unchanged.
- Do not modify correlation results, source CSVs, prepared artifacts, PCA outputs, or clustering outputs.
- Keep `extract_high_columns.py` as a clearly documented legacy compatibility exporter.
- Use TDD, run all Python verification in Conda environment `2026master`, preserve unrelated working-tree changes, and stage only files named by the current task.

## File Structure

- Create `scripts/pearson_correlation/build_feature_exclusions.py`: parsing, roles loading, categorized validation, JSON construction, atomic publication, CLI defaults, prompt, and user-facing result messages.
- Create `scripts/pearson_correlation/test_build_feature_exclusions.py`: unit, CLI, file-safety, existing-loader compatibility, and end-to-end preparation tests.
- Modify `README.md`: insert the generator between correlation analysis and clustering preparation, pass its JSON through `--exclusions`, and explain immutable regeneration and the legacy exporter.
- Do not modify `prepare_clustering_inputs.py`: the integration tests must prove that its existing public interface already accepts the generated artifact.

---

### Task 1: Parse and validate the researcher’s exclusion decision

**Files:**
- Create: `scripts/pearson_correlation/build_feature_exclusions.py`
- Create: `scripts/pearson_correlation/test_build_feature_exclusions.py`

**Interfaces:**
- Consumes: scenario text, raw comma-separated text, and a pandas `DataFrame` loaded from `feature_roles.csv`.
- Produces: `parse_exclusion_text(text: str) -> tuple[str, ...]`, `load_feature_roles(path: Path) -> pd.DataFrame`, and `validate_exclusions(roles: pd.DataFrame, scenario: object, names: Sequence[str]) -> tuple[str, ...]`.
- Raises: `ExclusionValidationError(ValueError)` containing every invalid name grouped by category.
- Guarantees: exact case-sensitive column matching, stable order, stable deduplication, active-role-only selection, and no filesystem writes during validation.

- [ ] **Step 1: Write failing parsing and validation tests**

Create `scripts/pearson_correlation/test_build_feature_exclusions.py` with a reusable roles table and focused tests:

```python
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.pearson_correlation import build_feature_exclusions as exclusions


class FeatureExclusionParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roles = pd.DataFrame(
            [
                {"column": "Per_extent", "role": "PF_FEATURE", "reason": ""},
                {"column": "PFAB2k_NOR", "role": "PF_FEATURE", "reason": ""},
                {"column": "Tem_extent", "role": "TF_FEATURE", "reason": ""},
                {"column": "fid", "role": "IDENTIFIER", "reason": ""},
                {
                    "column": "PFAB2k_lossR",
                    "role": "EXCLUDED",
                    "reason": "raw_ab_lossR_replaced_by_normalized_NOR",
                },
                {
                    "column": "PF_text",
                    "role": "NON_NUMERIC",
                    "reason": "active_scenario_feature_is_non_numeric",
                },
            ],
            columns=["column", "role", "reason"],
        )

    def test_parse_accepts_both_commas_and_preserves_first_occurrence(self) -> None:
        parsed = exclusions.parse_exclusion_text(
            " Per_extent，PFAB2k_NOR, ,Per_extent "
        )

        self.assertEqual(parsed, ("Per_extent", "PFAB2k_NOR"))

    def test_parse_blank_input_returns_an_empty_tuple(self) -> None:
        self.assertEqual(exclusions.parse_exclusion_text("   "), ())

    def test_validation_accepts_only_exact_active_scenario_features(self) -> None:
        self.assertEqual(
            exclusions.validate_exclusions(
                self.roles, "pf", ("Per_extent", "PFAB2k_NOR")
            ),
            ("Per_extent", "PFAB2k_NOR"),
        )
        self.assertEqual(
            exclusions.validate_exclusions(
                self.roles, "tf", ("Tem_extent",)
            ),
            ("Tem_extent",),
        )

        with self.assertRaisesRegex(
            exclusions.ExclusionValidationError,
            "absent from feature_roles.csv.*per_extent",
        ):
            exclusions.validate_exclusions(self.roles, "PF", ("per_extent",))

    def test_validation_reports_all_invalid_categories_in_one_error(self) -> None:
        with self.assertRaises(exclusions.ExclusionValidationError) as raised:
            exclusions.validate_exclusions(
                self.roles,
                "PF",
                (
                    "not_present",
                    "Tem_extent",
                    "fid",
                    "PFAB2k_lossR",
                    "PF_text",
                ),
            )

        message = str(raised.exception)
        self.assertIn("absent from feature_roles.csv: not_present", message)
        self.assertIn("opposite scenario: Tem_extent", message)
        self.assertIn("protected or non-feature: fid", message)
        self.assertIn(
            "already excluded by a built-in rule: PFAB2k_lossR", message
        )
        self.assertIn(
            "already excluded and should not be added", message
        )
        self.assertIn(
            "non-numeric active-scenario candidate: PF_text", message
        )

    def test_roles_loader_requires_file_columns_and_unique_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_path = root / "missing.csv"
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                exclusions.load_feature_roles(missing_path)

            malformed_path = root / "malformed.csv"
            pd.DataFrame({"column": ["Per_extent"]}).to_csv(
                malformed_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "role, reason"):
                exclusions.load_feature_roles(malformed_path)

            duplicate_path = root / "duplicate.csv"
            pd.concat([self.roles.iloc[[0]], self.roles.iloc[[0]]]).to_csv(
                duplicate_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                exclusions.load_feature_roles(duplicate_path)
```

- [ ] **Step 2: Run the new test module and confirm the missing-module failure**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.pearson_correlation.test_build_feature_exclusions -v
```

Expected: FAIL while importing `scripts.pearson_correlation.build_feature_exclusions`.

- [ ] **Step 3: Implement parsing, roles loading, and categorized validation**

Create the module with the direct-script project-root bootstrap and these concrete definitions:

```python
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


def _is_active_non_numeric(name: str, scenario: str, role: str) -> bool:
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
        elif _is_active_non_numeric(name, canonical, role):
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
```

The imports for `argparse`, `json`, `os`, and `tempfile` are used by Task 2 and may remain unused until that task is implemented.

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.pearson_correlation.test_build_feature_exclusions.FeatureExclusionParsingTests -v
```

Expected: all `FeatureExclusionParsingTests` pass.

- [ ] **Step 5: Commit the validated parsing boundary**

Run:

```powershell
git add scripts/pearson_correlation/build_feature_exclusions.py scripts/pearson_correlation/test_build_feature_exclusions.py
git diff --cached --check
git commit -m "feat: validate clustering feature exclusions"
```

Expected: only the two named files are staged, the diff check is clean, and the commit succeeds.

---

### Task 2: Add one-prompt CLI and atomic JSON publication

**Files:**
- Modify: `scripts/pearson_correlation/build_feature_exclusions.py`
- Modify: `scripts/pearson_correlation/test_build_feature_exclusions.py`

**Interfaces:**
- Consumes: `--scenario`, optional `--exclude`, optional `--roles`, optional `--output`, and `--overwrite`.
- Produces: `default_paths(scenario: object) -> tuple[Path, Path]`, `build_payload(scenario: object, names: Sequence[str]) -> dict[str, object]`, `write_exclusions_json(output_path: Path, payload: dict[str, object], overwrite: bool = False) -> Path`, `generate_exclusions(...) -> Path`, `build_parser() -> argparse.ArgumentParser`, and `main(argv: Sequence[str] | None = None) -> int`.
- Guarantees: output existence is checked before interactive input; omitted `--exclude` prompts exactly once; supplied `--exclude`, including `""`, never prompts; JSON is minimal and stable; temporary-write failures preserve the prior file.

- [ ] **Step 1: Write failing CLI, output-contract, and file-safety tests**

Extend the test module with these imports:

```python
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import subprocess
import sys
from unittest import mock
```

Then add:

```python
class FeatureExclusionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.roles_path = self.root / "feature_roles.csv"
        pd.DataFrame(
            [
                {"column": "Per_extent", "role": "PF_FEATURE", "reason": ""},
                {"column": "PFAB2k_NOR", "role": "PF_FEATURE", "reason": ""},
                {"column": "Tem_extent", "role": "TF_FEATURE", "reason": ""},
            ]
        ).to_csv(self.roles_path, index=False)
        self.output_path = self.root / "feature_exclusions.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_default_paths_are_scenario_specific_and_scenario_is_case_insensitive(self) -> None:
        pf_roles, pf_output = exclusions.default_paths("pf")
        tf_roles, tf_output = exclusions.default_paths("TF")

        self.assertEqual(
            pf_roles,
            exclusions.PROJECT_ROOT / "outputs_correlation/pf/feature_roles.csv",
        )
        self.assertEqual(
            pf_output,
            exclusions.PROJECT_ROOT
            / "outputs_correlation/pf/feature_exclusions.json",
        )
        self.assertEqual(
            tf_roles,
            exclusions.PROJECT_ROOT / "outputs_correlation/tf/feature_roles.csv",
        )
        self.assertEqual(
            tf_output,
            exclusions.PROJECT_ROOT
            / "outputs_correlation/tf/feature_exclusions.json",
        )

    def test_omitted_exclude_prompts_once_and_writes_exact_minimal_json(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch(
                "builtins.input",
                return_value=" Per_extent，PFAB2k_NOR,Per_extent ",
            ) as prompt,
            redirect_stdout(stdout),
        ):
            result = exclusions.main(
                [
                    "--scenario",
                    "pf",
                    "--roles",
                    str(self.roles_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        prompt.assert_called_once_with(exclusions.EXCLUSION_PROMPT)
        self.assertEqual(result, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            list(payload),
            ["scenario", "excluded_features", "reason"],
        )
        self.assertEqual(
            payload,
            {
                "scenario": "PF",
                "excluded_features": ["Per_extent", "PFAB2k_NOR"],
                "reason": exclusions.EXCLUSION_REASON,
            },
        )
        self.assertIn(
            "Selected exclusions: Per_extent, PFAB2k_NOR",
            stdout.getvalue(),
        )
        self.assertIn("prepare_clustering_inputs.py --exclusions", stdout.getvalue())

    def test_blank_noninteractive_exclude_does_not_prompt(self) -> None:
        with mock.patch(
            "builtins.input",
            side_effect=AssertionError("non-interactive input must not prompt"),
        ):
            result = exclusions.main(
                [
                    "--scenario",
                    "PF",
                    "--exclude",
                    "",
                    "--roles",
                    str(self.roles_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        self.assertEqual(result, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_features"], [])

    def test_blank_prompt_response_writes_an_empty_exclusion_list(self) -> None:
        with mock.patch("builtins.input", return_value="   ") as prompt:
            result = exclusions.main(
                [
                    "--scenario",
                    "PF",
                    "--roles",
                    str(self.roles_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        self.assertEqual(result, 0)
        prompt.assert_called_once_with(exclusions.EXCLUSION_PROMPT)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_features"], [])

    def test_existing_output_is_rejected_before_prompt_without_overwrite(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        stderr = io.StringIO()
        with (
            mock.patch("builtins.input") as prompt,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            exclusions.main(
                [
                    "--scenario",
                    "PF",
                    "--roles",
                    str(self.roles_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        prompt.assert_not_called()
        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"), "old decision"
        )
        self.assertIn("--overwrite", stderr.getvalue())

    def test_invalid_interactive_input_prompts_once_and_writes_nothing(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch("builtins.input", return_value="Tem_extent") as prompt,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            exclusions.main(
                [
                    "--scenario",
                    "PF",
                    "--roles",
                    str(self.roles_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        prompt.assert_called_once_with(exclusions.EXCLUSION_PROMPT)
        self.assertFalse(self.output_path.exists())
        self.assertIn("opposite scenario: Tem_extent", stderr.getvalue())

    def test_overwrite_replaces_the_existing_decision(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        result = exclusions.main(
            [
                "--scenario",
                "PF",
                "--exclude",
                "Per_extent",
                "--roles",
                str(self.roles_path),
                "--output",
                str(self.output_path),
                "--overwrite",
            ]
        )

        self.assertEqual(result, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_features"], ["Per_extent"])

    def test_interrupted_temp_write_preserves_existing_decision(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        payload = exclusions.build_payload("PF", ("Per_extent",))
        with mock.patch.object(
            exclusions.json,
            "dump",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                exclusions.write_exclusions_json(
                    self.output_path, payload, overwrite=True
                )

        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"), "old decision"
        )
        self.assertEqual(
            list(self.root.glob(f".{self.output_path.name}.*.tmp")), []
        )

    def test_validation_failure_with_overwrite_preserves_existing_decision(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        with self.assertRaises(exclusions.ExclusionValidationError):
            exclusions.generate_exclusions(
                scenario="PF",
                exclusion_text="Tem_extent",
                roles_path=self.roles_path,
                output_path=self.output_path,
                overwrite=True,
            )

        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"), "old decision"
        )

    def test_direct_script_help_can_import_project_packages(self) -> None:
        script_path = Path(exclusions.__file__)
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--scenario", completed.stdout)
        self.assertIn("--exclude", completed.stdout)
        self.assertIn("--overwrite", completed.stdout)
```

- [ ] **Step 2: Run the CLI tests and verify they fail on missing interfaces**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.pearson_correlation.test_build_feature_exclusions.FeatureExclusionCliTests -v
```

Expected: FAIL because `default_paths`, the writer, the generator, parser, prompt constant, and `main(argv)` are not implemented.

- [ ] **Step 3: Implement stable payload construction and atomic publication**

Add these definitions after validation:

```python
EXCLUSION_PROMPT = "Columns to exclude, separated by commas (blank keeps all):"
DEFAULT_CORRELATION_ROOT = PROJECT_ROOT / "outputs_correlation"


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
```

- [ ] **Step 4: Implement the required command-line behavior**

Add the parser and entry point:

```python
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
```

- [ ] **Step 5: Run all generator tests and compile the entry point**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.pearson_correlation.test_build_feature_exclusions -v
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m py_compile scripts/pearson_correlation/build_feature_exclusions.py scripts/pearson_correlation/test_build_feature_exclusions.py
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python scripts/pearson_correlation/build_feature_exclusions.py --help
```

Expected: all generator tests pass, compilation exits with no output, and help lists the five approved arguments.

- [ ] **Step 6: Commit the safe CLI and publisher**

Run:

```powershell
git add scripts/pearson_correlation/build_feature_exclusions.py scripts/pearson_correlation/test_build_feature_exclusions.py
git diff --cached --check
git commit -m "feat: generate clustering exclusion artifacts"
```

Expected: only the generator and its tests are staged and committed.

---

### Task 3: Prove compatibility with the existing preparation boundary

**Files:**
- Modify: `scripts/pearson_correlation/test_build_feature_exclusions.py`
- Verify without modification: `scripts/cluster_preparation/prepare_clustering_inputs.py`
- Verify without modification: `scripts/utils/clustering_preprocessing.py`

**Interfaces:**
- Consumes: the JSON written by `generate_exclusions` and the existing `prepare_clustering_inputs.load_exclusions(path, scenario)`.
- Produces: an immutable prepared artifact created by existing `run_preparation(...)`.
- Guarantees: generated user exclusions are accepted without schema translation and the selected feature is absent from the standardized matrix while non-excluded rows and features remain available.

- [ ] **Step 1: Write a failing integration test that crosses the real loader and preprocessor**

Add these imports:

```python
import numpy as np

from scripts.cluster_preparation import prepare_clustering_inputs as preparation
from scripts.utils import clustering_preprocessing
```

Then add:

```python
class FeatureExclusionPreparationIntegrationTests(unittest.TestCase):
    def test_generated_json_is_loaded_and_excludes_the_selected_prepared_feature(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            roles_path = root / "feature_roles.csv"
            pd.DataFrame(
                [
                    {"column": "Per_extent", "role": "PF_FEATURE", "reason": ""},
                    {"column": "PFABC_NOR", "role": "PF_FEATURE", "reason": ""},
                    {
                        "column": "PF_Index_Risk_equal",
                        "role": "COMPOSITE_INDEX",
                        "reason": "",
                    },
                ]
            ).to_csv(roles_path, index=False)
            exclusions_path = root / "feature_exclusions.json"
            exclusions.generate_exclusions(
                scenario="PF",
                exclusion_text="PFABC_NOR",
                roles_path=roles_path,
                output_path=exclusions_path,
            )

            self.assertEqual(
                preparation.load_exclusions(exclusions_path, "PF"),
                ("PFABC_NOR",),
            )

            input_path = root / "high_risk.csv"
            pd.DataFrame(
                {
                    "fid": [10, 11, 12, 13],
                    "MS_ID": ["s0", "s1", "s2", "s3"],
                    "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6],
                    "Per_extent": [-1.0, 0.0, 1.0, np.nan],
                    "PFABC_NOR": [-0.8, -0.2, 0.2, 0.8],
                    "TFABC_NOR": [0.1, 0.2, 0.3, 0.4],
                }
            ).to_csv(input_path, index=False)
            artifact_directory = preparation.run_preparation(
                input_path=input_path,
                scenario="PF",
                output_root=root / "prepared",
                exclusion_path=exclusions_path,
                risk_column="PF_Index_Risk_equal",
                risk_operator=">",
                risk_threshold=0.25,
            )
            artifact = clustering_preprocessing.load_prepared_artifact(
                artifact_directory
            )

        self.assertEqual(artifact.feature_names, ("Per_extent",))
        self.assertNotIn("PFABC_NOR", artifact.standardized_features.columns)
        self.assertEqual(len(artifact.standardized_features), 4)
        self.assertFalse(artifact.standardized_features.isna().any().any())
        recorded = artifact.excluded_features.set_index("column")
        self.assertEqual(
            recorded.loc["PFABC_NOR", "reason"],
            "explicit_research_exclusion",
        )
```

- [ ] **Step 2: Run the integration test before changing production code**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.pearson_correlation.test_build_feature_exclusions.FeatureExclusionPreparationIntegrationTests -v
```

Expected: PASS if Tasks 1–2 satisfy the existing loader contract. If it fails, change only the generator’s payload or validation behavior; do not add a second exclusion schema or alter `prepare_clustering_inputs.py`.

- [ ] **Step 3: Run the adjacent preprocessing and correlation regression suites**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.pearson_correlation.test_build_feature_exclusions scripts.cluster_preparation.test_prepare_clustering_inputs scripts.utils.test_clustering_preprocessing scripts.pearson_correlation.test_correlation_analysis scripts.pearson_correlation.test_extract_high_columns tests.test_unified_clustering_workflow -v
```

Expected: all tests pass, including the legacy-export deprecation tests and raw AB lossR built-in exclusion tests.

- [ ] **Step 4: Commit the loader-to-preparation compatibility proof**

Run:

```powershell
git add scripts/pearson_correlation/test_build_feature_exclusions.py
git diff --cached --check
git commit -m "test: verify feature exclusion preparation bridge"
```

Expected: only the generator test module is staged and committed.

---

### Task 4: Document the active run order and perform full verification

**Files:**
- Modify: `README.md`
- Verify: `scripts/pearson_correlation/build_feature_exclusions.py`
- Verify: `scripts/pearson_correlation/test_build_feature_exclusions.py`
- Verify: all existing Python tests

**Interfaces:**
- Consumes: the approved CLI and existing v3 workflow.
- Produces: copyable PF and TF commands from correlation through preparation, plus explicit regeneration and legacy-tool guidance.
- Guarantees: a researcher following README passes the generated JSON into the one shared preprocessing boundary before PCA, K selection, K-means, or HDBSCAN.

- [ ] **Step 1: Update the confirmed workflow commands**

In `README.md`, insert the generator commands immediately after the two correlation commands and add `--exclusions` to both preparation commands:

```powershell
python scripts/pearson_correlation/correlation_analysis.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF
python scripts/pearson_correlation/correlation_analysis.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF
python scripts/pearson_correlation/build_feature_exclusions.py --scenario PF
python scripts/pearson_correlation/build_feature_exclusions.py --scenario TF
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF --exclusions outputs_correlation/pf/feature_exclusions.json
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF --exclusions outputs_correlation/tf/feature_exclusions.json
```

After the command block, add:

```markdown
The exclusion generator does not choose correlated variables automatically.
Review each scenario's correlation tables or plots first, then enter the full
comma-separated list once. A blank list is valid and keeps every eligible
feature. To replace an existing decision file, rerun the generator with
`--overwrite`.

`feature_exclusions.json` is the active bridge from correlation review to
PCA, K selection, K-means, and HDBSCAN. `extract_high_columns.py` remains a
legacy compatibility exporter; its `pf_for_PCA.csv` and `tf_for_PCA.csv`
files are not active PCA inputs.
```

Keep the existing immutable-artifact warning, but make the response to a changed decision explicit:

```markdown
Changing the source file, risk threshold, or feature exclusions requires a
new `--output-root` for `prepare_clustering_inputs.py`, followed by
regeneration of every downstream result from that new prepared artifact.
```

- [ ] **Step 2: Run the complete regression suite in `2026master`**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest discover -v
```

Expected: all discovered tests pass with no failures or errors.

- [ ] **Step 3: Compile every changed Python file and smoke-test direct help**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m py_compile scripts/pearson_correlation/build_feature_exclusions.py scripts/pearson_correlation/test_build_feature_exclusions.py
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python scripts/pearson_correlation/build_feature_exclusions.py --help
```

Expected: compilation exits with no output; help exits zero and shows `--scenario`, `--exclude`, `--roles`, `--output`, and `--overwrite`.

- [ ] **Step 4: Audit the final diff and confirm no generated research data changed**

Run:

```powershell
git status --short
git diff --check
git diff -- scripts/pearson_correlation/build_feature_exclusions.py scripts/pearson_correlation/test_build_feature_exclusions.py README.md
```

Expected: the feature contains only the generator, its tests, and README changes. Existing user-modified files under `data/`, `outputs_correlation/`, `outputs_preprocessing/`, `output_pca/`, and `output_kmeans/` remain unstaged and unchanged by this implementation.

- [ ] **Step 5: Commit the workflow documentation**

Run:

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: document feature exclusion workflow"
```

Expected: only `README.md` is staged and committed.

- [ ] **Step 6: Record final verification evidence**

Run:

```powershell
git status --short --branch
git log -4 --oneline
```

Expected: the implementation commits are visible, the branch remains `exp/cluster_modifyvariable_kmeans260727`, and all pre-existing user output/data changes remain outside these commits.
