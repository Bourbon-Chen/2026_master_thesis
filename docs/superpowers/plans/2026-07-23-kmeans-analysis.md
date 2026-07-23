# Interactive K-means Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a directly runnable, scenario-specific K-means analysis script that uses the prepared standardized variables unchanged and exports the complete reproducible experiment package defined in the approved design.

**Architecture:** Move the PF/TF feature lists into one shared configuration imported by PCA, K-selection, and the new `scripts/cluster_kmeans/kmeans_analysis.py` workflow. Keep the new workflow as a modular script with dataclasses for validated data, fitted clusters, reordered labels, tables, and metrics; write all outputs through a staging directory before an atomic final-directory rename.

**Tech Stack:** Python 3.11 project environment (`2026master.yml`), pandas, NumPy, scikit-learn, matplotlib, pathlib, unittest

## Global Constraints

- PF and TF run separately; one invocation analyzes exactly one scenario and one K.
- Use the existing ordered seven-variable PF/TF lists and one shared configuration source.
- Do not import or apply `StandardScaler`, `MinMaxScaler`, normalization, or imputation in the new K-means workflow.
- Configure K-means with `init="k-means++"`, `n_init=50`, `max_iter=500`, `random_state=42`, `algorithm="lloyd"`, and `tol=1e-4`.
- Use final directory names of `{scenario}_k{k}_v{n_features}_kmeanspp_n50_mi500_rs42`.
- Do not add timestamps, hashes, or run numbers to final directory names.
- Refuse to run when the final directory already exists; never overwrite it.
- Preserve all original CSV fields and row order in `clustered_data.csv`.
- Use PCA only after full-feature K-means for plotting and cluster-label ordering.
- Use matplotlib at 300 dpi and do not use seaborn.
- Write CSV, JSON, and text artifacts as UTF-8.
- Follow the approved design at `docs/superpowers/specs/2026-07-23-kmeans-analysis-design.md`.

## File Map

- Create `scripts/utils/scenario_features.py`: ordered shared scenario features, metadata, and default input paths.
- Modify `scripts/cluster_preparation/pca_analysis.py`: import shared definitions without changing PCA behavior.
- Modify `scripts/cluster_preparation/choose_k.py`: import shared definitions without changing K-selection behavior.
- Create `scripts/cluster_kmeans/__init__.py`: package marker and short module description.
- Create `scripts/cluster_kmeans/kmeans_analysis.py`: interactions, validation, clustering, metrics, plots, exports, and entry point.
- Create `scripts/cluster_kmeans/test_kmeans_analysis.py`: unit and end-to-end tests.

---

### Task 1: Centralize the PF/TF feature configuration

**Files:**
- Create: `scripts/utils/scenario_features.py`
- Create: `scripts/cluster_kmeans/__init__.py`
- Create: `scripts/cluster_kmeans/test_kmeans_analysis.py`
- Modify: `scripts/cluster_preparation/pca_analysis.py:6-55`
- Modify: `scripts/cluster_preparation/choose_k.py:6-67`

**Interfaces:**
- Produces: `PF_FEATURES`, `TF_FEATURES`, `PF_METADATA_COLUMNS`, `TF_METADATA_COLUMNS`, `DEFAULT_INPUT_PATHS`, and `SCENARIO_CONFIG`
- Consumers: existing `pca_analysis.py`, existing `choose_k.py`, and later K-means tasks

- [ ] **Step 1: Write the failing synchronization test**

Create `scripts/cluster_kmeans/test_kmeans_analysis.py` with:

```python
from __future__ import annotations

import unittest

from scripts.cluster_preparation import choose_k
from scripts.cluster_preparation import pca_analysis


class SharedScenarioConfigurationTests(unittest.TestCase):
    def test_pca_and_choose_k_share_the_same_feature_objects(self) -> None:
        from scripts.utils import scenario_features

        self.assertIs(pca_analysis.PF_FEATURES, scenario_features.PF_FEATURES)
        self.assertIs(pca_analysis.TF_FEATURES, scenario_features.TF_FEATURES)
        self.assertIs(choose_k.PF_FEATURES, scenario_features.PF_FEATURES)
        self.assertIs(choose_k.TF_FEATURES, scenario_features.TF_FEATURES)
        self.assertEqual(len(scenario_features.PF_FEATURES), 7)
        self.assertEqual(len(scenario_features.TF_FEATURES), 7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run in the activated project environment:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.SharedScenarioConfigurationTests -v
```

Expected: import failure because `scripts.utils.scenario_features` does not yet
exist.

- [ ] **Step 3: Add the shared configuration**

Create `scripts/utils/scenario_features.py`:

```python
"""Shared PF/TF feature definitions for cluster preparation and modelling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PF_FEATURES = (
    "Per_extent",
    "N_PFlossR_2kiw",
    "N_PFlossR_2kmw",
    "PFResident_lossR",
    "PFDaynight_lossR",
    "PFAB5k_NOR",
    "PFABC_NOR",
)
TF_FEATURES = (
    "Tem_extent",
    "N_TFlossR_2kiw",
    "N_TFlossR_2kmw",
    "TFResident_lossR",
    "TFDaynight_lossR",
    "TFAB5k_NOR",
    "TFABC_NOR",
)

PF_METADATA_COLUMNS = ("fid", "MS_ID", "PF_Index_Risk_equal")
TF_METADATA_COLUMNS = ("fid", "MS_ID", "TF_Index_Risk_equal")

DEFAULT_INPUT_PATHS: Dict[str, Path] = {
    "PF": PROJECT_ROOT / "outputs_correlation" / "pf" / "pf_for_PCA.csv",
    "TF": PROJECT_ROOT / "outputs_correlation" / "tf" / "tf_for_PCA.csv",
}

SCENARIO_CONFIG: Dict[str, Dict[str, Sequence[str]]] = {
    "PF": {"features": PF_FEATURES, "metadata": PF_METADATA_COLUMNS},
    "TF": {"features": TF_FEATURES, "metadata": TF_METADATA_COLUMNS},
}
```

Create `scripts/cluster_kmeans/__init__.py`:

```python
"""Interactive K-means analysis for prepared PF and TF street data."""
```

In both existing scripts, add `import sys`, keep their local `PROJECT_ROOT`,
insert it into `sys.path` only when needed for direct script execution, and
import the shared names:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.scenario_features import (
    DEFAULT_INPUT_PATHS,
    PF_FEATURES,
    PF_METADATA_COLUMNS,
    SCENARIO_CONFIG,
    TF_FEATURES,
    TF_METADATA_COLUMNS,
)
```

Delete the duplicate `DEFAULT_INPUT_PATHS`, feature, metadata, and
`SCENARIO_CONFIG` declarations from both scripts. Keep each script's local
output-root settings and all analytical functions unchanged.

- [ ] **Step 4: Run focused and affected tests and verify GREEN**

Run:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.SharedScenarioConfigurationTests scripts.cluster_preparation.test_pca_analysis scripts.cluster_preparation.test_choose_k -v
```

Expected: all tests pass with the existing PCA and choose-K behavior unchanged.

- [ ] **Step 5: Commit the shared configuration**

```powershell
git add scripts/utils/scenario_features.py scripts/cluster_kmeans/__init__.py scripts/cluster_kmeans/test_kmeans_analysis.py scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/choose_k.py
git commit -m "refactor: share clustering scenario features"
```

---

### Task 2: Implement interaction, CSV loading, and feature validation

**Files:**
- Create: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Consumes: `SCENARIO_CONFIG` and `DEFAULT_INPUT_PATHS` from Task 1
- Produces: `prompt_scenario() -> str`, `prompt_k(n_samples: int | None = None) -> int`, `prompt_csv_path(scenario: str) -> Path`, `load_dataset(path: Path) -> DataFrame`, `select_features(data: DataFrame, scenario: str) -> FeatureSelection`, `validate_features(data: DataFrame, selection: FeatureSelection) -> ValidatedData`

- [ ] **Step 1: Write failing tests for terminal normalization and paths**

Add these imports to the test module:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.cluster_kmeans import kmeans_analysis
```

Add:

```python
class PromptTests(unittest.TestCase):
    def test_prompt_scenario_retries_and_accepts_aliases(self) -> None:
        with patch("builtins.input", side_effect=["bad", " 2 "]), patch(
            "builtins.print"
        ) as mocked_print:
            self.assertEqual(kmeans_analysis.prompt_scenario(), "TF")

        mocked_print.assert_any_call("Invalid selection. Enter 1/pf or 2/tf.")

    def test_prompt_k_retries_for_integer_and_sample_limit(self) -> None:
        with patch("builtins.input", side_effect=["2.0", "1", "5", "3"]), patch(
            "builtins.print"
        ):
            self.assertEqual(kmeans_analysis.prompt_k(n_samples=5), 3)

    def test_prompt_csv_path_strips_quotes_and_retries(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            valid_path = Path(temporary_directory) / "prepared data.csv"
            valid_path.write_text("a,b\n1,2\n", encoding="utf-8")
            with patch(
                "builtins.input",
                side_effect=["'missing.csv'", f'"{valid_path}"'],
            ), patch("builtins.print"):
                selected = kmeans_analysis.prompt_csv_path("PF")

        self.assertEqual(selected, valid_path.resolve())
```

- [ ] **Step 2: Write failing tests for feature selection and validation**

Add a reusable fixture and tests:

```python
def prepared_pf_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fid": [1, 2, 3, 4, 5, 6],
            "MS_ID": [101, 102, 103, 104, 105, 106],
            "Per_extent": [-1.0, -0.9, -0.8, 0.8, 0.9, 1.0],
            "N_PFlossR_2kiw": [-1.1, -1.0, np.nan, 0.9, 1.0, 1.1],
            "N_PFlossR_2kmw": [-0.9, -0.8, -0.7, 0.7, 0.8, 0.9],
            "PFResident_lossR": [-1.2, -1.0, -0.8, 0.8, 1.0, 1.2],
            "PFDaynight_lossR": [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0],
            "PFAB5k_NOR": [-0.7, -0.6, -0.5, 0.5, 0.6, 0.7],
            "PFABC_NOR": [-0.8, -0.7, -0.6, 0.6, 0.7, 0.8],
            "grid_density": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "PF_Index_Risk_equal": [0.2] * 6,
            "street_name": ["a", "b", "c", "d", "e", "f"],
        }
    )


class FeatureValidationTests(unittest.TestCase):
    def test_selects_only_shared_pf_features_and_exact_ids(self) -> None:
        data = prepared_pf_frame()
        selection = kmeans_analysis.select_features(data, "PF")

        self.assertEqual(
            list(selection.feature_names),
            list(kmeans_analysis.PF_FEATURES),
        )
        self.assertEqual(list(selection.id_columns), ["fid", "MS_ID"])
        self.assertNotIn("grid_density", selection.id_columns)
        reasons = {item.column: item.reason for item in selection.exclusions}
        self.assertEqual(reasons["grid_density"], "not in scenario feature configuration")
        self.assertEqual(reasons["PF_Index_Risk_equal"], "excluded keyword: index")

    def test_validate_features_drops_missing_rows_without_imputation(self) -> None:
        data = prepared_pf_frame()
        validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )

        self.assertEqual(len(validated.original_data), 6)
        self.assertEqual(len(validated.features), 5)
        self.assertFalse(validated.valid_mask.iloc[2])
        self.assertEqual(validated.missing_counts["N_PFlossR_2kiw"], 1)
        self.assertNotIn(0.0, validated.features["N_PFlossR_2kiw"].tolist())

    def test_excludes_constant_and_infinite_configured_columns(self) -> None:
        data = prepared_pf_frame()
        data["PFAB5k_NOR"] = 1.0
        data.loc[0, "PFABC_NOR"] = np.inf

        validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )

        self.assertNotIn("PFAB5k_NOR", validated.feature_names)
        self.assertNotIn("PFABC_NOR", validated.feature_names)
        actions = {issue.column: issue.action for issue in validated.issues}
        self.assertEqual(actions["PFAB5k_NOR"], "excluded feature")
        self.assertEqual(actions["PFABC_NOR"], "excluded feature")

    def test_rejects_missing_nonnumeric_and_reserved_columns(self) -> None:
        cases = [
            (
                prepared_pf_frame().drop(columns=["Per_extent"]),
                "Missing configured feature: Per_extent",
            ),
            (
                prepared_pf_frame().assign(Per_extent=["x"] * 6),
                "Configured feature is not numeric: Per_extent",
            ),
            (
                prepared_pf_frame().assign(kmeans_cluster=[1] * 6),
                "Reserved output column already exists: kmeans_cluster",
            ),
        ]
        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    kmeans_analysis.DataValidationError, message
                ):
                    kmeans_analysis.select_features(data, "PF")
```

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.PromptTests scripts.cluster_kmeans.test_kmeans_analysis.FeatureValidationTests -v
```

Expected: import failure because `kmeans_analysis.py` does not yet exist.

- [ ] **Step 4: Implement interaction and validation foundations**

Create `scripts/cluster_kmeans/kmeans_analysis.py` with these constants and
interfaces:

```python
#!/usr/bin/env python3
"""Run one reproducible K-means experiment on prepared PF or TF data."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.scenario_features import (
    DEFAULT_INPUT_PATHS,
    PF_FEATURES,
    SCENARIO_CONFIG,
    TF_FEATURES,
)

OUTPUT_ROOT = PROJECT_ROOT / "output_kmeans"
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
KMEANS_RANDOM_STATE = 42
KMEANS_ALGORITHM = "lloyd"
KMEANS_TOL = 1e-4
OUTPUT_DPI = 300

EXCLUSION_KEYWORDS = (
    "index",
    "risk",
    "cluster",
    "label",
    "type",
    "typology",
    "intervention",
    "score",
    "geometry",
)
EXACT_ID_NAMES = {
    "id",
    "fid",
    "gid",
    "objectid",
    "ogc_fid",
    "ms_id",
    "street_id",
    "segment_id",
}


class KMeansAnalysisError(RuntimeError):
    """Base exception for expected user-facing workflow errors."""


class DataValidationError(KMeansAnalysisError):
    """Raised when prepared input cannot safely enter K-means."""


@dataclass(frozen=True)
class ColumnExclusion:
    column: str
    reason: str


@dataclass(frozen=True)
class DataQualityIssue:
    issue_type: str
    column: str
    count: int
    action: str
    details: str


@dataclass
class FeatureSelection:
    feature_names: tuple[str, ...]
    exclusions: list[ColumnExclusion]
    id_columns: tuple[str, ...]
    issues: list[DataQualityIssue]


@dataclass
class ValidatedData:
    original_data: pd.DataFrame
    features: pd.DataFrame
    valid_mask: pd.Series
    feature_names: tuple[str, ...]
    exclusions: list[ColumnExclusion]
    id_columns: tuple[str, ...]
    missing_counts: dict[str, int]
    issues: list[DataQualityIssue]


def normalize_scenario(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"1": "PF", "pf": "PF", "2": "TF", "tf": "TF"}
    if normalized not in aliases:
        raise ValueError("Invalid selection. Enter 1/pf or 2/tf.")
    return aliases[normalized]


def prompt_scenario() -> str:
    while True:
        print("Select flood scenario:")
        print("1. PF")
        print("2. TF")
        try:
            return normalize_scenario(input().strip())
        except ValueError as error:
            print(str(error))


def prompt_k(n_samples: int | None = None) -> int:
    while True:
        value = input("Enter the number of clusters K: ").strip()
        try:
            k = int(value)
        except ValueError:
            print("K must be an integer greater than or equal to 2.")
            continue
        if str(k) != value and value not in {f"+{k}", f"-{abs(k)}"}:
            print("K must be an integer greater than or equal to 2.")
            continue
        if k < 2:
            print("K must be an integer greater than or equal to 2.")
            continue
        if n_samples is not None and k >= n_samples:
            print(f"K must be smaller than the valid sample count ({n_samples}).")
            continue
        return k


def _strip_matching_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1].strip()
    return stripped


def prompt_csv_path(scenario: str) -> Path:
    default_path = DEFAULT_INPUT_PATHS[scenario]
    default_name = default_path.name
    while True:
        entered = input(f"Enter CSV path [default: {default_name}]: ")
        candidate = default_path if not entered.strip() else Path(
            _strip_matching_quotes(entered)
        ).expanduser()
        resolved = candidate.resolve()
        if not resolved.exists():
            print(f"CSV file not found: {resolved}")
            continue
        if not resolved.is_file():
            print(f"CSV path is not a file: {resolved}")
            continue
        return resolved


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except FileNotFoundError as error:
        raise KMeansAnalysisError(f"CSV file not found: {path}") from error
    except PermissionError as error:
        raise KMeansAnalysisError(f"Permission denied while reading: {path}") from error
    except pd.errors.EmptyDataError as error:
        raise KMeansAnalysisError(f"CSV is empty: {path}") from error
    except pd.errors.ParserError as error:
        raise KMeansAnalysisError(f"CSV could not be parsed: {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise KMeansAnalysisError(f"CSV is not valid UTF-8 text: {path}") from error


def is_id_column(name: str) -> bool:
    normalized = name.strip().lower()
    return (
        normalized in EXACT_ID_NAMES
        or normalized.endswith("_id")
        or normalized.startswith("id_")
    )


def _retained_exclusion_reason(column: str, series: pd.Series) -> str:
    if is_id_column(column):
        return "identifier field"
    if not is_numeric_dtype(series):
        return "non-numeric retained field"
    lowered = column.lower()
    for keyword in EXCLUSION_KEYWORDS:
        if keyword in lowered:
            return f"excluded keyword: {keyword}"
    return "not in scenario feature configuration"
```

Complete `select_features()` and `validate_features()` so they implement the
tests exactly:

```python
def select_features(data: pd.DataFrame, scenario: str) -> FeatureSelection:
    if "kmeans_cluster" in data.columns:
        raise DataValidationError(
            "Reserved output column already exists: kmeans_cluster"
        )
    configured = tuple(SCENARIO_CONFIG[scenario]["features"])
    for column in configured:
        if column not in data.columns:
            raise DataValidationError(f"Missing configured feature: {column}")
        if not is_numeric_dtype(data[column]):
            raise DataValidationError(
                f"Configured feature is not numeric: {column}"
            )

    exclusions = [
        ColumnExclusion(column, _retained_exclusion_reason(column, data[column]))
        for column in data.columns
        if column not in configured
    ]
    id_columns = tuple(column for column in data.columns if is_id_column(column))
    return FeatureSelection(configured, exclusions, id_columns, [])


def validate_features(
    data: pd.DataFrame, selection: FeatureSelection
) -> ValidatedData:
    active = list(selection.feature_names)
    exclusions = list(selection.exclusions)
    issues = list(selection.issues)

    for column in list(active):
        series = data[column].astype(float)
        finite_nonmissing = series.dropna()
        if series.isna().all():
            reason = "all values are missing"
            issue_count = int(series.isna().sum())
        elif np.isinf(series.to_numpy(dtype=float)).any():
            reason = "contains infinity"
            issue_count = int(np.isinf(series.to_numpy(dtype=float)).sum())
        elif finite_nonmissing.nunique() <= 1:
            reason = "constant feature"
            issue_count = int(finite_nonmissing.size)
        else:
            continue
        active.remove(column)
        exclusions.append(ColumnExclusion(column, reason))
        issues.append(
            DataQualityIssue(reason, column, issue_count, "excluded feature", reason)
        )

    while active:
        frame = data.loc[:, active].astype(float)
        valid_mask = frame.notna().all(axis=1)
        valid_frame = frame.loc[valid_mask]
        post_drop_constants = [
            column for column in active if valid_frame[column].nunique() <= 1
        ]
        if not post_drop_constants:
            break
        for column in post_drop_constants:
            active.remove(column)
            exclusions.append(
                ColumnExclusion(column, "constant after missing-row removal")
            )
            issues.append(
                DataQualityIssue(
                    "constant_after_row_removal",
                    column,
                    int(len(valid_frame)),
                    "excluded feature",
                    "Only one valid value remained after complete-case filtering.",
                )
            )

    if not active:
        raise DataValidationError("No usable clustering features remain.")

    frame = data.loc[:, active].astype(float)
    missing_counts = {
        column: int(count)
        for column, count in frame.isna().sum().items()
        if int(count) > 0
    }
    valid_mask = frame.notna().all(axis=1)
    valid_frame = frame.loc[valid_mask].copy()
    if not np.isfinite(valid_frame.to_numpy(dtype=float)).all():
        raise DataValidationError("Non-finite values remain after feature validation.")
    removed_count = int((~valid_mask).sum())
    for column, count in missing_counts.items():
        issues.append(
            DataQualityIssue(
                "missing_values",
                column,
                count,
                "removed affected rows",
                "No imputation was applied.",
            )
        )
    if removed_count:
        issues.append(
            DataQualityIssue(
                "invalid_rows",
                "",
                removed_count,
                "excluded from K-means",
                "Rows remain in clustered_data.csv with an empty label.",
            )
        )
    if len(valid_frame) < 3:
        raise DataValidationError(
            "At least 3 valid samples are required because 2 <= K < n_samples."
        )
    return ValidatedData(
        original_data=data.copy(),
        features=valid_frame,
        valid_mask=valid_mask,
        feature_names=tuple(active),
        exclusions=exclusions,
        id_columns=selection.id_columns,
        missing_counts=missing_counts,
        issues=issues,
    )
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 3 command again.

Expected: all prompt and validation tests pass.

- [ ] **Step 6: Commit the input boundary**

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: validate prepared k-means input"
```

---

### Task 3: Fit K-means and stabilize cluster labels

**Files:**
- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Consumes: `ValidatedData.features`, user-selected K, fixed K-means constants
- Produces: `run_kmeans(values: ndarray, k: int) -> FittedKMeans`, `reorder_cluster_labels(fitted: FittedKMeans, values: ndarray, feature_names: Sequence[str]) -> ClusterOrdering`

- [ ] **Step 1: Write failing model and remapping tests**

Add:

```python
class ModelTests(unittest.TestCase):
    def test_run_kmeans_uses_fixed_parameters_and_distances(self) -> None:
        values = np.array(
            [[-1.1, -1.0], [-0.9, -1.2], [0.9, 1.0], [1.1, 1.2]],
            dtype=float,
        )
        fitted = kmeans_analysis.run_kmeans(values, 2)

        params = fitted.model.get_params()
        self.assertEqual(params["init"], "k-means++")
        self.assertEqual(params["n_init"], 50)
        self.assertEqual(params["max_iter"], 500)
        self.assertEqual(params["random_state"], 42)
        self.assertEqual(params["algorithm"], "lloyd")
        self.assertEqual(fitted.assigned_distances.shape, (4,))
        self.assertGreaterEqual(fitted.inertia, 0.0)

    def test_reorder_uses_centroid_pc1_and_preserves_membership(self) -> None:
        values = np.array(
            [[-2.0, -1.8], [-1.8, -2.0], [1.8, 2.0], [2.0, 1.8]],
            dtype=float,
        )
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordered = kmeans_analysis.reorder_cluster_labels(
            fitted, values, ("a", "b")
        )

        original_same = (
            fitted.original_labels[:, None] == fitted.original_labels[None, :]
        )
        reordered_same = (
            ordered.reordered_labels[:, None] == ordered.reordered_labels[None, :]
        )
        np.testing.assert_array_equal(original_same, reordered_same)
        self.assertEqual(
            ordered.mapping["reordered_cluster_id"].tolist(), [1, 2]
        )
        self.assertTrue(
            ordered.mapping["ordering_value"].is_monotonic_increasing
        )
        self.assertEqual(ordered.ordering_method, "centroid PC1 projection")

    def test_reorder_falls_back_to_centroid_mean_for_one_feature(self) -> None:
        values = np.array([[-2.0], [-1.8], [1.8], [2.0]], dtype=float)
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordered = kmeans_analysis.reorder_cluster_labels(fitted, values, ("a",))

        self.assertEqual(ordered.ordering_method, "centroid mean")
        self.assertIsNone(ordered.pca_scores)
        self.assertTrue(any("PCA" in warning for warning in ordered.warnings))
```

- [ ] **Step 2: Run the focused tests and verify RED**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.ModelTests -v
```

Expected: failures because the fitted-model and ordering interfaces do not
exist.

- [ ] **Step 3: Implement model fitting and compatibility checks**

Add imports for `inspect`, `time`, and `warnings`, then add:

```python
class CompatibilityError(KMeansAnalysisError):
    """Raised when installed scikit-learn cannot honor configured settings."""


@dataclass
class FittedKMeans:
    model: object
    original_labels: np.ndarray
    assigned_distances: np.ndarray
    inertia: float
    n_iter: int
    converged: bool
    fit_seconds: float
    warnings: list[str]


@dataclass
class ClusterOrdering:
    reordered_labels: np.ndarray
    mapping: pd.DataFrame
    reordered_centroids: pd.DataFrame
    ordering_method: str
    pca_scores: np.ndarray | None
    explained_variance_ratio: np.ndarray | None
    warnings: list[str]


def run_kmeans(values: np.ndarray, k: int) -> FittedKMeans:
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.cluster import KMeans
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as error:
        raise CompatibilityError(
            "scikit-learn is required. Activate the environment from 2026master.yml."
        ) from error

    configured = {
        "n_clusters": k,
        "init": KMEANS_INIT,
        "n_init": KMEANS_N_INIT,
        "max_iter": KMEANS_MAX_ITER,
        "random_state": KMEANS_RANDOM_STATE,
        "algorithm": KMEANS_ALGORITHM,
        "tol": KMEANS_TOL,
    }
    supported = set(inspect.signature(KMeans).parameters)
    unsupported = sorted(set(configured) - supported)
    if unsupported:
        raise CompatibilityError(
            f"scikit-learn {sklearn_version} does not support configured "
            f"KMeans arguments: {', '.join(unsupported)}"
        )

    model = KMeans(**configured)
    captured_messages: list[str] = []
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        labels = model.fit_predict(values)
    fit_seconds = time.perf_counter() - started
    captured_messages.extend(str(item.message) for item in caught)
    all_distances = model.transform(values)
    assigned = all_distances[np.arange(len(values)), labels]
    converged = int(model.n_iter_) < KMEANS_MAX_ITER
    if not converged:
        captured_messages.append(
            f"K-means reached max_iter={KMEANS_MAX_ITER}; convergence is not confirmed."
        )
    return FittedKMeans(
        model=model,
        original_labels=np.asarray(labels, dtype=int),
        assigned_distances=np.asarray(assigned, dtype=float),
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
        converged=converged,
        fit_seconds=float(fit_seconds),
        warnings=captured_messages,
    )
```

- [ ] **Step 4: Implement PC1 ordering and fallback**

Add:

```python
def reorder_cluster_labels(
    fitted: FittedKMeans,
    values: np.ndarray,
    feature_names: Sequence[str],
) -> ClusterOrdering:
    warnings_list: list[str] = []
    pca_scores: np.ndarray | None = None
    explained: np.ndarray | None = None
    centroids = np.asarray(fitted.model.cluster_centers_, dtype=float)

    if min(values.shape[0], values.shape[1]) >= 2:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            pca_scores = pca.fit_transform(values)
            ordering_values = pca.transform(centroids)[:, 0]
            explained = np.asarray(pca.explained_variance_ratio_, dtype=float)
            method = "centroid PC1 projection"
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings_list.append(f"PCA unavailable; used centroid mean: {error}")
            ordering_values = centroids.mean(axis=1)
            method = "centroid mean"
    else:
        warnings_list.append(
            "PCA requires at least two samples and two features; used centroid mean."
        )
        ordering_values = centroids.mean(axis=1)
        method = "centroid mean"

    original_order = sorted(
        range(len(centroids)),
        key=lambda label: (float(ordering_values[label]), int(label)),
    )
    remap = {
        original_label: reordered_id
        for reordered_id, original_label in enumerate(original_order, start=1)
    }
    reordered_labels = np.array(
        [remap[int(label)] for label in fitted.original_labels], dtype=int
    )
    mapping = pd.DataFrame(
        {
            "sklearn_original_label": original_order,
            "reordered_cluster_id": range(1, len(original_order) + 1),
            "ordering_value": [
                float(ordering_values[label]) for label in original_order
            ],
        }
    )
    reordered_centroids = pd.DataFrame(
        centroids[original_order], columns=list(feature_names)
    )
    reordered_centroids.insert(
        0, "cluster_id", np.arange(1, len(original_order) + 1)
    )
    return ClusterOrdering(
        reordered_labels=reordered_labels,
        mapping=mapping,
        reordered_centroids=reordered_centroids,
        ordering_method=method,
        pca_scores=pca_scores,
        explained_variance_ratio=explained,
        warnings=warnings_list,
    )
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Step 2 command again.

Expected: all model and label-ordering tests pass.

- [ ] **Step 6: Commit modelling**

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: fit and reorder k-means clusters"
```

---

### Task 4: Calculate retained labels, profiles, distances, and metrics

**Files:**
- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Consumes: `ValidatedData`, `FittedKMeans`, and `ClusterOrdering`
- Produces: `calculate_cluster_profiles(...) -> AnalysisTables`, `calculate_metrics(...) -> MetricResults`

- [ ] **Step 1: Write failing table and metric tests**

Add:

```python
class AnalysisTableTests(unittest.TestCase):
    def setUp(self) -> None:
        data = prepared_pf_frame()
        self.validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )
        values = self.validated.features.to_numpy(dtype=float)
        self.fitted = kmeans_analysis.run_kmeans(values, 2)
        self.ordered = kmeans_analysis.reorder_cluster_labels(
            self.fitted, values, self.validated.feature_names
        )

    def test_profiles_preserve_all_rows_and_blank_invalid_label(self) -> None:
        tables = kmeans_analysis.calculate_cluster_profiles(
            self.validated, self.fitted, self.ordered
        )

        self.assertEqual(len(tables.clustered_data), 6)
        self.assertTrue(
            pd.isna(tables.clustered_data.loc[2, "kmeans_cluster"])
        )
        self.assertEqual(len(tables.cluster_summary), 2)
        self.assertAlmostEqual(
            tables.cluster_summary["sample_percentage"].sum(), 100.0
        )
        self.assertEqual(
            len(tables.feature_profile),
            2 * len(self.validated.feature_names),
        )
        self.assertEqual(len(tables.sample_distances), 5)
        self.assertEqual(
            list(tables.sample_distances.columns),
            [
                "fid",
                "MS_ID",
                "kmeans_cluster",
                "distance_to_assigned_centroid",
            ],
        )

    def test_calculate_metrics_returns_all_requested_values(self) -> None:
        values = self.validated.features.to_numpy(dtype=float)
        result = kmeans_analysis.calculate_metrics(
            values, self.ordered.reordered_labels
        )

        self.assertEqual(
            set(result.values),
            {
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
            },
        )
        self.assertTrue(all(value is not None for value in result.values.values()))
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.AnalysisTableTests -v
```

Expected: failures because table and metric calculators do not exist.

- [ ] **Step 3: Implement analytical tables**

Add dataclasses:

```python
@dataclass
class AnalysisTables:
    clustered_data: pd.DataFrame
    cluster_summary: pd.DataFrame
    centroids: pd.DataFrame
    feature_profile: pd.DataFrame
    sample_distances: pd.DataFrame
    label_mapping: pd.DataFrame
    data_quality_issues: pd.DataFrame


@dataclass
class MetricResults:
    values: dict[str, float | None]
    warnings: list[str]
```

Implement `calculate_cluster_profiles()` with positional assignment so duplicate
source index values cannot disturb original row order:

```python
def calculate_cluster_profiles(
    validated: ValidatedData,
    fitted: FittedKMeans,
    ordering: ClusterOrdering,
) -> AnalysisTables:
    valid_positions = np.flatnonzero(validated.valid_mask.to_numpy())
    full_labels = pd.array([pd.NA] * len(validated.original_data), dtype="Int64")
    full_labels[valid_positions] = ordering.reordered_labels
    clustered = validated.original_data.copy()
    clustered["kmeans_cluster"] = full_labels

    summaries = []
    profiles = []
    for cluster_id in range(1, len(ordering.mapping) + 1):
        member_mask = ordering.reordered_labels == cluster_id
        distances = fitted.assigned_distances[member_mask]
        count = int(member_mask.sum())
        summaries.append(
            {
                "cluster_id": cluster_id,
                "sample_count": count,
                "sample_percentage": count / len(ordering.reordered_labels) * 100.0,
                "distance_to_centroid_mean": float(np.mean(distances)) if count else np.nan,
                "distance_to_centroid_median": float(np.median(distances)) if count else np.nan,
                "distance_to_centroid_std": float(np.std(distances, ddof=0)) if count else np.nan,
                "distance_to_centroid_max": float(np.max(distances)) if count else np.nan,
            }
        )
        members = validated.features.iloc[np.flatnonzero(member_mask)]
        for feature in validated.feature_names:
            values = members[feature]
            profiles.append(
                {
                    "cluster_id": cluster_id,
                    "feature": feature,
                    "mean": float(values.mean()) if count else np.nan,
                    "median": float(values.median()) if count else np.nan,
                    "std": float(values.std(ddof=0)) if count else np.nan,
                    "min": float(values.min()) if count else np.nan,
                    "max": float(values.max()) if count else np.nan,
                    "q25": float(values.quantile(0.25)) if count else np.nan,
                    "q75": float(values.quantile(0.75)) if count else np.nan,
                }
            )

    source_valid = validated.original_data.iloc[valid_positions]
    distance_table = source_valid.loc[:, list(validated.id_columns)].reset_index(
        drop=True
    )
    distance_table["kmeans_cluster"] = ordering.reordered_labels
    distance_table["distance_to_assigned_centroid"] = fitted.assigned_distances
    issues = pd.DataFrame(
        [
            {
                "issue_type": issue.issue_type,
                "column": issue.column,
                "count": issue.count,
                "action": issue.action,
                "details": issue.details,
            }
            for issue in validated.issues
        ],
        columns=["issue_type", "column", "count", "action", "details"],
    )
    return AnalysisTables(
        clustered_data=clustered,
        cluster_summary=pd.DataFrame(summaries),
        centroids=ordering.reordered_centroids.copy(),
        feature_profile=pd.DataFrame(profiles),
        sample_distances=distance_table,
        label_mapping=ordering.mapping.copy(),
        data_quality_issues=issues,
    )
```

- [ ] **Step 4: Implement independently guarded metrics**

```python
def calculate_metrics(
    values: np.ndarray, labels: np.ndarray
) -> MetricResults:
    try:
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
    except ImportError as error:
        raise CompatibilityError(
            "scikit-learn metrics are unavailable in the active environment."
        ) from error

    calculators = {
        "silhouette_score": lambda: silhouette_score(values, labels),
        "calinski_harabasz_score": lambda: calinski_harabasz_score(values, labels),
        "davies_bouldin_score": lambda: davies_bouldin_score(values, labels),
    }
    results: dict[str, float | None] = {}
    warning_messages: list[str] = []
    for name, calculator in calculators.items():
        try:
            results[name] = float(calculator())
        except ValueError as error:
            results[name] = None
            warning_messages.append(f"{name} unavailable: {error}")
    return MetricResults(results, warning_messages)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command again.

Expected: both analysis-table tests pass.

- [ ] **Step 6: Commit analytical outputs**

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: calculate k-means profiles and metrics"
```

---

### Task 5: Generate all required matplotlib figures

**Files:**
- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Produces: `generate_pca_plot()`, `generate_cluster_size_plot()`, `generate_centroid_heatmap()`, and `generate_profile_plot()`
- Each plotting function accepts prepared analytical objects plus an explicit `Path` and closes its figure.

- [ ] **Step 1: Write failing plot tests**

Add:

```python
class PlotTests(AnalysisTableTests):
    def test_all_required_plots_are_written(self) -> None:
        tables = kmeans_analysis.calculate_cluster_profiles(
            self.validated, self.fitted, self.ordered
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [
                root / "pca_cluster_scatter.png",
                root / "cluster_size_bar.png",
                root / "cluster_centroid_heatmap.png",
                root / "cluster_feature_profiles.png",
            ]
            kmeans_analysis.generate_pca_plot(
                self.ordered, "PF", 2, self.validated.feature_names, paths[0]
            )
            kmeans_analysis.generate_cluster_size_plot(
                tables.cluster_summary, paths[1]
            )
            kmeans_analysis.generate_centroid_heatmap(
                tables.centroids, self.validated.feature_names, paths[2]
            )
            kmeans_analysis.generate_profile_plot(
                tables.feature_profile, self.validated.feature_names, paths[3]
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_pca_unavailable_still_writes_placeholder(self) -> None:
        values = np.array([[-2.0], [-1.8], [1.8], [2.0]], dtype=float)
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordered = kmeans_analysis.reorder_cluster_labels(fitted, values, ("a",))
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pca_cluster_scatter.png"
            kmeans_analysis.generate_pca_plot(ordered, "PF", 2, ("a",), path)
            self.assertGreater(path.stat().st_size, 0)
```

- [ ] **Step 2: Run plot tests and verify RED**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.PlotTests -v
```

Expected: failures because the four plot functions do not exist.

- [ ] **Step 3: Add the non-interactive matplotlib setup and PCA/size plots**

Add before importing pyplot:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Implement `generate_pca_plot()` and `generate_cluster_size_plot()` with:

- one discrete color per reordered cluster;
- PC1/PC2 explained-variance labels;
- scenario/K/feature-count title;
- a projection-only figure note;
- a text placeholder when PCA data is absent;
- count and percentage annotations above every size bar;
- `figure.tight_layout()`, `savefig(..., dpi=OUTPUT_DPI)`, and
  `plt.close(figure)` in each function.

Use this exact placeholder branch:

```python
if ordering.pca_scores is None or ordering.explained_variance_ratio is None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        "Two-component PCA is unavailable.\n"
        "K-means was still fitted in the full standardized feature space.",
        ha="center",
        va="center",
    )
    figure.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(figure)
    return
```

- [ ] **Step 4: Add centroid heatmap and feature-profile plots**

Implement the heatmap with a symmetric range:

```python
values = centroids.loc[:, list(feature_names)].to_numpy(dtype=float)
limit = max(float(np.nanmax(np.abs(values))), 1e-12)
image = axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
```

Write every value in its cell, use white annotation text when
`abs(value) > limit * 0.55`, label rows as `Cluster N`, rotate feature labels,
add `figure.colorbar(image, ax=axis, label="Standardized centroid")`, and size
the canvas from feature count and maximum label length.

Implement the profile plot by selecting each cluster from the long-form
profile table, plotting its ordered `mean` values with markers, adding
`axis.axhline(0.0, ...)`, readable labels, a legend, and grid.

- [ ] **Step 5: Run plot tests and verify GREEN**

Run the Step 2 command again.

Expected: both tests pass and four non-empty PNG files are produced in the
temporary directory.

- [ ] **Step 6: Commit visualizations**

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: visualize k-means cluster results"
```

---

### Task 6: Build safe output paths and export every artifact

**Files:**
- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Produces: `build_output_directory() -> Path`, `staged_output_directory()` context manager, `export_results()`, `build_run_config()`, and `build_run_report()`
- Final output is renamed into place only after every staged artifact succeeds.

- [ ] **Step 1: Write failing naming, collision, and export tests**

Add:

```python
class ExportTests(AnalysisTableTests):
    def test_build_output_directory_is_readable_and_refuses_existing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = kmeans_analysis.build_output_directory(root, "TF", 4, 7)
            self.assertEqual(
                target.name, "tf_k4_v7_kmeanspp_n50_mi500_rs42"
            )
            target.mkdir()
            with self.assertRaisesRegex(
                kmeans_analysis.OutputDirectoryExistsError,
                "move or rename",
            ):
                kmeans_analysis.build_output_directory(root, "TF", 4, 7)

    def test_export_results_writes_required_utf8_files(self) -> None:
        tables = kmeans_analysis.calculate_cluster_profiles(
            self.validated, self.fitted, self.ordered
        )
        metrics = kmeans_analysis.calculate_metrics(
            self.validated.features.to_numpy(dtype=float),
            self.ordered.reordered_labels,
        )
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            config = kmeans_analysis.build_run_config(
                scenario="PF",
                input_path=Path("pf.csv").resolve(),
                output_path=(output / "final").resolve(),
                k=2,
                validated=self.validated,
                fitted=self.fitted,
                ordering=self.ordered,
                warnings_list=[],
                started_at="2026-07-23T12:00:00+02:00",
                ended_at="2026-07-23T12:00:01+02:00",
            )
            report = kmeans_analysis.build_run_report(
                config, tables.cluster_summary, metrics.values, 1.0
            )
            kmeans_analysis.export_results(
                output,
                tables,
                metrics,
                config,
                report,
                runtime_seconds=1.0,
            )

            required = set(kmeans_analysis.TABULAR_TEXT_OUTPUTS)
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            saved_config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(saved_config["standard_scaler_applied"])
            self.assertEqual(saved_config["feature_list"], list(self.validated.feature_names))
```

Add `import json` to the test module.

- [ ] **Step 2: Run export tests and verify RED**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.ExportTests -v
```

Expected: failures because output helpers and exporters do not exist.

- [ ] **Step 3: Implement directory protection and staging**

Add `contextlib`, `json`, `platform`, `shutil`, `tempfile`, and datetime imports.
Define:

```python
class OutputDirectoryExistsError(KMeansAnalysisError):
    """Raised when a readable final experiment directory already exists."""


def build_output_directory(
    output_root: Path, scenario: str, k: int, n_features: int
) -> Path:
    name = (
        f"{scenario.lower()}_k{k}_v{n_features}_"
        f"kmeanspp_n{KMEANS_N_INIT}_mi{KMEANS_MAX_ITER}_"
        f"rs{KMEANS_RANDOM_STATE}"
    )
    target = Path(output_root) / name
    if target.exists():
        raise OutputDirectoryExistsError(
            f"Output folder already exists: {target}. "
            "Please move or rename the existing result before running again."
        )
    return target
```

Implement `staged_output_directory(final_path)` as a context manager:

1. Create `final_path.parent`.
2. Recheck `final_path`.
3. Create a temporary directory with `tempfile.mkdtemp()` under that parent.
4. Yield the staging `Path`.
5. Recheck `final_path` and rename the staging directory to it.
6. In `finally`, remove only the still-existing, resolved staging path whose
   parent equals the resolved output root.

Catch `OSError` around directory creation/rename and raise
`KMeansAnalysisError` naming the affected path.

- [ ] **Step 4: Implement config, report, and UTF-8 exports**

Define:

```python
TABULAR_TEXT_OUTPUTS = (
    "clustered_data.csv",
    "cluster_summary.csv",
    "cluster_centroids_standardized.csv",
    "cluster_feature_profile.csv",
    "sample_distances.csv",
    "model_metrics.csv",
    "cluster_label_mapping.csv",
    "data_quality_issues.csv",
    "run_config.json",
    "run_report.txt",
)
PLOT_OUTPUTS = (
    "pca_cluster_scatter.png",
    "cluster_size_bar.png",
    "cluster_centroid_heatmap.png",
    "cluster_feature_profiles.png",
)
ALL_OUTPUTS = TABULAR_TEXT_OUTPUTS + PLOT_OUTPUTS
```

`build_run_config()` must return a JSON-serializable dictionary with all fields
from the approved design. Convert exclusions and issues to ordinary
dictionaries, paths to absolute strings, version values to strings, NumPy
numbers to Python numbers, and PCA explained variance to percentages or
`None`. Import `sklearn.__version__` only inside this function.

`build_run_report()` must produce a newline-terminated report with scenario,
paths, K, feature list, sample counts, excluded columns/reasons, missing
treatment, K-means settings, label-ordering rule, cluster counts/percentages,
metrics, PCA variance, runtime, warnings, every filename in `ALL_OUTPUTS`, and
this exact statement:

```text
The input data had already been standardized with StandardScaler.
This script did not apply StandardScaler or any other scaling transformation.
```

`export_results()` must:

1. Write the eight DataFrames to their specified names with
   `index=False, encoding="utf-8"`.
2. Build the one-row `model_metrics.csv` from config, fitted values, metric
   values, and runtime.
3. Write JSON with `ensure_ascii=False, indent=2`.
4. Write the report with UTF-8.
5. Catch `PermissionError` and `OSError`, name the failed output path, and raise
   `KMeansAnalysisError`.

- [ ] **Step 5: Run export tests and verify GREEN**

Run the Step 2 command again.

Expected: naming, collision refusal, JSON contents, and required UTF-8 exports
all pass.

- [ ] **Step 6: Commit safe exporting**

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: export reproducible k-means experiments"
```

---

### Task 7: Orchestrate progress, reporting, and direct execution

**Files:**
- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Produces: `run_analysis(scenario: str, k: int, input_path: Path, output_root: Path = OUTPUT_ROOT) -> Path`, `main() -> None`
- The module works both through imports and `python scripts/cluster_kmeans/kmeans_analysis.py`.

- [ ] **Step 1: Write the failing end-to-end test**

Add:

```python
class EndToEndTests(unittest.TestCase):
    def test_small_pf_run_creates_complete_final_directory_without_scaling(self) -> None:
        data = pd.concat(
            [
                prepared_pf_frame().fillna(
                    {"N_PFlossR_2kiw": -0.95}
                ),
                prepared_pf_frame().fillna(
                    {"N_PFlossR_2kiw": -0.85}
                ).assign(fid=lambda frame: frame["fid"] + 10),
            ],
            ignore_index=True,
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "pf_for_PCA.csv"
            output_root = root / "output_kmeans"
            data.to_csv(input_path, index=False, encoding="utf-8")
            with patch("builtins.print"):
                final_path = kmeans_analysis.run_analysis(
                    "PF", 2, input_path, output_root
                )

            self.assertEqual(
                final_path.name, "pf_k2_v7_kmeanspp_n50_mi500_rs42"
            )
            self.assertEqual(
                {path.name for path in final_path.iterdir()},
                set(kmeans_analysis.ALL_OUTPUTS),
            )
            clustered = pd.read_csv(final_path / "clustered_data.csv")
            self.assertEqual(len(clustered), len(data))
            self.assertEqual(clustered["kmeans_cluster"].notna().sum(), len(data))
            source_values = data[list(kmeans_analysis.PF_FEATURES)].to_numpy()
            exported_source = clustered[
                list(kmeans_analysis.PF_FEATURES)
            ].to_numpy()
            np.testing.assert_allclose(exported_source, source_values)

    def test_source_contains_no_scaler_import_or_call(self) -> None:
        source = Path(kmeans_analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from sklearn.preprocessing", source)
        self.assertNotIn("StandardScaler(", source)
        self.assertNotIn("MinMaxScaler(", source)
```

- [ ] **Step 2: Run end-to-end tests and verify RED**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.EndToEndTests -v
```

Expected: failure because `run_analysis()` and `main()` orchestration are not
implemented.

- [ ] **Step 3: Implement preparation and the validated execution path**

Implement the preparation helper:

```python
def prepare_data(scenario: str, input_path: Path) -> ValidatedData:
    print("[1/7] Reading input data...")
    data = load_dataset(input_path)
    print("[2/7] Selecting clustering features...")
    selection = select_features(data, scenario)
    print("[3/7] Validating data...")
    return validate_features(data, selection)
```

Implement `_run_validated_analysis()` as the only path that fits, calculates,
plots, and exports:

```python
def _run_validated_analysis(
    scenario: str,
    k: int,
    input_path: Path,
    output_root: Path,
    validated: ValidatedData,
    started_at_dt: datetime,
    started_perf: float,
) -> Path:
    if k < 2 or k >= len(validated.features):
        raise DataValidationError(
            f"K must satisfy 2 <= K < {len(validated.features)} valid samples."
        )
    final_path = build_output_directory(
        output_root, scenario, k, len(validated.feature_names)
    )
    print_preflight_summary(scenario, input_path, k, validated)

    print("[4/7] Running K-means...")
    values = validated.features.to_numpy(dtype=float, copy=True)
    fitted = run_kmeans(values, k)
    ordering = reorder_cluster_labels(
        fitted, values, validated.feature_names
    )

    print("[5/7] Calculating metrics and profiles...")
    metric_results = calculate_metrics(values, ordering.reordered_labels)
    tables = calculate_cluster_profiles(validated, fitted, ordering)
    warnings_list = (
        list(fitted.warnings)
        + list(ordering.warnings)
        + list(metric_results.warnings)
    )

    with staged_output_directory(final_path) as staging:
        print("[6/7] Generating visualizations...")
        generate_pca_plot(
            ordering,
            scenario,
            k,
            validated.feature_names,
            staging / "pca_cluster_scatter.png",
        )
        generate_cluster_size_plot(
            tables.cluster_summary, staging / "cluster_size_bar.png"
        )
        generate_centroid_heatmap(
            tables.centroids,
            validated.feature_names,
            staging / "cluster_centroid_heatmap.png",
        )
        generate_profile_plot(
            tables.feature_profile,
            validated.feature_names,
            staging / "cluster_feature_profiles.png",
        )

        ended_at_dt = datetime.now().astimezone()
        runtime_seconds = time.perf_counter() - started_perf
        config = build_run_config(
            scenario=scenario,
            input_path=input_path.resolve(),
            output_path=final_path.resolve(),
            k=k,
            validated=validated,
            fitted=fitted,
            ordering=ordering,
            warnings_list=warnings_list,
            started_at=started_at_dt.isoformat(timespec="seconds"),
            ended_at=ended_at_dt.isoformat(timespec="seconds"),
        )
        report = build_run_report(
            config,
            tables.cluster_summary,
            metric_results.values,
            runtime_seconds,
        )
        print("[7/7] Exporting results...")
        export_results(
            staging,
            tables,
            metric_results,
            config,
            report,
            runtime_seconds,
        )

    print_completion_summary(
        scenario,
        k,
        validated,
        tables.cluster_summary,
        metric_results.values,
        fitted.inertia,
        runtime_seconds,
        final_path,
    )
    return final_path
```

Add the public non-interactive wrapper:

```python
def run_analysis(
    scenario: str,
    k: int,
    input_path: Path,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    started_at_dt = datetime.now().astimezone()
    started_perf = time.perf_counter()
    validated = prepare_data(scenario, input_path)
    return _run_validated_analysis(
        scenario,
        k,
        input_path,
        output_root,
        validated,
        started_at_dt,
        started_perf,
    )
```

Implement `print_preflight_summary()` so it prints every item required by the
design, including every excluded column/reason and:

```text
No StandardScaler, normalization, or other scaling will be applied.
```

Implement `print_completion_summary()` with the exact leading line:

```text
K-means analysis completed.
```

and scenario, K, features, valid samples, output folder, every cluster
count/percentage, silhouette or `not available`, inertia, and runtime.

- [ ] **Step 4: Implement interactive `main()` with one read and K retry**

Use the same preparation and validated execution helpers so the interactive
path reads the CSV once and avoids raw tracebacks:

```python
def main() -> None:
    try:
        scenario = prompt_scenario()
        k = prompt_k()
        input_path = prompt_csv_path(scenario)
        started_at_dt = datetime.now().astimezone()
        started_perf = time.perf_counter()
        validated = prepare_data(scenario, input_path)
        if k >= len(validated.features):
            print(
                f"K must be smaller than the valid sample count "
                f"({len(validated.features)})."
            )
            k = prompt_k(n_samples=len(validated.features))

        _run_validated_analysis(
            scenario,
            k,
            input_path,
            OUTPUT_ROOT,
            validated,
            started_at_dt,
            started_perf,
        )
    except (
        KMeansAnalysisError,
        PermissionError,
        OSError,
    ) as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run end-to-end and full new-module tests and verify GREEN**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis -v
```

Expected: all new tests pass and no scaler import/call appears in the new
script.

- [ ] **Step 6: Verify the entry-point guard and interactive orchestration**

Add a test that patches `prompt_scenario()`, `prompt_k()`,
`prompt_csv_path()`, `prepare_data()`, and `_run_validated_analysis()`, then
calls `main()`. Assert that:

- prompts are called in scenario, K, path order;
- `prepare_data()` is called once;
- a K that is too large triggers `prompt_k(n_samples=...)`;
- `_run_validated_analysis()` receives the already validated object.

Also assert that the final source lines contain:

```python
if __name__ == "__main__":
    main()
```

Run:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis.EndToEndTests -v
```

Expected: the end-to-end and entry-point orchestration tests pass without
writing outside temporary directories.

- [ ] **Step 7: Commit orchestration**

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: add interactive k-means workflow"
```

---

### Task 8: Run regression, syntax, artifact, and requirement verification

**Files:**
- Verify: `scripts/utils/scenario_features.py`
- Verify: `scripts/cluster_preparation/pca_analysis.py`
- Verify: `scripts/cluster_preparation/choose_k.py`
- Verify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Verify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**
- Consumes: all completed tasks
- Produces: fresh evidence that the approved design is satisfied without regressions

- [ ] **Step 1: Compile affected Python files**

```powershell
python -m py_compile scripts/utils/scenario_features.py scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/choose_k.py scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Run focused new tests**

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis -v
```

Expected: all new tests pass.

- [ ] **Step 3: Run affected existing regression tests**

```powershell
python -m unittest scripts.cluster_preparation.test_pca_analysis scripts.cluster_preparation.test_choose_k -v
```

Expected: all existing PCA and choose-K tests pass.

- [ ] **Step 4: Run repository unittest discovery**

```powershell
python -m unittest discover -v
```

Expected: exit code 0 with zero failures and zero errors.

- [ ] **Step 5: Check prohibited transformations and required outputs**

```powershell
rg -n "StandardScaler|MinMaxScaler|normalize|fit_transform" scripts/cluster_kmeans/kmeans_analysis.py
```

Expected: only explanatory report strings may mention `StandardScaler`; no
preprocessing import and no scaling call are present.

Run:

```powershell
rg -n "clustered_data.csv|cluster_summary.csv|cluster_centroids_standardized.csv|cluster_feature_profile.csv|sample_distances.csv|model_metrics.csv|run_config.json|run_report.txt|cluster_label_mapping.csv|pca_cluster_scatter.png|cluster_size_bar.png|cluster_centroid_heatmap.png|cluster_feature_profiles.png" scripts/cluster_kmeans/kmeans_analysis.py
```

Expected: every required filename is present.

- [ ] **Step 6: Inspect the final diff and working tree**

```powershell
git diff --check
git status --short
git diff HEAD~4 -- scripts/utils/scenario_features.py scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/choose_k.py scripts/cluster_kmeans
```

Expected: no whitespace errors, only in-scope files, and no generated
`output_kmeans/` data staged for commit.

- [ ] **Step 7: Commit any verification-only correction**

If verification exposed a defect, first add a focused failing regression test,
confirm RED, apply the smallest correction, rerun Steps 1-6, then commit only
that correction:

```powershell
git add scripts/cluster_kmeans scripts/utils/scenario_features.py scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/choose_k.py
git commit -m "fix: complete k-means workflow verification"
```

If no correction was required, do not create an empty commit.

## Execution Notes

- Activate the conda environment represented by `2026master.yml` before
  running tests; the system Python is not an adequate substitute when it lacks
  scikit-learn.
- Do not write real experiment outputs during automated tests. Every
  end-to-end test uses `TemporaryDirectory`.
- Review every task against the approved design before its commit.
- After the final verification, use `superpowers:requesting-code-review` and
  `superpowers:verification-before-completion` before claiming completion.
