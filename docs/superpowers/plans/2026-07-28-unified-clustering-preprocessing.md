# Unified Clustering Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PCA, K selection, K-means, and primary HDBSCAN consume one immutable PF/TF prepared artifact in which structural nulls are represented as semantic zero and one scenario-specific `StandardScaler` is fitted after high-risk row filtering and final feature selection.

**Architecture:** Keep semantic cleaning and high-risk filtering as upstream stages. Add one canonical preprocessing module that dynamically classifies columns, validates semantic values, applies explicit feature exclusions, fills only actual nulls with `0.0`, removes post-fill constants, standardizes once, and serializes a hash-verified artifact. Refactor every downstream algorithm into a read-only consumer of that artifact; keep original and filled semantic values beside the standardized matrix for planning interpretation and centroid/profile reporting.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn (`StandardScaler`, `PCA`, `KMeans`, clustering metrics), HDBSCAN through the project’s existing environment, matplotlib/seaborn, `unittest`, `unittest.mock`, JSON, CSV, SHA-256.

## Global Constraints

- Treat null in an eligible PF/TF indicator as structural “no measurable change” and replace it with `0.0` only in the modelling-preparation layer.
- Never overwrite nulls in the semantic cleaned CSV or high-risk source CSV.
- Preserve real `-1.0`, `0.0`, and `1.0` values unchanged before standardization.
- Reject malformed strings, infinity, and finite values outside `[-1, 1]`; do not coerce them to zero.
- Discover PF features with anchored, case-insensitive `^PF`, `^N_PF`, and `^Per_` markers.
- Discover TF features with anchored, case-insensitive `^TF`, `^N_TF`, and `^Tem_` markers.
- Exclude identifiers, composite `Index`/`Risk` fields, results/labels, shared fields, explicit research exclusions, and post-fill constants from the model matrix.
- Reject an active-scenario PF/TF-marked field when it is non-numeric; do not silently exclude or coerce it.
- Preserve source-column order as feature order and source-row order as sample order.
- Fit one default `StandardScaler` separately for each final PF/TF high-risk dataset.
- PCA, K selection, K-means, and primary HDBSCAN must not rediscover features, fill nulls, drop null-bearing rows, or fit another scaler.
- Correlation analysis uses the shared discovery and semantic-zero rules but does not standardize.
- Prepared scenario directories and downstream result directories remain non-overwriting.
- Do not modify `scripts/clean_mt_update_ms_hev.py` or the mathematical definitions of existing indicators.
- Do not stage or alter generated outputs or unrelated user data already present in the worktree.
- Use the confirmed design as the source of truth: `docs/superpowers/specs/2026-07-28-unified-clustering-preprocessing-design.md`.

## File Map

### Add

- `scripts/utils/clustering_preprocessing.py`
- `scripts/utils/test_clustering_preprocessing.py`
- `scripts/cluster_preparation/prepare_clustering_inputs.py`
- `scripts/cluster_preparation/test_prepare_clustering_inputs.py`
- `scripts/clustering/test_hdbscan_runner.py`
- `tests/test_unified_clustering_workflow.py`

### Modify

- `scripts/utils/scenario_features.py`
- `scripts/utils/feature_utils.py`
- `scripts/pearson_correlation/correlation_analysis.py`
- `scripts/pearson_correlation/test_correlation_analysis.py`
- `scripts/pearson_correlation/extract_high_columns.py`
- `scripts/pearson_correlation/test_extract_high_columns.py`
- `scripts/cluster_preparation/pca_analysis.py`
- `scripts/cluster_preparation/test_pca_analysis.py`
- `scripts/cluster_preparation/choose_k.py`
- `scripts/cluster_preparation/test_choose_k.py`
- `scripts/cluster_kmeans/kmeans_analysis.py`
- `scripts/cluster_kmeans/test_kmeans_analysis.py`
- `scripts/clustering/hdbscan_runner.py`
- `scripts/run_hdbscan_street_typology.py`
- `scripts/utils/config.py`
- `README.md`

### Verify without modifying

- `scripts/clean_mt_update_ms_hev.py`
- `scripts/filter_high_risk_dataset.py`
- `data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv`
- `data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv`

---

### Task 1: Implement canonical column classification and dynamic feature discovery

**Files:**

- Create: `scripts/utils/clustering_preprocessing.py`
- Create: `scripts/utils/test_clustering_preprocessing.py`

**Interfaces:**

- Consumes: a pandas `DataFrame`, scenario text, and optional explicit exclusions.
- Produces: `normalize_scenario(value) -> str`, `classify_column(name, series) -> FieldRole`, and `discover_scenario_features(data, scenario, exclusions=()) -> FeatureDiscovery`.
- Guarantees: anchored classification, source-order features, one recorded role for every source column, and strict validation of explicit exclusions.

- [ ] **Step 1: Write failing classification tests**

Create `scripts/utils/test_clustering_preprocessing.py` with a reusable fixture and these assertions:

```python
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.utils import clustering_preprocessing as preprocessing


class FeatureDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "fid": [1, 2, 3],
                "MS_ID": ["a", "b", "c"],
                "PF_Index_Risk_equal": [0.4, 0.5, 0.6],
                "Per_extent": [0.0, 0.2, 0.4],
                "N_PFlossR_2kiw": [-1.0, np.nan, 1.0],
                "PFResident_lossR": [0.0, -0.2, -0.4],
                "PFABC_NOR": [0.1, 0.2, 0.3],
                "Tem_extent": [0.3, 0.2, 0.1],
                "TFABC_NOR": [0.4, 0.5, 0.6],
                "Street_type_NOR": [0.2, 0.3, 0.4],
                "PF_result_label": ["x", "y", "z"],
                "unrelated_pf_text": [1.0, 2.0, 3.0],
            }
        )

    def test_classification_is_anchored_case_insensitive_and_role_complete(self):
        discovery = preprocessing.discover_scenario_features(self.data, "pf")

        self.assertEqual(discovery.scenario, "PF")
        self.assertEqual(
            discovery.feature_names,
            ("Per_extent", "N_PFlossR_2kiw", "PFResident_lossR", "PFABC_NOR"),
        )
        roles = discovery.column_roles.set_index("column")["role"].to_dict()
        self.assertEqual(roles["fid"], "IDENTIFIER")
        self.assertEqual(roles["PF_Index_Risk_equal"], "COMPOSITE_INDEX")
        self.assertEqual(roles["Tem_extent"], "TF_FEATURE")
        self.assertEqual(roles["Street_type_NOR"], "SHARED_FEATURE")
        self.assertEqual(roles["PF_result_label"], "RESULT_OR_LABEL")
        self.assertEqual(roles["unrelated_pf_text"], "EXCLUDED")

    def test_explicit_exclusions_must_exist_and_belong_to_active_scenario(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            preprocessing.discover_scenario_features(
                self.data, "PF", exclusions=("PF_missing",)
            )
        with self.assertRaisesRegex(ValueError, "TFABC_NOR"):
            preprocessing.discover_scenario_features(
                self.data, "PF", exclusions=("TFABC_NOR",)
            )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing.FeatureDiscoveryTests -v
```

Expected: import error because `clustering_preprocessing.py` does not exist.

- [ ] **Step 3: Add the public role and discovery structures**

Implement these public definitions:

```python
class FieldRole(str, Enum):
    PF_FEATURE = "PF_FEATURE"
    TF_FEATURE = "TF_FEATURE"
    SHARED_FEATURE = "SHARED_FEATURE"
    IDENTIFIER = "IDENTIFIER"
    COMPOSITE_INDEX = "COMPOSITE_INDEX"
    RESULT_OR_LABEL = "RESULT_OR_LABEL"
    NON_NUMERIC = "NON_NUMERIC"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True)
class FeatureDiscovery:
    scenario: str
    feature_names: tuple[str, ...]
    column_roles: pd.DataFrame


PF_PATTERN = re.compile(r"^(?:PF|N_PF|Per_)", re.IGNORECASE)
TF_PATTERN = re.compile(r"^(?:TF|N_TF|Tem_)", re.IGNORECASE)
IDENTIFIER_NAMES = frozenset({"fid", "ms_id"})
SHARED_FEATURE_NAMES = frozenset(
    {"street_type_nor", "resident_nor", "daynight_nor", "prm_nor"}
)
RESULT_OR_LABEL_PATTERN = re.compile(
    r"(?:cluster|label|rule|typology|intervention|result)", re.IGNORECASE
)
```

Apply role precedence in this exact order: identifier; `Index`/`Risk`; result/label; known shared feature; PF marker; TF marker; excluded. When an active-scenario PF/TF-marked series is not numeric, record `NON_NUMERIC` and raise an error naming the column; never silently omit it from the candidate set.

- [ ] **Step 4: Implement strict exclusion validation and stable selection**

`discover_scenario_features()` must:

1. normalize only `PF`, `TF`, `1`, or `2`;
2. classify all columns in source order;
3. reject exclusions absent from the table;
4. reject exclusions whose base role is not the active scenario feature role;
5. relabel valid exclusions as `EXCLUDED` with reason `explicit_research_exclusion`;
6. fail with `non-numeric` and the column names if any remaining active candidate is not numeric;
7. return remaining active numeric features in source order;
8. fail with `No usable PF features discovered` or `No usable TF features discovered` when empty.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing.FeatureDiscoveryTests -v
python -m py_compile scripts/utils/clustering_preprocessing.py scripts/utils/test_clustering_preprocessing.py
git diff --check
```

Expected: all focused tests pass and compilation exits 0.

Commit:

```powershell
git add scripts/utils/clustering_preprocessing.py scripts/utils/test_clustering_preprocessing.py
git commit -m "feat: add dynamic clustering feature discovery"
```

---

### Task 2: Validate semantic values, fill structural nulls, and standardize once

**Files:**

- Modify: `scripts/utils/clustering_preprocessing.py`
- Modify: `scripts/utils/test_clustering_preprocessing.py`

**Interfaces:**

- Consumes: a non-empty high-risk table, scenario, explicit exclusions, semantic bounds.
- Produces: `prepare_clustering_data(...) -> PreparedClusteringData` and `inverse_transform_frame(...) -> pd.DataFrame`.
- Guarantees: no source mutation, no row loss, auditable null replacement, post-fill constant removal, finite standardized values, and reversible scaling.

- [ ] **Step 1: Write failing preparation and validation tests**

Add tests built from a four-row PF fixture containing real `-1`, `0`, `1`, one null, and one post-fill constant:

```python
class PrepareClusteringDataTests(unittest.TestCase):
    def test_structural_zero_preserves_rows_extremes_and_source(self):
        source = pd.DataFrame(
            {
                "fid": [10, 11, 12, 13],
                "MS_ID": ["s0", "s1", "s2", "s3"],
                "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6],
                "Per_extent": [-1.0, 0.0, 1.0, np.nan],
                "PFABC_NOR": [-0.8, -0.2, 0.2, 0.8],
                "PF_constant": [np.nan, 0.0, np.nan, 0.0],
            }
        )
        original = source.copy(deep=True)

        prepared = preprocessing.prepare_clustering_data(source, "PF")

        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(prepared.feature_names, ("Per_extent", "PFABC_NOR"))
        self.assertEqual(prepared.metadata["source_row_position"].tolist(), [0, 1, 2, 3])
        self.assertEqual(prepared.filled_features["Per_extent"].tolist(), [-1.0, 0.0, 1.0, 0.0])
        self.assertEqual(prepared.missing_counts["Per_extent"], 1)
        self.assertEqual(prepared.rows_with_structural_null.tolist(), [False, False, False, True])
        np.testing.assert_allclose(
            prepared.standardized_features.mean(axis=0).to_numpy(),
            np.zeros(2),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            prepared.standardized_features.std(axis=0, ddof=0).to_numpy(),
            np.ones(2),
            atol=1e-10,
            rtol=1e-10,
        )
        recovered = preprocessing.inverse_transform_frame(
            prepared.standardized_features,
            prepared.scaler_parameters,
        )
        pd.testing.assert_frame_equal(
            recovered,
            prepared.filled_features,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_invalid_semantic_values_fail_instead_of_becoming_zero(self):
        base = pd.DataFrame(
            {
                "fid": [1, 2],
                "MS_ID": ["a", "b"],
                "PF_Index_Risk_equal": [0.4, 0.5],
                "PFABC_NOR": [0.0, 0.5],
            }
        )
        cases = {
            "non-numeric": ["bad", "0.5"],
            "infinity": [0.0, np.inf],
            "outside": [0.0, 1.01],
        }
        for message, values in cases.items():
            with self.subTest(message=message):
                candidate = base.copy()
                candidate["PFABC_NOR"] = values
                with self.assertRaisesRegex(ValueError, message):
                    preprocessing.prepare_clustering_data(candidate, "PF")
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing.PrepareClusteringDataTests -v
```

Expected: `AttributeError` for missing `prepare_clustering_data`.

- [ ] **Step 3: Add the preparation boundary**

Add:

```python
@dataclass(frozen=True)
class PreparedClusteringData:
    scenario: str
    source_data: pd.DataFrame
    metadata: pd.DataFrame
    original_features: pd.DataFrame
    filled_features: pd.DataFrame
    standardized_features: pd.DataFrame
    feature_names: tuple[str, ...]
    missing_counts: pd.Series
    rows_with_structural_null: pd.Series
    excluded_columns: pd.DataFrame
    audit: pd.DataFrame
    scaler: StandardScaler
    scaler_parameters: pd.DataFrame
```

Require `fid`, `MS_ID`, and `{scenario}_Index_Risk_equal`. Add `source_row_position = np.arange(len(data), dtype=np.int64)` to metadata. Duplicate IDs produce an audit warning but do not change row alignment.

- [ ] **Step 4: Implement validation and transformation in the specified order**

Use this sequence inside `prepare_clustering_data()`:

```python
original = data.loc[:, discovery.feature_names].copy()
missing_counts = original.isna().sum().astype("int64")
row_null_counts = original.isna().sum(axis=1).astype("int64")
filled = original.fillna(0.0)
constant_names = tuple(
    name for name in filled.columns if filled[name].nunique(dropna=False) <= 1
)
filled = filled.drop(columns=list(constant_names))
original = original.drop(columns=list(constant_names))
missing_counts = missing_counts.drop(index=list(constant_names))
scaler = StandardScaler()
standardized = pd.DataFrame(
    scaler.fit_transform(filled),
    columns=filled.columns,
    index=filled.index,
)
```

Before `fillna`, reject non-numeric active columns, infinity, and finite values outside the inclusive configured bounds. After constant removal, fail if no feature remains. After scaling, use `np.isfinite` and `np.allclose(..., atol=1e-10, rtol=1e-10)` to enforce finite values, zero means, and population standard deviations of one.

Build `scaler_parameters` with columns:

```text
feature, mean, variance, scale, original_null_count,
original_min, original_max, filled_min, filled_max
```

`inverse_transform_frame()` must validate exact feature order and calculate `z * scale + mean` using the persisted parameter table.

- [ ] **Step 5: Add edge-case tests**

Cover:

- empty input;
- missing `fid`, `MS_ID`, or active risk column;
- all features excluded or constant;
- non-default index values with preserved row order;
- deterministic repeated preparation;
- opposite-scenario columns never entering PF;
- correct per-row `structural_null_count` and `had_structural_null`.

- [ ] **Step 6: Run the shared suite and commit**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing -v
python -m py_compile scripts/utils/clustering_preprocessing.py
git diff --check
```

Commit:

```powershell
git add scripts/utils/clustering_preprocessing.py scripts/utils/test_clustering_preprocessing.py
git commit -m "feat: prepare structural-zero standardized matrices"
```

---

### Task 3: Serialize and verify immutable prepared artifacts

**Files:**

- Modify: `scripts/utils/clustering_preprocessing.py`
- Modify: `scripts/utils/test_clustering_preprocessing.py`

**Interfaces:**

- Consumes: `PreparedClusteringData`, source path, output root, risk-filter definition, optional exclusion-artifact path.
- Produces: `save_prepared_artifact(...) -> Path` and `load_prepared_artifact(path) -> PreparedArtifact`.
- Guarantees: complete artifact layout, SHA-256 provenance, exact feature order, strict load-time integrity validation, and no overwrite.

- [ ] **Step 1: Write failing round-trip and tamper tests**

Use `TemporaryDirectory`, write the source fixture to CSV, prepare it, save it, then assert:

```python
artifact_dir = preprocessing.save_prepared_artifact(
    prepared,
    source_path=source_path,
    output_root=output_root,
    risk_filter={
        "column": "PF_Index_Risk_equal",
        "operator": ">",
        "threshold": 0.25,
    },
    exclusion_path=None,
)
loaded = preprocessing.load_prepared_artifact(artifact_dir)

self.assertEqual(loaded.feature_names, prepared.feature_names)
pd.testing.assert_frame_equal(
    loaded.standardized_features,
    prepared.standardized_features,
    check_exact=False,
    atol=1e-14,
    rtol=1e-14,
)
with self.assertRaisesRegex(FileExistsError, "immutable"):
    preprocessing.save_prepared_artifact(
        prepared,
        source_path,
        output_root,
        {"column": "PF_Index_Risk_equal", "operator": ">", "threshold": 0.25},
    )
```

After the successful round trip, change one value in `features_standardized.csv` and assert `load_prepared_artifact()` raises an error containing `matrix SHA-256`.

- [ ] **Step 2: Run the artifact tests and confirm RED**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing.PreparedArtifactTests -v
```

Expected: missing save/load APIs.

- [ ] **Step 3: Add the loaded-artifact structure and hashing helpers**

Add:

```python
@dataclass(frozen=True)
class PreparedArtifact:
    directory: Path
    config: dict[str, object]
    metadata: pd.DataFrame
    original_features: pd.DataFrame
    filled_features: pd.DataFrame
    standardized_features: pd.DataFrame
    feature_names: tuple[str, ...]
    excluded_features: pd.DataFrame
    scaler_parameters: pd.DataFrame
    audit: pd.DataFrame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

Hash feature order as UTF-8 text joined by `"\n"` and terminated by one newline. Save JSON with `sort_keys=True`, `indent=2`, and UTF-8 encoding.

- [ ] **Step 4: Materialize the exact artifact layout**

Create `{output_root}/{source_path.stem}/{scenario}/` containing only:

```text
metadata.csv
features_original.csv
features_filled.csv
features_standardized.csv
selected_features.csv
excluded_features.csv
scaler_parameters.csv
preprocessing_audit.csv
preprocessing_config.json
```

Check the scenario directory does not exist before creating any child. Write to a temporary sibling directory and rename it to the final directory only after every file succeeds. On failure, remove only that newly created temporary sibling.

The config must include all provenance required by the design: resolved source path and hash, source row count, scenario, risk filter, marker rules, exclusion path/hash or `null`, feature order/hash, structural-null policy, scaler configuration/statistics, standardized-matrix hash, Python/pandas/NumPy/scikit-learn versions, and UTC creation timestamp.

- [ ] **Step 5: Implement strict loader validation**

`load_prepared_artifact()` must fail before returning when:

- one of the nine files is absent;
- config scenario is not `PF` or `TF`;
- selected-feature order differs from config;
- any feature CSV column order differs;
- row counts differ between metadata and feature matrices;
- source-row positions are not `0..n-1`;
- matrix hash differs;
- feature-order hash differs;
- standardized values are non-finite.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing -v
python -m py_compile scripts/utils/clustering_preprocessing.py
git diff --check
```

Commit:

```powershell
git add scripts/utils/clustering_preprocessing.py scripts/utils/test_clustering_preprocessing.py
git commit -m "feat: persist verified preprocessing artifacts"
```

---

### Task 4: Add the modelling-preparation command-line entry point

**Files:**

- Create: `scripts/cluster_preparation/prepare_clustering_inputs.py`
- Create: `scripts/cluster_preparation/test_prepare_clustering_inputs.py`

**Interfaces:**

- Consumes: an already high-risk-filtered CSV, `PF`/`TF`, output root, optional exclusion JSON, and recorded risk comparison.
- Produces: one immutable scenario artifact and a concise terminal audit.
- Does not: filter full data implicitly or default to correlation-extracted `pf_for_PCA.csv`/`tf_for_PCA.csv`.

- [ ] **Step 1: Write failing CLI-boundary tests**

Define and test:

```python
artifact_dir = prepare_inputs.run_preparation(
    input_path=input_path,
    scenario="PF",
    output_root=output_root,
    exclusion_path=exclusion_path,
    risk_column="PF_Index_Risk_equal",
    risk_operator=">",
    risk_threshold=0.25,
)
loaded = preprocessing.load_prepared_artifact(artifact_dir)
self.assertEqual(len(loaded.metadata), len(source))
self.assertNotIn("PFABC_NOR", loaded.feature_names)
self.assertEqual(loaded.config["structural_null"]["fill_value"], 0.0)
```

Use this exclusion schema:

```json
{
  "scenario": "PF",
  "excluded_features": ["PFABC_NOR"],
  "reason": "documented correlation review decision"
}
```

Also test scenario mismatch, malformed JSON, duplicate exclusion names, absent exclusions, and any source row that fails the recorded risk comparison.

- [ ] **Step 2: Run the CLI tests and confirm RED**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_prepare_clustering_inputs -v
```

Expected: import error for the new entry point.

- [ ] **Step 3: Implement exclusion loading and risk-filter verification**

Expose:

```python
def load_exclusions(path: Path | None, scenario: str) -> tuple[str, ...]:
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scenario", "").upper() != scenario:
        raise ValueError("Exclusion artifact scenario does not match active scenario.")
    names = payload.get("excluded_features")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("excluded_features must be a JSON list of strings.")
    if len(names) != len(set(names)):
        raise ValueError("excluded_features contains duplicate names.")
    return tuple(names)
```

Support only `>` and `>=` risk operators. Verify every row satisfies the recorded predicate; never silently remove a row at this stage.

- [ ] **Step 4: Implement `run_preparation()` and `main()`**

Arguments:

```text
--input PATH                 required
--scenario {PF,TF}           required
--output-root PATH           default outputs_preprocessing
--exclusions PATH            optional
--risk-column NAME           defaults to {scenario}_Index_Risk_equal
--risk-operator {>,>=}       default >
--risk-threshold FLOAT       default 0.25
```

Read with pandas, verify risk status, call `prepare_clustering_data()`, save the artifact, then print input rows, retained rows, feature count, rows containing structural nulls, per-feature null counts, removed constants, and final artifact path.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_prepare_clustering_inputs -v
python -m py_compile scripts/cluster_preparation/prepare_clustering_inputs.py
git diff --check
```

Commit:

```powershell
git add scripts/cluster_preparation/prepare_clustering_inputs.py scripts/cluster_preparation/test_prepare_clustering_inputs.py
git commit -m "feat: add clustering preparation command"
```

---

### Task 5: Refactor correlation review to shared dynamic semantics

**Files:**

- Modify: `scripts/pearson_correlation/correlation_analysis.py`
- Modify: `scripts/pearson_correlation/test_correlation_analysis.py`

**Interfaces:**

- Consumes: semantic/high-risk CSV, one selected scenario, shared discovery rules.
- Produces: the existing correlation tables and plots using dynamically discovered, structural-zero-filled features.
- Guarantees: no fixed complete PF/TF feature list and no `StandardScaler`.

- [ ] **Step 1: Replace fixed-list tests with dynamic-discovery tests**

Create a fixture containing PF, TF, shared, index/risk, unrelated numeric, and null-bearing PF fields. Assert:

```python
feature_groups, original, filled = correlation.prepare_correlation_features(
    data, "PF"
)
self.assertEqual(
    tuple(filled.columns),
    ("Per_extent", "N_PFlossR_2kiw", "PFResident_lossR", "PFABC_NOR"),
)
self.assertTrue(np.isnan(original.loc[1, "N_PFlossR_2kiw"]))
self.assertEqual(filled.loc[1, "N_PFlossR_2kiw"], 0.0)
self.assertNotIn("TFABC_NOR", filled.columns)
self.assertNotIn("PF_Index_Risk_equal", filled.columns)
```

Patch `scripts.pearson_correlation.correlation_analysis.StandardScaler` with `create=True` and a side effect that fails if instantiated; run `analyze_scenario()` and confirm the patch is not called.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
python -m unittest scripts.pearson_correlation.test_correlation_analysis -v
```

Expected: failures because fixed feature groups remain active.

- [ ] **Step 3: Add dynamic correlation preparation**

Import shared discovery and semantic validation/fill helpers. Remove `PF_FEATURE_GROUPS`, `TF_FEATURE_GROUPS`, and `SCENARIO_FEATURE_GROUPS` as complete source-of-truth lists.

Add deterministic grouping:

```python
def correlation_group(column: str) -> str:
    lowered = column.lower()
    if lowered.startswith(("per_", "tem_")):
        return "extent"
    if lowered.startswith(("n_pf", "n_tf")):
        return "network_loss"
    if "resident" in lowered or "daynight" in lowered:
        return "population_exposure"
    if "ab5k" in lowered or "abc" in lowered:
        return "accessibility"
    return "other_scenario_indicator"
```

`prepare_correlation_features()` returns the group mapping, original semantic features, and zero-filled features. Correlation coefficients and pair tables use `filled`; summaries retain original null counts and filled-zero counts.

Update argument parsing to accept `--scenario {PF,TF}` while retaining the existing interactive scenario prompt when it is omitted. `--input` continues to accept the semantic or high-risk CSV selected by the researcher.

- [ ] **Step 4: Preserve output contracts and record feature roles**

Keep these outputs:

```text
feature_summary.csv
data_quality_warnings.csv
pearson_correlation_matrix.csv
all_correlation_pairs.csv
high_correlation_pairs.csv
high_correlation_pairs_within_group.csv
high_correlation_pairs_cross_group.csv
```

Add `feature_roles.csv` from shared discovery. Update the text report to state `structural null -> 0.0` and `StandardScaler not applied because Pearson correlation is affine-invariant`.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
python -m unittest scripts.pearson_correlation.test_correlation_analysis -v
python -m py_compile scripts/pearson_correlation/correlation_analysis.py
git diff --check
```

Commit:

```powershell
git add scripts/pearson_correlation/correlation_analysis.py scripts/pearson_correlation/test_correlation_analysis.py
git commit -m "refactor: use shared dynamic correlation features"
```

---

### Task 6: Remove fixed feature lists and isolate the legacy extraction path

**Files:**

- Modify: `scripts/utils/scenario_features.py`
- Modify: `scripts/utils/feature_utils.py`
- Modify: `scripts/pearson_correlation/extract_high_columns.py`
- Modify: `scripts/pearson_correlation/test_extract_high_columns.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**

- Consumes: canonical preprocessing APIs.
- Produces: metadata/default-path compatibility constants and a temporary `feature_utils` adapter.
- Guarantees: no active default points to `outputs_correlation/*/*_for_PCA.csv`; legacy extraction requires an explicit input and warns.

- [ ] **Step 1: Write failing source-of-truth and deprecation tests**

Assert:

```python
self.assertFalse(hasattr(scenario_features, "PF_FEATURES"))
self.assertFalse(hasattr(scenario_features, "TF_FEATURES"))
self.assertNotIn("outputs_correlation", str(scenario_features.DEFAULT_INPUT_PATHS["PF"]))
```

For `extract_high_columns.main()`, patch arguments without `--input` and assert parser termination. With an explicit input, assert one `DeprecationWarning` whose message directs users to `prepare_clustering_inputs.py`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
python -m unittest scripts.pearson_correlation.test_extract_high_columns scripts.cluster_kmeans.test_kmeans_analysis.SharedScenarioConfigurationTests -v
```

Expected: current fixed-list assertions/defaults fail the new contract.

- [ ] **Step 3: Reduce `scenario_features.py` to metadata/default path compatibility**

Keep:

```python
PF_METADATA_COLUMNS = ("source_row_position", "fid", "MS_ID", "PF_Index_Risk_equal")
TF_METADATA_COLUMNS = ("source_row_position", "fid", "MS_ID", "TF_Index_Risk_equal")

DEFAULT_INPUT_PATHS = {
    "PF": PROJECT_ROOT / "data" / "MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv",
    "TF": PROJECT_ROOT / "data" / "MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv",
}
```

Remove `PF_FEATURES`, `TF_FEATURES`, and `SCENARIO_CONFIG["features"]`.

- [ ] **Step 4: Turn `feature_utils.py` into a compatibility adapter**

Re-export `FieldRole`, `FeatureDiscovery`, `PreparedClusteringData`, `discover_scenario_features`, `prepare_clustering_data`, and artifact loader APIs from `clustering_preprocessing`. Keep only adapters needed until HDBSCAN is migrated; mark them deprecated in docstrings and do not maintain a second feature-selection implementation.

- [ ] **Step 5: Make extraction explicitly legacy**

Remove the default input value. Make `--input` required in non-interactive execution. Print:

```text
DEPRECATED: extract_high_columns.py is a legacy export tool. Active PCA,
K-selection, K-means, and HDBSCAN workflows must use
prepare_clustering_inputs.py and an explicit exclusion JSON.
```

Do not change its historical export calculation beyond removing active defaults.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
python -m unittest scripts.pearson_correlation.test_extract_high_columns scripts.cluster_kmeans.test_kmeans_analysis.SharedScenarioConfigurationTests -v
python -m py_compile scripts/utils/scenario_features.py scripts/utils/feature_utils.py scripts/pearson_correlation/extract_high_columns.py
git diff --check
```

Commit:

```powershell
git add scripts/utils/scenario_features.py scripts/utils/feature_utils.py scripts/pearson_correlation/extract_high_columns.py scripts/pearson_correlation/test_extract_high_columns.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "refactor: retire fixed clustering feature lists"
```

---

### Task 7: Make PCA a read-only prepared-artifact consumer

**Files:**

- Modify: `scripts/cluster_preparation/pca_analysis.py`
- Modify: `scripts/cluster_preparation/test_pca_analysis.py`

**Interfaces:**

- Consumes: prepared scenario directory or `preprocessing_config.json`.
- Produces: current PCA tables/plots plus copied preprocessing identity.
- Guarantees: the exact persisted standardized matrix enters `PCA.fit_transform`.

- [ ] **Step 1: Rewrite PCA input tests around one saved fixture artifact**

Assert:

```python
artifact = preprocessing.load_prepared_artifact(artifact_dir)
with patch.object(pca, "run_pca", wraps=pca.run_pca) as run_pca:
    output_dir = pca.run_analysis(artifact_dir, output_root)

actual = run_pca.call_args.args[0]
np.testing.assert_allclose(
    actual,
    artifact.standardized_features.to_numpy(dtype=float),
)
```

Patch `sklearn.preprocessing.StandardScaler.fit` to raise `AssertionError`; PCA analysis must still complete. Remove tests for PCA-local fixed whitelists, `fillna`, and `standardize()`.

- [ ] **Step 2: Run PCA tests and confirm RED**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_pca_analysis -v
```

Expected: current `run_analysis(input_path, scenario, output_root)` cannot consume an artifact.

- [ ] **Step 3: Replace PCA-local preparation**

Remove PCA’s `PreparedDataset`, `_scenario_features`, `_scenario_metadata`, `load_data`, `standardize`, and fixed-feature imports. Add:

```python
def load_pca_input(path: Path) -> PreparedArtifact:
    return load_prepared_artifact(path)


def run_analysis(prepared_path: Path, output_root: Path) -> Path:
    artifact = load_pca_input(prepared_path)
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    result = run_pca(values, artifact.feature_names)
```

Use artifact metadata in `pca_scores.csv`. Write `preprocessing_reference.json` containing resolved manifest path, manifest SHA-256, matrix SHA-256, scenario, row count, and ordered feature names.

- [ ] **Step 4: Update CLI and reporting**

Arguments:

```text
--prepared PATH     required unless prompted
--output-root PATH  default output_pca
```

Remove raw CSV and scenario selection as modelling inputs. Scenario comes from the verified artifact. Report explicitly that PCA is diagnostic and K-means does not consume PCA scores.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_pca_analysis -v
python -m py_compile scripts/cluster_preparation/pca_analysis.py
git diff --check
```

Commit:

```powershell
git add scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/test_pca_analysis.py
git commit -m "refactor: consume prepared matrix in PCA"
```

---

### Task 8: Make K selection use the identical prepared geometry

**Files:**

- Modify: `scripts/cluster_preparation/choose_k.py`
- Modify: `scripts/cluster_preparation/test_choose_k.py`

**Interfaces:**

- Consumes: the same prepared scenario artifact used by PCA and final K-means.
- Produces: existing K metrics/plots, recommendation text, and preprocessing identity.
- Guarantees: no local feature selection, null fill, scaler fitting, or row deletion.

- [ ] **Step 1: Replace local-preparation tests with matrix identity tests**

Load a saved artifact and patch `evaluate_k()`:

```python
with patch.object(choose_k, "evaluate_k", return_value=expected) as evaluate:
    output_dir = choose_k.run_analysis(
        prepared_path=artifact_dir,
        output_root=output_root,
        k_values=range(2, 5),
    )
np.testing.assert_allclose(
    evaluate.call_args.args[0],
    artifact.standardized_features.to_numpy(dtype=float),
)
self.assertEqual(evaluate.call_args.args[1], [2, 3, 4])
```

Patch `StandardScaler.fit` to fail if called. Assert output config matrix hash equals the artifact hash.

- [ ] **Step 2: Run choose-K tests and confirm RED**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_choose_k -v
```

Expected: failures from current raw-CSV loading and local scaling.

- [ ] **Step 3: Remove local preprocessing and update the public boundary**

Delete choose-K’s `PreparedDataset`, fixed feature imports, `load_data`, and `standardize_features`. Implement:

```python
def run_analysis(
    prepared_path: Path,
    output_root: Path,
    k_values: Iterable[int],
) -> Path:
    artifact = load_prepared_artifact(prepared_path)
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    valid_k = _validate_k_range(values, k_values)
    results = evaluate_k(values, valid_k)
```

Persist `preprocessing_reference.json` with the same fields as PCA.

- [ ] **Step 4: Preserve and strengthen recommendation reporting**

Keep `k_evaluation.csv`, elbow, silhouette, Calinski-Harabasz, Davies-Bouldin plots, and `recommended_k.txt`. Add minimum/maximum candidate cluster sizes and state that final K remains a documented research choice.

- [ ] **Step 5: Run tests and commit**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_choose_k -v
python -m py_compile scripts/cluster_preparation/choose_k.py
git diff --check
```

Commit:

```powershell
git add scripts/cluster_preparation/choose_k.py scripts/cluster_preparation/test_choose_k.py
git commit -m "refactor: evaluate K on prepared matrix"
```

---

### Task 9: Make K-means retain every prepared row

**Files:**

- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**

- Consumes: verified prepared artifact and selected K.
- Produces: one K-means label and distance record for every prepared row.
- Guarantees: standardized-matrix identity, `2 <= K < n_samples`, stable row-position join, and zero null-driven row exclusions.

- [ ] **Step 1: Write failing prepared-input tests**

Replace complete-case behavior tests with:

```python
validated = kmeans.prepare_data(artifact_dir)
self.assertEqual(validated.row_count, len(source))
self.assertEqual(validated.feature_names, artifact.feature_names)
np.testing.assert_allclose(
    validated.values,
    artifact.standardized_features.to_numpy(dtype=float),
)
self.assertEqual(validated.metadata["source_row_position"].tolist(), list(range(len(source))))
```

Run analysis with one structural-null row and assert:

```python
self.assertEqual(len(result.tables.clustered_data), len(source))
self.assertFalse(result.tables.clustered_data["kmeans_cluster"].isna().any())
self.assertEqual(
    result.tables.clustered_data.loc[
        result.tables.clustered_data["had_structural_null"], "structural_null_count"
    ].min(),
    1,
)
```

Patch K-means module access to `DataFrame.dropna`, `DataFrame.fillna`, and `StandardScaler.fit` so the test fails if any is used during input preparation or fitting.

- [ ] **Step 2: Run focused K-means tests and confirm RED**

Run:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis -v
```

Expected: current input path expects a CSV and `validate_features()` removes rows.

- [ ] **Step 3: Replace raw-input selection and validation**

Refactor `ValidatedData` to carry:

```python
@dataclass(frozen=True)
class ValidatedData:
    artifact: PreparedArtifact
    values: np.ndarray
    feature_names: tuple[str, ...]
    metadata: pd.DataFrame
    row_count: int
```

Implement:

```python
def prepare_data(prepared_path: Path) -> ValidatedData:
    artifact = load_prepared_artifact(prepared_path)
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise DataValidationError("Prepared standardized matrix is not finite.")
    return ValidatedData(
        artifact=artifact,
        values=values,
        feature_names=artifact.feature_names,
        metadata=artifact.metadata.copy(),
        row_count=len(artifact.metadata),
    )
```

Remove active calls to `select_features()` and complete-case `validate_features()`. Keep any old helpers only until all tests are migrated, then delete them in this task.

- [ ] **Step 4: Validate K and fit unchanged reproducible K-means settings**

Before fitting:

```python
if not 2 <= k < validated.row_count:
    raise DataValidationError(
        f"K must satisfy 2 <= K < {validated.row_count}; received {k}."
    )
```

Continue using current `KMeans` settings and existing label reordering. Join labels/distances to metadata by `source_row_position`, assert one-to-one length, and fail if any cluster label is missing.

- [ ] **Step 5: Update CLI, preflight, and run config**

Replace CSV prompt/argument with prepared-directory prompt/argument:

```text
--prepared PATH
--k INTEGER
--output-root PATH
```

Preflight prints scenario, source path/hash, matrix hash, row count, feature count/order, rows with structural null, and confirms `rows excluded for null = 0`. `run_config.json` records manifest path/hash and standardized matrix hash.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis -v
python -m py_compile scripts/cluster_kmeans/kmeans_analysis.py
git diff --check
```

Commit:

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "refactor: cluster every prepared street row"
```

---

### Task 10: Preserve original-scale interpretation in K-means outputs

**Files:**

- Modify: `scripts/cluster_kmeans/kmeans_analysis.py`
- Modify: `scripts/cluster_kmeans/test_kmeans_analysis.py`

**Interfaces:**

- Consumes: fitted standardized centroids plus artifact original/filled matrices and scaler parameters.
- Produces: standardized and inverse-transformed centroids, null-aware profiles, and enriched clustered data.
- Guarantees: filled zeros do not erase the distinction between original non-null values and structural nulls in reporting.

- [ ] **Step 1: Write failing centroid/profile tests**

Assert:

```python
expected_original_centroids = preprocessing.inverse_transform_frame(
    fitted.standardized_centroids,
    artifact.scaler_parameters,
)
pd.testing.assert_frame_equal(
    tables.cluster_centroids_original_scale,
    expected_original_centroids,
    check_exact=False,
    atol=1e-12,
    rtol=1e-12,
)
self.assertEqual(
    set(tables.cluster_feature_profile.columns),
    {
        "cluster",
        "feature",
        "original_non_null_count",
        "original_structural_null_count",
        "original_structural_null_percent",
        "original_non_null_mean",
        "original_non_null_median",
        "filled_mean",
        "standardized_centroid",
        "inverse_transformed_centroid",
    },
)
```

Check `clustered_data.csv` contains identifiers, active risk, relevant original semantic columns, `kmeans_cluster`, `had_structural_null`, and `structural_null_count`.

- [ ] **Step 2: Run focused output tests and confirm RED**

Run the centroid/profile/export test classes in:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis -v
```

Expected: missing original-scale centroid and null-aware profile outputs.

- [ ] **Step 3: Calculate both centroid scales**

Name standardized centroid columns in artifact feature order. Use `inverse_transform_frame()` for semantic-scale centroids; do not fit or reconstruct a scaler in K-means.

- [ ] **Step 4: Build null-aware long-form profiles**

For every cluster and feature, use:

- `artifact.original_features` for original non-null count/mean/median and null count;
- `artifact.filled_features` for filled mean;
- fitted centers for standardized centroid;
- inverse transformation for semantic-scale centroid.

Use `NaN` for original non-null mean/median only when every original value in that cluster/feature was structural null. Do not convert this reporting `NaN` to zero.

- [ ] **Step 5: Export the confirmed output set**

Ensure result directories contain:

```text
clustered_data.csv
cluster_summary.csv
cluster_centroids_standardized.csv
cluster_centroids_original_scale.csv
cluster_feature_profile.csv
sample_distances.csv
model_metrics.csv
cluster_label_mapping.csv
preprocessing_audit.csv
run_config.json
run_report.txt
pca_cluster_scatter.png
cluster_size_bar.png
cluster_centroid_heatmap.png
cluster_feature_profiles.png
```

The centroid heatmap uses standardized centroids. The profile plot labels the unit used. Copy the preprocessing audit without changing it.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
python -m unittest scripts.cluster_kmeans.test_kmeans_analysis -v
python -m py_compile scripts/cluster_kmeans/kmeans_analysis.py
git diff --check
```

Commit:

```powershell
git add scripts/cluster_kmeans/kmeans_analysis.py scripts/cluster_kmeans/test_kmeans_analysis.py
git commit -m "feat: report null-aware K-means profiles"
```

---

### Task 11: Migrate primary and legacy HDBSCAN entry points

**Files:**

- Modify: `scripts/clustering/hdbscan_runner.py`
- Modify: `scripts/run_hdbscan_street_typology.py`
- Modify: `scripts/utils/config.py`
- Create: `scripts/clustering/test_hdbscan_runner.py`

**Interfaces:**

- Consumes: prepared artifact; standardized matrix for primary Version B and filled matrix for explicitly labelled sensitivity Version A.
- Produces: existing HDBSCAN outputs with preprocessing identity.
- Guarantees: no HDBSCAN feature discovery, imputation, median fill, or scaler fitting.

- [ ] **Step 1: Write failing HDBSCAN matrix-selection tests**

Assert:

```python
primary = hdbscan_runner.load_hdbscan_input(artifact_dir, "version_b")
sensitivity = hdbscan_runner.load_hdbscan_input(artifact_dir, "version_a")

np.testing.assert_allclose(
    primary.values,
    artifact.standardized_features.to_numpy(dtype=float),
)
np.testing.assert_allclose(
    sensitivity.values,
    artifact.filled_features.to_numpy(dtype=float),
)
self.assertEqual(primary.scale_label, "standardized_primary")
self.assertEqual(sensitivity.scale_label, "filled_raw_scale_sensitivity")
```

Patch `prepare_features`, `DataFrame.fillna`, and `StandardScaler.fit` to fail if invoked by `run_experiments()`.

- [ ] **Step 2: Run HDBSCAN tests and confirm RED**

Run:

```powershell
python -m unittest scripts.clustering.test_hdbscan_runner -v
```

Expected: current runner reads a raw CSV and prepares/scales internally.

- [ ] **Step 3: Replace HDBSCAN preprocessing**

Add:

```python
@dataclass(frozen=True)
class HdbscanInput:
    artifact: PreparedArtifact
    values: np.ndarray
    scale_label: str


def load_hdbscan_input(path: Path, version: str) -> HdbscanInput:
    artifact = load_prepared_artifact(path)
    if version == "version_b":
        frame = artifact.standardized_features
        label = "standardized_primary"
    elif version == "version_a":
        frame = artifact.filled_features
        label = "filled_raw_scale_sensitivity"
    else:
        raise ValueError(f"Unsupported HDBSCAN version: {version}")
    return HdbscanInput(artifact, frame.to_numpy(dtype=float, copy=True), label)
```

Change runner arguments from raw input/scenario lists to one prepared scenario path per run. Scenario and feature order come from the artifact. Remove `MISSING_VALUE_STRATEGY` as an active transformation control; report `structural_zero_from_prepared_artifact`.

- [ ] **Step 4: Replace the legacy monolith with a thin wrapper**

`scripts/run_hdbscan_street_typology.py` must contain only argument parsing, this warning, and delegation:

```text
DEPRECATED: run_hdbscan_street_typology.py now delegates to the modular
prepared-artifact HDBSCAN runner. Median imputation is no longer supported.
```

Require `--prepared`; delegate to `scripts.clustering.hdbscan_runner.main`. Delete legacy feature discovery, median imputation, scaler, fitting, plotting, and output code from the wrapper.

- [ ] **Step 5: Record preprocessing provenance in every run**

HDBSCAN run config and audit must include prepared manifest path/hash, matrix hash, scenario, feature order, scale label, structural-null policy, and algorithm parameters. Labels join to metadata by source row position.

- [ ] **Step 6: Run tests and commit**

Run:

```powershell
python -m unittest scripts.clustering.test_hdbscan_runner -v
python -m py_compile scripts/clustering/hdbscan_runner.py scripts/run_hdbscan_street_typology.py scripts/utils/config.py
git diff --check
```

Commit:

```powershell
git add scripts/clustering/hdbscan_runner.py scripts/clustering/test_hdbscan_runner.py scripts/run_hdbscan_street_typology.py scripts/utils/config.py
git commit -m "refactor: consume prepared artifacts in HDBSCAN"
```

---

### Task 12: Add cross-consumer contracts, v3 reconciliation, and workflow documentation

**Files:**

- Create: `tests/test_unified_clustering_workflow.py`
- Modify: `README.md`
- Verify: all files changed in Tasks 1–11

**Interfaces:**

- Consumes: one synthetic prepared artifact in automated tests and existing v3 high-risk PF/TF CSVs in an explicit regression smoke check.
- Produces: evidence that every consumer shares row count, feature order, matrix hash, and null policy.
- Guarantees: no unrelated generated artifacts are committed.

- [ ] **Step 1: Write a cross-consumer contract test**

Prepare one synthetic PF artifact, then patch only algorithm-heavy functions. Assert:

```python
self.assertEqual(pca_input.config["matrix_sha256"], choose_k_input.config["matrix_sha256"])
self.assertEqual(choose_k_input.config["matrix_sha256"], kmeans_input.artifact.config["matrix_sha256"])
self.assertEqual(kmeans_input.artifact.config["matrix_sha256"], hdbscan_input.artifact.config["matrix_sha256"])
self.assertEqual(pca_input.feature_names, choose_k_input.feature_names)
self.assertEqual(choose_k_input.feature_names, kmeans_input.feature_names)
self.assertEqual(kmeans_input.feature_names, hdbscan_input.artifact.feature_names)
self.assertEqual(len(kmeans_labels), len(prepared.metadata))
self.assertFalse(pd.Series(kmeans_labels).isna().any())
```

Patch `StandardScaler.fit` after artifact creation and confirm PCA, choose-K, K-means, and primary HDBSCAN input loading all succeed.

- [ ] **Step 2: Run the cross-consumer test and resolve only contract failures**

Run:

```powershell
python -m unittest tests.test_unified_clustering_workflow -v
```

Expected after Tasks 1–11: all contract tests pass.

- [ ] **Step 3: Add the confirmed run order to README**

Document exact commands:

```powershell
python scripts/clean_mt_update_ms_hev.py
python scripts/filter_high_risk_dataset.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned.csv --scenario PF --threshold 0.25
python scripts/filter_high_risk_dataset.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned.csv --scenario TF --threshold 0.25
python scripts/pearson_correlation/correlation_analysis.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF
python scripts/pearson_correlation/correlation_analysis.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF
python scripts/cluster_preparation/pca_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
python scripts/cluster_preparation/choose_k.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
python scripts/cluster_kmeans/kmeans_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF --k 4
```

State that TF repeats the last three commands with the TF artifact, PCA is diagnostic only, and changing source/threshold/exclusions requires a new output root and regeneration.

- [ ] **Step 4: Run automated regression suites**

Run:

```powershell
python -m unittest scripts.utils.test_clustering_preprocessing scripts.cluster_preparation.test_prepare_clustering_inputs scripts.pearson_correlation.test_correlation_analysis scripts.pearson_correlation.test_extract_high_columns scripts.cluster_preparation.test_pca_analysis scripts.cluster_preparation.test_choose_k scripts.cluster_kmeans.test_kmeans_analysis scripts.clustering.test_hdbscan_runner tests.test_unified_clustering_workflow -v
```

Expected: all tests pass.

- [ ] **Step 5: Compile every changed Python entry point**

Run:

```powershell
python -m py_compile scripts/utils/clustering_preprocessing.py scripts/utils/feature_utils.py scripts/utils/scenario_features.py scripts/cluster_preparation/prepare_clustering_inputs.py scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/choose_k.py scripts/pearson_correlation/correlation_analysis.py scripts/pearson_correlation/extract_high_columns.py scripts/cluster_kmeans/kmeans_analysis.py scripts/clustering/hdbscan_runner.py scripts/run_hdbscan_street_typology.py
```

Expected: exit code 0 with no output.

- [ ] **Step 6: Run a read-only v3 reconciliation check**

Use shared preparation in memory without writing an artifact:

```powershell
@'
from pathlib import Path
import pandas as pd
from scripts.utils.clustering_preprocessing import prepare_clustering_data

cases = (
    ("PF", Path("data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv"), 4100),
    ("TF", Path("data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv"), 13584),
)
for scenario, path, expected_rows in cases:
    data = pd.read_csv(path)
    prepared = prepare_clustering_data(data, scenario)
    assert len(prepared.metadata) == expected_rows
    assert len(prepared.standardized_features) == expected_rows
    assert not prepared.standardized_features.isna().any().any()
    print(
        scenario,
        "rows=", len(prepared.metadata),
        "features=", len(prepared.feature_names),
        "rows_removed_for_null=0",
    )
'@ | python -
```

Expected: PF reports 4,100 rows; TF reports 13,584 rows; both report `rows_removed_for_null=0`. The feature count is discovered and reported, not hard-coded.

- [ ] **Step 7: Review scope and provenance coverage**

Run:

```powershell
git diff --check
git status --short
rg -n "PF_FEATURES|TF_FEATURES|fillna\\(|dropna\\(|StandardScaler" scripts/cluster_preparation scripts/cluster_kmeans scripts/clustering scripts/utils
```

Review every match. Permitted active matches are:

- `fillna(0.0)` and `StandardScaler.fit_transform` only in canonical preprocessing;
- no null-filling/scaler fit in PCA, choose-K, K-means, or primary HDBSCAN;
- reporting-only `dropna` may remain when calculating statistics on original non-null values;
- no complete PF/TF feature tuple.

Confirm generated output files and existing user data remain unstaged.

- [ ] **Step 8: Commit final contracts and documentation**

```powershell
git add tests/test_unified_clustering_workflow.py README.md
git commit -m "test: verify unified clustering workflow"
```

- [ ] **Step 9: Perform final full-suite verification**

Run:

```powershell
python -m unittest discover -v
git diff --check
git status --short
```

Expected: all discoverable tests pass; no whitespace errors; only pre-existing user outputs/data may remain modified or untracked.
