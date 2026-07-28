# Shared Raw AB lossR Exclusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude raw PF/TF AB 2 km and 5 km loss-ratio columns from Pearson correlation and every clustering consumer while retaining their `_NOR` counterparts.

**Architecture:** Put one anchored, case-insensitive built-in rule in `scripts/utils/clustering_preprocessing.py`, the existing shared discovery boundary used by both correlation and clustering preparation. Persist matches as auditable `EXCLUDED` roles so every downstream consumer receives the same selected features without modifying source CSVs.

**Tech Stack:** Python 3, pandas, NumPy, scikit-learn, `unittest`, Windows PowerShell, Conda environment `2026master`.

## Global Constraints

- Exclude exactly `^(?:N_)?(?:PF|TF)AB(?:2k|5k)_lossR$`, case-insensitively.
- Preserve `PFAB2k_NOR`, `PFAB5k_NOR`, `TFAB2k_NOR`, and `TFAB5k_NOR`.
- Record built-in matches with `role=EXCLUDED` and `reason=raw_ab_lossR_replaced_by_normalized_NOR`.
- Keep identifier, Index/Risk, result/label, and shared-field precedence unchanged.
- Do not modify semantic cleaned CSVs or high-risk source CSVs.
- Do not add separate exclusion lists to correlation, PCA, choose-K, K-means, or HDBSCAN.
- Preserve explicit research-exclusion JSON behavior for other active features.
- Use `D:\anaconda3\Scripts\conda.exe run --no-capture-output -n 2026master python ...` for every Python verification command.
- Do not stage, overwrite, or delete existing user-generated outputs, prepared artifacts, or v4 data.

---

### Task 1: Add the shared built-in classification rule

**Files:**

- Modify: `scripts/utils/clustering_preprocessing.py:87-95`
- Modify: `scripts/utils/clustering_preprocessing.py:140-165`
- Modify: `scripts/utils/clustering_preprocessing.py:206-225`
- Test: `scripts/utils/test_clustering_preprocessing.py`
- Test: `scripts/pearson_correlation/test_correlation_analysis.py`
- Test: `tests/test_unified_clustering_workflow.py`

**Interfaces:**

- Consumes: source column names passed to `classify_column(...)` and `discover_scenario_features(...)`.
- Produces: `_builtin_exclusion_reason(name: str) -> str`, an `EXCLUDED` role, the stable reason `raw_ab_lossR_replaced_by_normalized_NOR`, and consumer-level regression contracts.
- Guarantees: exact raw AB lossR names are excluded before PF/TF candidate assignment; `_NOR` and incidental longer names remain eligible in correlation and every prepared-artifact consumer.

- [ ] **Step 1: Write the failing discovery test**

Add this method to `FeatureDiscoveryTests`:

```python
def test_raw_ab_lossr_is_builtin_excluded_but_normalized_ab_is_retained(self):
    raw_names = (
        "PFAB2k_lossR",
        "N_PFAB2k_lossR",
        "PFAB5k_lossR",
        "N_PFAB5k_lossR",
        "TFAB2k_lossR",
        "N_TFAB2k_lossR",
        "TFAB5k_lossR",
        "N_TFAB5k_lossR",
    )
    normalized_names = (
        "PFAB2k_NOR",
        "PFAB5k_NOR",
        "TFAB2k_NOR",
        "TFAB5k_NOR",
    )
    data = self.data.copy()
    for offset, name in enumerate(raw_names + normalized_names):
        data[name] = [-0.8 + offset * 0.01, 0.0, 0.8 - offset * 0.01]
    data["PFAB2k_lossR_extra"] = [-0.5, 0.0, 0.5]

    pf = preprocessing.discover_scenario_features(data, "PF")
    tf = preprocessing.discover_scenario_features(data, "TF")
    pf_roles = pf.column_roles.set_index("column")
    tf_roles = tf.column_roles.set_index("column")

    for name in raw_names:
        self.assertEqual(pf_roles.loc[name, "role"], "EXCLUDED")
        self.assertEqual(
            pf_roles.loc[name, "reason"],
            "raw_ab_lossR_replaced_by_normalized_NOR",
        )
        self.assertEqual(tf_roles.loc[name, "role"], "EXCLUDED")
    self.assertIn("PFAB2k_NOR", pf.feature_names)
    self.assertIn("PFAB5k_NOR", pf.feature_names)
    self.assertIn("TFAB2k_NOR", tf.feature_names)
    self.assertIn("TFAB5k_NOR", tf.feature_names)
    self.assertIn("PFAB2k_lossR_extra", pf.feature_names)
```

- [ ] **Step 2: Write the failing correlation consumer test**

Add to `CorrelationAnalysisTests`:

```python
def test_correlation_keeps_normalized_ab_and_excludes_raw_ab_lossr(self):
    data = pd.DataFrame(
        {
            "fid": [1, 2, 3],
            "MS_ID": ["a", "b", "c"],
            "PF_Index_Risk_equal": [0.4, 0.5, 0.6],
            "PFAB2k_lossR": [-0.8, 0.0, 0.8],
            "N_PFAB2k_lossR": [-0.7, 0.0, 0.7],
            "PFAB5k_lossR": [-0.6, 0.0, 0.6],
            "N_PFAB5k_lossR": [-0.5, 0.0, 0.5],
            "PFAB2k_NOR": [-1.0, 0.0, 1.0],
            "PFAB5k_NOR": [-0.9, 0.0, 0.9],
        }
    )

    _groups, _original, filled = ca.prepare_correlation_features(data, "PF")

    self.assertEqual(
        tuple(filled.columns),
        ("PFAB2k_NOR", "PFAB5k_NOR"),
    )
```

- [ ] **Step 3: Write the failing prepared-artifact consumer test**

Add these entries to the `self.source` fixture in
`UnifiedClusteringWorkflowTests.setUp()`:

```python
"PFAB2k_lossR": [-0.9, -0.5, -0.1, 0.2, 0.6, 0.9],
"N_PFAB2k_lossR": [-0.8, -0.4, 0.0, 0.3, 0.7, 1.0],
"PFAB5k_lossR": [-1.0, -0.6, -0.2, 0.1, 0.5, 0.8],
"N_PFAB5k_lossR": [-0.7, -0.3, 0.0, 0.4, 0.8, 0.9],
"PFAB2k_NOR": [-0.95, -0.55, -0.15, 0.25, 0.65, 0.95],
"PFAB5k_NOR": [-0.85, -0.45, -0.05, 0.35, 0.75, 1.0],
```

Then add:

```python
def test_raw_ab_lossr_never_reaches_any_prepared_consumer(self):
    artifact = pca_analysis.load_pca_input(self.artifact_dir)
    raw_names = {
        "PFAB2k_lossR",
        "N_PFAB2k_lossR",
        "PFAB5k_lossR",
        "N_PFAB5k_lossR",
    }

    self.assertTrue(raw_names.isdisjoint(artifact.feature_names))
    self.assertIn("PFAB2k_NOR", artifact.feature_names)
    self.assertIn("PFAB5k_NOR", artifact.feature_names)
    excluded = artifact.excluded_features.set_index("column")
    for name in raw_names:
        self.assertEqual(excluded.loc[name, "role"], "EXCLUDED")
        self.assertEqual(
            excluded.loc[name, "reason"],
            "raw_ab_lossR_replaced_by_normalized_NOR",
        )
```

- [ ] **Step 4: Run all three tests and confirm RED**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.utils.test_clustering_preprocessing.FeatureDiscoveryTests.test_raw_ab_lossr_is_builtin_excluded_but_normalized_ab_is_retained scripts.pearson_correlation.test_correlation_analysis.CorrelationAnalysisTests.test_correlation_keeps_normalized_ab_and_excludes_raw_ab_lossr tests.test_unified_clustering_workflow.UnifiedClusteringWorkflowTests.test_raw_ab_lossr_never_reaches_any_prepared_consumer -v
```

Expected: all three tests FAIL because raw AB lossR columns currently have `PF_FEATURE` or `TF_FEATURE` roles and enter both consumer matrices.

- [ ] **Step 5: Implement the minimal shared rule**

Add beside the existing PF/TF patterns:

```python
RAW_AB_LOSSR_PATTERN = re.compile(
    r"^(?:N_)?(?:PF|TF)AB(?:2k|5k)_lossR$",
    re.IGNORECASE,
)
RAW_AB_LOSSR_EXCLUSION_REASON = (
    "raw_ab_lossR_replaced_by_normalized_NOR"
)


def _builtin_exclusion_reason(name: str) -> str:
    """Return the stable reason for an intrinsic modelling exclusion."""
    if RAW_AB_LOSSR_PATTERN.fullmatch(name):
        return RAW_AB_LOSSR_EXCLUSION_REASON
    return ""
```

In `classify_column(...)`, after shared-field classification and before the PF pattern:

```python
if _builtin_exclusion_reason(name):
    return FieldRole.EXCLUDED
```

In `discover_scenario_features(...)`, initialize each record reason from the same helper:

```python
for name in source_columns:
    role = base_roles[name]
    reason = _builtin_exclusion_reason(name)
```

Keep the existing explicit-exclusion override:

```python
if name in excluded_names:
    role = FieldRole.EXCLUDED
    reason = "explicit_research_exclusion"
```

- [ ] **Step 6: Run shared and consumer tests and confirm GREEN**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.utils.test_clustering_preprocessing scripts.pearson_correlation.test_correlation_analysis tests.test_unified_clustering_workflow -v
```

Expected: all shared preprocessing, correlation, and unified consumer tests pass.

- [ ] **Step 7: Compile and check the task diff**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m py_compile scripts/utils/clustering_preprocessing.py scripts/utils/test_clustering_preprocessing.py scripts/pearson_correlation/test_correlation_analysis.py tests/test_unified_clustering_workflow.py
git diff --check
```

Expected: both commands exit `0`.

- [ ] **Step 8: Commit the shared rule and consumer contracts**

```powershell
git add scripts/utils/clustering_preprocessing.py scripts/utils/test_clustering_preprocessing.py scripts/pearson_correlation/test_correlation_analysis.py tests/test_unified_clustering_workflow.py
git diff --cached --check
git commit -m "feat: exclude raw AB lossR clustering features"
```

---

### Task 2: Document and reconcile the shared exclusion

**Files:**

- Modify: `README.md`

**Interfaces:**

- Consumes: the shared rule and regression contracts committed in Task 1.
- Produces: user-facing workflow documentation and real-data reconciliation evidence.
- Guarantees: v3 PF/TF each retain ten intended candidates, including both normalized AB variables, with four active-scenario raw AB candidates excluded.

- [ ] **Step 1: Document the built-in feature rule**

After the confirmed workflow command block in `README.md`, add:

```markdown
Raw `PF/TF AB2k/AB5k lossR` columns, including their `N_` variants, are
built-in exclusions in the shared feature classifier. Their `AB2k_NOR` and
`AB5k_NOR` counterparts remain eligible. Pearson correlation and clustering
preparation therefore use the same feature decision.
```

- [ ] **Step 2: Run a read-only v3 PF/TF feature reconciliation**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
@'
from pathlib import Path
import pandas as pd
from scripts.utils.clustering_preprocessing import prepare_semantic_features

cases = (
    ("PF", Path("data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv")),
    ("TF", Path("data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv")),
)
for scenario, path in cases:
    prepared = prepare_semantic_features(pd.read_csv(path), scenario)
    assert len(prepared.feature_names) == 10
    assert f"{scenario}AB2k_NOR" in prepared.feature_names
    assert f"{scenario}AB5k_NOR" in prepared.feature_names
    assert not any(
        name.casefold().endswith(("ab2k_lossr", "ab5k_lossr"))
        for name in prepared.feature_names
    )
    roles = prepared.discovery.column_roles.set_index("column")
    excluded = roles.loc[
        roles["reason"].eq("raw_ab_lossR_replaced_by_normalized_NOR")
    ]
    assert len(excluded) == 8
    print(scenario, "features=", len(prepared.feature_names), "raw_ab_excluded=4")
'@ | & 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -
```

Expected:

```text
PF features= 10 raw_ab_excluded=4
TF features= 10 raw_ab_excluded=4
```

The role table reports all eight source-wide raw PF/TF exclusions; each active scenario loses four candidates.

- [ ] **Step 3: Run the full regression and compile checks**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest discover -v
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m py_compile scripts/utils/clustering_preprocessing.py scripts/pearson_correlation/correlation_analysis.py scripts/cluster_preparation/prepare_clustering_inputs.py tests/test_unified_clustering_workflow.py
git diff --check
git status --short
```

Expected: the full suite passes, compilation and diff checks exit `0`, and only the intended source/test/README files are staged or unstaged. Existing generated outputs remain untouched.

- [ ] **Step 4: Commit the documentation**

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: document raw AB lossR exclusions"
```
