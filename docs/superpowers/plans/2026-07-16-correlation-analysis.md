# Pearson Correlation Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `scripts/pearson_correlation/correlation_analysis.py` script that audits PF and TF candidate features, computes pairwise-complete Pearson correlations, creates tabular and graphical outputs, and never modifies or automatically removes input features.

**Architecture:** Keep configuration and orchestration in one directly runnable script, with small functions for validation, profiling, quality checks, pairwise calculations, tables, charts, summaries, and reporting. Use explicit finite-value masks per feature pair so correlation coefficients and `pairwise_n` share exactly the same observations. Use seaborn when installed and an annotated matplotlib implementation otherwise.

**Tech Stack:** Python 3, pathlib, pandas, NumPy, matplotlib, optional seaborn, unittest.

## Global Constraints

- Input is `data/MT_UPDATE_MS_HEV_v2_NORcleaned.csv`; output root is `outputs_correlation/`.
- PF and TF are analyzed separately, but every scenario's four theoretical groups are combined before correlation calculation.
- The threshold is strict: `abs(pearson_r) > 0.6`.
- Missing values are handled pairwise; there is no whole-table `dropna` and no imputation.
- Non-finite observations are warned about and excluded only from the affected pair's numerical calculation.
- Expected ranges are exposure `[0, 1]`, accessibility loss `[-1, 1]`, population loss `[-1, 0]`, and centrality influence `[0, 1]`.
- Constant and all-missing variables remain in all applicable outputs; undefined Pearson coefficients are `NaN` and never exceed the threshold.
- Do not run clustering, PCA, scaling, feature deletion, truncation, or source-data modification.
- Heatmaps use full labels, two-decimal annotations, fixed `[-1, 1]` color limits, a diverging palette, at least 300 dpi, and tight layout.

---

### Task 1: Lock the analytical behavior with tests

**Files:**
- Create: `scripts/pearson_correlation/test_correlation_analysis.py`
- Test: `scripts/pearson_correlation/test_correlation_analysis.py`

**Interfaces:**
- Consumes: planned public functions from `correlation_analysis.py`.
- Produces: executable specifications for validation, summaries, warnings, pairwise calculations, pair-table classification, combined summary, reports, and chart files.

- [ ] **Step 1: Write failing tests**

Create unittest cases that import `correlation_analysis`, construct small DataFrames containing missing, constant, non-numeric, infinite, and out-of-range values, and assert:

```python
matrix, counts = calculate_pairwise_correlation(frame, ["a", "b", "constant"])
self.assertEqual(counts.loc["a", "b"], 2)
self.assertAlmostEqual(matrix.loc["a", "b"], 1.0)
self.assertTrue(np.isnan(matrix.loc["a", "constant"]))
```

Also assert that column validation raises `ValueError`, summary columns are complete, the strict threshold excludes exactly `0.6`, A-B appears once, warnings identify every requested quality condition, maximum-pair summary fields are correct, the report contains the manual-review warning, and both plotting functions create non-empty PNG files.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_correlation_analysis -v`

Expected: import failure because `scripts/pearson_correlation/correlation_analysis.py` does not exist yet.

---

### Task 2: Implement configuration, validation, profiling, and data-quality checks

**Files:**
- Create: `scripts/pearson_correlation/correlation_analysis.py`
- Test: `scripts/pearson_correlation/test_correlation_analysis.py`

**Interfaces:**
- Produces: `load_data(Path) -> pd.DataFrame`, `validate_columns(pd.DataFrame, Mapping[str, Mapping[str, Sequence[str]]]) -> None`, `build_feature_group_mapping(Mapping[str, Sequence[str]]) -> dict[str, str]`, `create_feature_summary(...) -> pd.DataFrame`, and `check_data_quality(...) -> pd.DataFrame`.

- [ ] **Step 1: Implement top-level configuration**

Define `INPUT_FILE`, `OUTPUT_DIRECTORY`, `CORRELATION_THRESHOLD`, `IDENTIFIER_COLUMNS`, `PF_FEATURE_GROUPS`, `TF_FEATURE_GROUPS`, `SCENARIO_FEATURE_GROUPS`, and `EXPECTED_RANGES` at the start of the script. Resolve the project root with `Path(__file__).resolve().parents[2]`.

- [ ] **Step 2: Implement validation and feature summary**

Validate identifiers plus every configured candidate column and print all missing names before raising `ValueError`. For every feature, emit scenario, name, group, dtype, non-null count, missing count/percentage, distinct non-null count, and numeric descriptive fields; emit `NaN` statistics for non-numeric columns.

- [ ] **Step 3: Implement warnings**

Return a stable-schema warning table with `scenario`, `feature`, `theoretical_group`, `warning_type`, `severity`, `affected_count`, and `details`. Detect all-missing, exactly-one-distinct-value constant, non-numeric dtype, positive infinity, negative infinity, and finite observations outside the configured group range. Print each warning without altering values.

- [ ] **Step 4: Run the tests**

Run: `python -m unittest tests.test_correlation_analysis -v`

Expected: profiling and quality tests pass; unimplemented correlation/output tests remain failing.

---

### Task 3: Implement pairwise correlations and pair tables

**Files:**
- Modify: `scripts/pearson_correlation/correlation_analysis.py`
- Test: `scripts/pearson_correlation/test_correlation_analysis.py`

**Interfaces:**
- Produces: `calculate_pairwise_correlation(pd.DataFrame, Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]` and `build_correlation_pairs_table(...) -> pd.DataFrame`.

- [ ] **Step 1: Calculate each pair from one explicit mask**

Convert each configured series to numeric, build `np.isfinite(left) & np.isfinite(right)`, store its sum in the symmetric pair-count matrix, and compute Pearson only when at least two observations and both masked series have nonzero variance. Preserve undefined results as `NaN`.

- [ ] **Step 2: Create unique pair records**

Iterate only `i < j`. Add scenario, groups, coefficient, absolute coefficient, pair count, same-group flag, direction (`positive`, `negative`, `zero`, or `undefined`), and strict-threshold flag. Sort descending by absolute coefficient with undefined pairs last.

- [ ] **Step 3: Run the tests**

Run: `python -m unittest tests.test_correlation_analysis -v`

Expected: pairwise and pair-table tests pass.

---

### Task 4: Implement output tables, charts, combined summary, report, and main orchestration

**Files:**
- Modify: `scripts/pearson_correlation/correlation_analysis.py`
- Test: `scripts/pearson_correlation/test_correlation_analysis.py`

**Interfaces:**
- Produces: `save_correlation_outputs(...)`, `plot_full_heatmap(...)`, `plot_high_correlation_heatmap(...)`, `create_combined_summary(...)`, `write_text_report(...)`, `analyze_scenario(...)`, and `main()`.

- [ ] **Step 1: Save scenario CSV outputs**

Always create PF/TF directories and save summary, warning table, full matrix, all 45 unique pairs, all high pairs, within-group high pairs, and cross-group high pairs with indexes disabled except for the labeled correlation matrix.

- [ ] **Step 2: Draw charts**

Try importing seaborn inside a helper. If unavailable, use matplotlib `imshow`, a colorbar, and manual cell text. In the high-correlation chart, mask `abs(r) <= 0.6`, `NaN`, and the diagonal.

- [ ] **Step 3: Build combined summary and text report**

Aggregate counts and maximum finite absolute pair for each scenario. The report lists feature groups, data-quality findings, within/cross-group high pairs, the strongest pair, the non-causality caveat, and the requirement for manual theoretical review.

- [ ] **Step 4: Implement main**

Print loading, shape, PF/TF progress and counts, completion path, and exactly the recommended next step: manually review high-correlation pairs and determine final features. Validate all configured columns before creating analysis outputs.

- [ ] **Step 5: Run the complete test suite**

Run: `python -m unittest tests.test_correlation_analysis -v`

Expected: all tests pass without errors.

---

### Task 5: Verify against the real dataset

**Files:**
- Verify: `scripts/pearson_correlation/correlation_analysis.py`
- Verify: `outputs_correlation/PF/*`
- Verify: `outputs_correlation/TF/*`
- Verify: `outputs_correlation/combined_summary/*`

**Interfaces:**
- Consumes: the real 116,959-row, 45-column CSV.
- Produces: all 20 requested output artifacts and terminal progress output.

- [ ] **Step 1: Run syntax and tests**

Run: `python -m py_compile scripts/pearson_correlation/correlation_analysis.py`

Run: `python -m unittest tests.test_correlation_analysis -v`

Expected: compilation succeeds and all tests pass.

- [ ] **Step 2: Run the script end to end**

Run: `python scripts/pearson_correlation/correlation_analysis.py`

Expected: PF and TF each analyze 10 features, finish without exceptions, and print high/within/cross counts.

- [ ] **Step 3: Audit generated artifacts**

Check that every requested file exists and is non-empty; each all-pairs table has 45 rows; thresholds are strict; within plus cross counts equal total high counts; matrices are 10-by-10; PNG dimensions and DPI metadata are adequate; and the combined summary/report contain both scenarios.

- [ ] **Step 4: Review the final diff**

Run: `git status --short` and inspect `scripts/pearson_correlation/` plus `docs/superpowers/plans/2026-07-16-correlation-analysis.md`.

Expected: only the planned new files plus generated outputs appear; pre-existing user modifications remain untouched.
