# Pearson Correlation Package Relocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the Pearson correlation script and its tests under `scripts/pearson_correlation/` while preserving all analytical behavior and root-level data/output locations.

**Architecture:** Move the standalone analysis module and unittest module into one Python package. Resolve the project root two levels above the relocated analysis file so direct VSCode/script execution still reads `data/` and writes `outputs_correlation/` at the repository root.

**Tech Stack:** Python 3, pathlib, unittest, pandas, NumPy, matplotlib, optional seaborn.

## Global Constraints

- Final main script: `scripts/pearson_correlation/correlation_analysis.py`.
- Final tests: `scripts/pearson_correlation/test_correlation_analysis.py`.
- Package marker: `scripts/pearson_correlation/__init__.py`.
- Remove the former root `correlation_analysis.py` and `tests/` paths.
- Preserve `data/MT_UPDATE_MS_HEV_v2_NORcleaned.csv` and `outputs_correlation/` at the project root.
- Preserve all existing correlation calculations, output names, and tests.
- New direct command: `python scripts/pearson_correlation/correlation_analysis.py`.

---

### Task 1: Verify the target module path is initially absent

**Files:**
- Inspect: `scripts/pearson_correlation/test_correlation_analysis.py`

**Interfaces:**
- Produces: RED evidence that the requested package path has not yet been created.

- [ ] **Step 1: Run the target test module**

Run: `python -m unittest scripts.pearson_correlation.test_correlation_analysis -v`

Expected: import failure because `scripts/pearson_correlation/` does not exist.

---

### Task 2: Move the module and tests as one package

**Files:**
- Move: `correlation_analysis.py` to `scripts/pearson_correlation/correlation_analysis.py`
- Move: `tests/test_correlation_analysis.py` to `scripts/pearson_correlation/test_correlation_analysis.py`
- Replace: `tests/__init__.py` with `scripts/pearson_correlation/__init__.py`

**Interfaces:**
- Consumes: the existing public functions and unittest suite.
- Produces: importable module `scripts.pearson_correlation.correlation_analysis` and colocated tests.

- [ ] **Step 1: Move the three files without changing analytical logic**

Use repository-aware file moves so the former root files are removed and the new package contains the same code.

- [ ] **Step 2: Correct the test import**

Use:

```python
from scripts.pearson_correlation import correlation_analysis as ca
```

- [ ] **Step 3: Correct project-root resolution**

Use:

```python
PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
```

- [ ] **Step 4: Run tests**

Run: `python -m unittest scripts.pearson_correlation.test_correlation_analysis -v`

Expected: 10 tests pass.

---

### Task 3: Update documentation and verify direct execution

**Files:**
- Modify: `docs/superpowers/plans/2026-07-16-correlation-analysis.md`
- Verify: `scripts/pearson_correlation/correlation_analysis.py`

**Interfaces:**
- Produces: accurate paths/commands and unchanged root-level outputs.

- [ ] **Step 1: Replace obsolete paths and commands in the original plan**

Document `scripts/pearson_correlation/correlation_analysis.py`, its colocated test module, `Path(__file__).resolve().parents[2]`, and the new direct command.

- [ ] **Step 2: Compile and run from the new path**

Run: `python -m py_compile scripts/pearson_correlation/correlation_analysis.py scripts/pearson_correlation/test_correlation_analysis.py`

Run: `python scripts/pearson_correlation/correlation_analysis.py`

Expected: the 116,959-row dataset is loaded from root `data/`, and requested outputs are regenerated under root `outputs_correlation/`.

- [ ] **Step 3: Audit relocation**

Assert the new three files exist, the old root script and `tests/` directory do not exist, all 10 tests pass, and the combined summary still contains PF and TF.
