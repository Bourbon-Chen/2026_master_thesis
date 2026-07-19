# Single Correlation Scenario Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prompt for PF or TF on every correlation-analysis run and execute only the selected scenario.

**Architecture:** Add a focused `prompt_scenario() -> str` input boundary that normalizes valid input to uppercase and retries invalid input. Replace the two-scenario loop in `main()` with a one-entry selected-scenario mapping while reusing every existing validation, analysis, summary, and reporting function.

**Tech Stack:** Python 3.8, `unittest`, `unittest.mock`, and the script's existing pandas/numpy/matplotlib stack.

## Global Constraints

- The scenario prompt and validation message must be English.
- The scenario must be prompted on every run with no CLI bypass.
- Accept PF and TF case-insensitively and reject every other value.
- Validate and analyze only the selected scenario.
- Do not delete the other scenario's existing outputs.
- Preserve input-path handling, calculations, thresholds, charts, reports, and file naming.

---

### Task 1: Add and test the scenario prompt

**Files:**
- Modify: `tests/test_interactive_dataset_input.py`
- Modify: `scripts/pearson_correlation/correlation_analysis.py`

**Interfaces:**
- Consumes: `SCENARIO_FEATURE_GROUPS` keys `PF` and `TF`.
- Produces: `prompt_scenario() -> str`, returning `PF` or `TF`.

- [ ] **Step 1: Write a failing prompt test**

```python
def test_correlation_scenario_prompt_retries_and_normalizes(self):
    self.assertTrue(
        hasattr(CORRELATION, "prompt_scenario"),
        "script must define prompt_scenario",
    )
    with patch("builtins.input", side_effect=["invalid", "tf"]) as mocked_input:
        with patch("builtins.print") as mocked_print:
            self.assertEqual(CORRELATION.prompt_scenario(), "TF")
    self.assertEqual(mocked_input.call_count, 2)
    mocked_input.assert_called_with("Enter correlation scenario (pf/tf): ")
    mocked_print.assert_called_once_with("Scenario must be 'pf' or 'tf'.")
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python -m unittest tests.test_interactive_dataset_input.InteractiveInputTests.test_correlation_scenario_prompt_retries_and_normalizes -v`

Expected: assertion failure `script must define prompt_scenario`.

- [ ] **Step 3: Add the minimal prompt implementation**

```python
def prompt_scenario() -> str:
    while True:
        scenario = input("Enter correlation scenario (pf/tf): ").strip().upper()
        if scenario in SCENARIO_FEATURE_GROUPS:
            return scenario
        print("Scenario must be 'pf' or 'tf'.")
```

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run: `python -m unittest tests.test_interactive_dataset_input.InteractiveInputTests.test_correlation_scenario_prompt_retries_and_normalizes -v`

Expected: one test passes with `OK`.

### Task 2: Execute only the selected scenario

**Files:**
- Modify: `tests/test_interactive_dataset_input.py`
- Modify: `scripts/pearson_correlation/correlation_analysis.py`

**Interfaces:**
- Consumes: `prompt_scenario() -> str` and `SCENARIO_FEATURE_GROUPS`.
- Produces: A one-entry `selected_feature_groups` mapping used by validation, analysis, combined summary, and report generation.

- [ ] **Step 1: Write a failing main-flow test**

Use a temporary output directory and mock only heavy/output boundaries. Call
`main()` with `prompt_scenario()` returning `PF`, then assert:

```python
selected = {"PF": CORRELATION.PF_FEATURE_GROUPS}
mocked_validate.assert_called_once_with(data, selected)
mocked_analyze.assert_called_once_with(
    data,
    "PF",
    CORRELATION.PF_FEATURE_GROUPS,
    output_directory / "PF",
    CORRELATION.CORRELATION_THRESHOLD,
    CORRELATION.EXPECTED_RANGES,
)
mocked_summary.assert_called_once()
self.assertEqual(mocked_summary.call_args.args[0], {"PF": result})
mocked_report.assert_called_once_with(
    {"PF": result},
    selected,
    output_directory / "combined_summary" / "correlation_report.txt",
    CORRELATION.CORRELATION_THRESHOLD,
)
```

Also patch `prompt_scenario()` in the existing correlation `main()` input-path tests so those tests remain focused on input selection.

- [ ] **Step 2: Run the main-flow test and confirm RED**

Run: `python -m unittest tests.test_interactive_dataset_input.InteractiveInputTests.test_correlation_main_runs_only_selected_scenario -v`

Expected: failure because the current `main()` validates and analyzes both PF and TF.

- [ ] **Step 3: Replace the two-scenario loop with one selected scenario**

At the start of `main()`, call `scenario = prompt_scenario()`. Build:

```python
feature_groups = SCENARIO_FEATURE_GROUPS[scenario]
selected_feature_groups = {scenario: feature_groups}
```

Pass `selected_feature_groups` to `validate_columns()` and `write_text_report()`, call `analyze_scenario()` once, and set:

```python
scenario_results = {scenario: result}
```

- [ ] **Step 4: Run the focused interaction suite**

Run: `python -m unittest tests.test_interactive_dataset_input -v`

Expected: all interaction tests pass.

### Task 3: Regression verification and commit

**Files:**
- Verify: `scripts/pearson_correlation/correlation_analysis.py`
- Verify: `tests/test_interactive_dataset_input.py`

**Interfaces:**
- Consumes: Completed scenario selection behavior.
- Produces: Verification evidence with no unrelated output files staged.

- [ ] **Step 1: Run all relevant tests**

Run: `python -m unittest tests.test_interactive_dataset_input scripts.pearson_correlation.test_correlation_analysis scripts.pearson_correlation.test_extract_high_columns -v`

Expected: all tests pass.

- [ ] **Step 2: Compile modified Python files**

Run: `python -m py_compile scripts/pearson_correlation/correlation_analysis.py tests/test_interactive_dataset_input.py`

Expected: exit code 0 with no output.

- [ ] **Step 3: Review scope**

Run: `git diff --check`

Expected: no whitespace errors. Confirm existing modified output files and the generated high-risk CSV remain unstaged.

- [ ] **Step 4: Commit only implementation and tests**

```bash
git add scripts/pearson_correlation/correlation_analysis.py tests/test_interactive_dataset_input.py
git commit -m "feat: select one correlation scenario per run"
```
