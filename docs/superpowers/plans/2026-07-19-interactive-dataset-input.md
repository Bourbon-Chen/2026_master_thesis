# Interactive Dataset Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add consistent optional `-i/--input` handling and English interactive dataset-path prompts to the filtering and Pearson correlation scripts.

**Architecture:** Each script keeps its own existing default input constant and adds a focused `prompt_input_path(default_path: Path) -> Path` function. Argument parsing uses `None` to distinguish an omitted input option from an explicit one; `main()` prompts only for the omitted case and otherwise preserves the existing data-processing flow.

**Tech Stack:** Python 3 standard library (`argparse`, `pathlib`, `unittest`, `unittest.mock`) plus the correlation script's existing pandas/matplotlib/numpy dependencies.

## Global Constraints

- All new terminal prompt text must be English.
- Pressing Enter at the input-path prompt must select the script's existing default input path.
- A typed path must override the default path.
- Supplying `-i/--input` must bypass the prompt.
- Preserve the filtering script's existing uncommitted update to `data/MT_UPDATE_MS_HEV_v2_NORcleaned.csv`.
- Do not change filtering, correlation, reporting, charting, or output-path behavior.

---

### Task 1: Add failing input-selection tests

**Files:**
- Create: `tests/test_interactive_dataset_input.py`
- Test: `scripts/filter_high_risk_dataset.py`
- Test: `scripts/pearson_correlation/correlation_analysis.py`

**Interfaces:**
- Consumes: Each script module loaded directly from its filesystem path.
- Produces: Behavioral requirements for `prompt_input_path(default_path: Path) -> Path` and `parse_args()` in both scripts.

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FILTER = load_module("filter_high_risk_dataset", "scripts/filter_high_risk_dataset.py")
CORRELATION = load_module(
    "correlation_analysis",
    "scripts/pearson_correlation/correlation_analysis.py",
)


class InteractiveInputTests(unittest.TestCase):
    def assert_prompt_behavior(self, module, default_constant):
        self.assertTrue(
            hasattr(module, "prompt_input_path"),
            "script must define prompt_input_path",
        )
        with patch("builtins.input", return_value="") as mocked_input:
            self.assertEqual(module.prompt_input_path(default_constant), default_constant)
        self.assertEqual(
            mocked_input.call_args.args[0],
            f"Enter input CSV path [{default_constant}]: ",
        )

        custom = Path("data/custom.csv")
        with patch("builtins.input", return_value=f"  {custom}  "):
            self.assertEqual(module.prompt_input_path(default_constant), custom)

    def assert_argument_behavior(self, module):
        self.assertTrue(
            hasattr(module, "parse_args"),
            "script must define parse_args",
        )
        with patch.object(sys, "argv", ["script.py"]):
            self.assertIsNone(module.parse_args().input)
        with patch.object(sys, "argv", ["script.py", "--input", "data/custom.csv"]):
            self.assertEqual(module.parse_args().input, Path("data/custom.csv"))

    def test_filter_input_behavior(self):
        self.assert_prompt_behavior(FILTER, FILTER.DEFAULT_INPUT)
        self.assert_argument_behavior(FILTER)

    def test_correlation_input_behavior(self):
        self.assert_prompt_behavior(CORRELATION, CORRELATION.INPUT_FILE)
        self.assert_argument_behavior(CORRELATION)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the missing behavior fails**

Run: `python -m unittest tests.test_interactive_dataset_input -v`

Expected: both test methods fail with `script must define prompt_input_path`.

### Task 2: Implement the filtering-script prompt

**Files:**
- Modify: `scripts/filter_high_risk_dataset.py`
- Test: `tests/test_interactive_dataset_input.py`

**Interfaces:**
- Consumes: `DEFAULT_INPUT: Path` and the existing `parse_args()`/`main()` flow.
- Produces: `prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path`; `parse_args().input` is `None` when omitted.

- [ ] **Step 1: Add the minimal prompt function**

```python
def prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path:
    entered_path = input(f"Enter input CSV path [{default_path}]: ").strip()
    return Path(entered_path) if entered_path else default_path
```

- [ ] **Step 2: Make omission trigger the prompt**

Change the `--input` default to `None`, update its English help text, and select the path in `main()`:

```python
input_path = (
    args.input if args.input is not None else prompt_input_path()
).resolve()
```

- [ ] **Step 3: Run the focused filter test**

Run: `python -m unittest tests.test_interactive_dataset_input.InteractiveInputTests.test_filter_input_behavior -v`

Expected: `OK`.

### Task 3: Implement the correlation-script prompt and CLI option

**Files:**
- Modify: `scripts/pearson_correlation/correlation_analysis.py`
- Test: `tests/test_interactive_dataset_input.py`

**Interfaces:**
- Consumes: `INPUT_FILE: Path` and the existing `load_data()`/`main()` flow.
- Produces: `prompt_input_path(default_path: Path = INPUT_FILE) -> Path` and `parse_args() -> argparse.Namespace` with optional `input: Optional[Path]`.

- [ ] **Step 1: Import and configure argparse**

```python
import argparse


def prompt_input_path(default_path: Path = INPUT_FILE) -> Path:
    entered_path = input(f"Enter input CSV path [{default_path}]: ").strip()
    return Path(entered_path) if entered_path else default_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Pearson correlations among PF and TF clustering candidates."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=f"Input CSV path (prompted when omitted; blank uses: {INPUT_FILE})",
    )
    return parser.parse_args()
```

- [ ] **Step 2: Select the input path at the start of main**

```python
args = parse_args()
input_file = (
    args.input if args.input is not None else prompt_input_path()
).resolve()
print("Loading dataset...")
data = load_data(input_file)
```

- [ ] **Step 3: Run the focused correlation test**

Run: `python -m unittest tests.test_interactive_dataset_input.InteractiveInputTests.test_correlation_input_behavior -v`

Expected: `OK`.

### Task 4: Verify the complete behavior

**Files:**
- Verify: `scripts/filter_high_risk_dataset.py`
- Verify: `scripts/pearson_correlation/correlation_analysis.py`
- Verify: `tests/test_interactive_dataset_input.py`

**Interfaces:**
- Consumes: Both completed CLI/prompt implementations.
- Produces: Evidence that prompt defaults, custom paths, explicit CLI paths, and syntax all work.

- [ ] **Step 1: Run the full focused test module**

Run: `python -m unittest tests.test_interactive_dataset_input -v`

Expected: two tests pass and the command ends with `OK`.

- [ ] **Step 2: Check both help screens**

Run: `python scripts/filter_high_risk_dataset.py --help`

Run: `python scripts/pearson_correlation/correlation_analysis.py --help`

Expected: both help screens list `-i INPUT, --input INPUT` and explain the prompt/default behavior in English.

- [ ] **Step 3: Compile both scripts and the test**

Run: `python -m py_compile scripts/filter_high_risk_dataset.py scripts/pearson_correlation/correlation_analysis.py tests/test_interactive_dataset_input.py`

Expected: exit code 0 with no output.

- [ ] **Step 4: Review the final diff**

Run: `git diff --check`

Expected: no whitespace errors. Confirm the filtering script's pre-existing default filename update remains present and no processing logic changed.

- [ ] **Step 5: Commit only the implementation files**

```bash
git add scripts/filter_high_risk_dataset.py scripts/pearson_correlation/correlation_analysis.py tests/test_interactive_dataset_input.py
git commit -m "feat: prompt for dataset input paths"
```
