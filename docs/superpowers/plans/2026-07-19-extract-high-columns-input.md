# Extract High Columns Interactive Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prompt for the input CSV path when `extract_high_columns.py` is run without `-i/--input`, with blank input selecting the existing default.

**Architecture:** Add a focused `prompt_input_path()` boundary matching the other dataset scripts. Use `None` as the parser default so `main()` can choose between an explicit CLI path and the prompt result before passing either through the unchanged `resolve_input_path()` validation.

**Tech Stack:** Python 3.8, `argparse`, `pathlib`, `unittest`, `unittest.mock`, and existing pandas code.

## Global Constraints

- Keep `DEFAULT_INPUT` unchanged.
- Preserve `-i/--input` and its existing path-resolution semantics.
- Use an English prompt when `-i/--input` is omitted.
- Empty input selects the default; non-empty input is trimmed and used.
- Explicit `-i/--input` bypasses the prompt.
- Do not change scenario, removal, extraction, or output behavior.

---

### Task 1: Add failing interaction and main-flow tests

**Files:**
- Modify: `scripts/pearson_correlation/test_extract_high_columns.py`
- Test: `scripts/pearson_correlation/extract_high_columns.py`

**Interfaces:**
- Consumes: `DEFAULT_INPUT`, `parse_args()`, `resolve_input_path()`, and `main()`.
- Produces: Requirements for `prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path` and `parse_args().input: Optional[Path]`.

- [ ] **Step 1: Test prompt behavior and parsing**

```python
def test_input_prompt_uses_default_or_typed_path(self) -> None:
    self.assertTrue(
        hasattr(ehc, "prompt_input_path"),
        "script must define prompt_input_path",
    )
    with patch("builtins.input", return_value="") as mocked_input:
        self.assertEqual(ehc.prompt_input_path(), ehc.DEFAULT_INPUT)
    mocked_input.assert_called_once_with(
        f"Enter input CSV path [{ehc.DEFAULT_INPUT}]: "
    )
    with patch("builtins.input", return_value="  custom.csv  "):
        self.assertEqual(ehc.prompt_input_path(), Path("custom.csv"))

def test_input_argument_is_optional_and_accepts_path(self) -> None:
    with patch.object(sys, "argv", ["extract_high_columns.py"]):
        self.assertIsNone(ehc.parse_args().input)
    with patch.object(
        sys,
        "argv",
        ["extract_high_columns.py", "--input", "custom.csv"],
    ):
        self.assertEqual(ehc.parse_args().input, Path("custom.csv"))
```

- [ ] **Step 2: Test `main()` prompt and bypass branches**

Call `main()` with fixed scenario/removal arguments while mocking CSV header
loading and final extraction. For omitted input, assert `prompt_input_path()`
is called once and its return value reaches `resolve_input_path()`. For an
explicit input, make `prompt_input_path()` raise if called and assert the CLI
path reaches `resolve_input_path()`.

```python
arguments = [
    "extract_high_columns.py",
    "--scenario",
    "pf",
    "--remove",
    "PFAB5k_NOR",
]
with patch.object(sys, "argv", arguments):
    with patch.object(ehc, "prompt_input_path", return_value=Path("custom.csv")) as prompt:
        with patch.object(ehc, "resolve_input_path", return_value=resolved) as resolve:
            with patch.object(ehc.pd, "read_csv", return_value=self.data.iloc[:0]):
                with patch.object(ehc, "extract_dataset") as extract:
                    ehc.main()
prompt.assert_called_once_with()
resolve.assert_called_once_with(Path("custom.csv"))
extract.assert_called_once_with(
    resolved,
    "pf",
    ["PFAB5k_NOR"],
    ehc.DEFAULT_OUTPUT_ROOT,
)
```

- [ ] **Step 3: Run tests and confirm RED**

Run: `python -m unittest scripts.pearson_correlation.test_extract_high_columns -v`

Expected: failures because `prompt_input_path` is absent and the parser still supplies the fixed default.

### Task 2: Implement the input prompt

**Files:**
- Modify: `scripts/pearson_correlation/extract_high_columns.py`
- Test: `scripts/pearson_correlation/test_extract_high_columns.py`

**Interfaces:**
- Consumes: `DEFAULT_INPUT: Path` and `resolve_input_path(value: str | Path) -> Path`.
- Produces: `prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path`; omitted parser input is `None`.

- [ ] **Step 1: Add the prompt function**

```python
def prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path:
    """Prompt for an input CSV path, using the configured path when blank."""
    entered_path = input(f"Enter input CSV path [{default_path}]: ").strip()
    return Path(entered_path) if entered_path else default_path
```

- [ ] **Step 2: Update parser and main selection**

Set `type=Path`, `default=None`, and English help on `-i/--input`. Select before
resolution:

```python
raw_input_path = (
    args.input if args.input is not None else prompt_input_path()
)
input_path = resolve_input_path(raw_input_path)
```

- [ ] **Step 3: Run focused tests and confirm GREEN**

Run: `python -m unittest scripts.pearson_correlation.test_extract_high_columns -v`

Expected: all extraction tests pass.

### Task 3: Regression verification and commit

**Files:**
- Verify: `scripts/pearson_correlation/extract_high_columns.py`
- Verify: `scripts/pearson_correlation/test_extract_high_columns.py`

**Interfaces:**
- Consumes: Completed prompt and CLI behavior.
- Produces: Verification evidence and a scoped implementation commit.

- [ ] **Step 1: Run all relevant tests**

Run: `python -m unittest scripts.pearson_correlation.test_extract_high_columns tests.test_interactive_dataset_input scripts.pearson_correlation.test_correlation_analysis -v`

Expected: all tests pass.

- [ ] **Step 2: Check help and compile**

Run: `python scripts/pearson_correlation/extract_high_columns.py --help`

Expected: `-i INPUT, --input INPUT` states that omission prompts and blank uses the default.

Run: `python -m py_compile scripts/pearson_correlation/extract_high_columns.py scripts/pearson_correlation/test_extract_high_columns.py`

Expected: exit code 0.

- [ ] **Step 3: Review and commit only scoped files**

Run: `git diff --check`

```bash
git add scripts/pearson_correlation/extract_high_columns.py scripts/pearson_correlation/test_extract_high_columns.py
git commit -m "feat: prompt for extraction input path"
```

Existing output files and generated CSVs must remain unstaged.
