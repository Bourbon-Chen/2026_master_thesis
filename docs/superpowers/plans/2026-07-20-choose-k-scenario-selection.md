# Choose K Scenario Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `choose_k.py` accept case-insensitive PF/TF command-line values and prompt interactively when the scenario argument is omitted.

**Architecture:** Reuse `normalize_scenario()` as the single normalization boundary for argparse and a new retrying prompt. Keep `main()` responsible for choosing between the supplied normalized value and the prompt, after which the existing scenario-to-input-path and analysis flow remains unchanged.

**Tech Stack:** Python 3, argparse, unittest, unittest.mock

## Global Constraints

- Preserve all clustering features, K-means parameters, metrics, recommendation logic, input data, and output directory structure.
- Accept `PF`, `TF`, `pf`, and `tf` through `--scenario`.
- Prompt repeatedly for PF or TF only when `--scenario` is omitted.
- Keep automation non-interactive whenever `--scenario` is supplied.

---

### Task 1: Scenario parsing and interactive prompt

**Files:**
- Modify: `scripts/cluster_preparation/test_choose_k.py`
- Modify: `scripts/cluster_preparation/choose_k.py:28-34,70-78,407-417`

**Interfaces:**
- Consumes: `normalize_scenario(value: str) -> str`
- Produces: `prompt_scenario() -> str`; `parse_args().scenario` is uppercase PF/TF when supplied and `None` when omitted

- [ ] **Step 1: Write failing tests for parsing and prompting**

Add `sys` and `patch` imports, then add these tests to `ChooseKTests`:

```python
def test_parse_args_accepts_case_insensitive_scenario(self) -> None:
    for entered, expected in (("PF", "PF"), ("pf", "PF"), ("TF", "TF"), ("tf", "TF")):
        with self.subTest(entered=entered):
            with patch.object(sys, "argv", ["choose_k.py", "--scenario", entered]):
                self.assertEqual(choose_k.parse_args().scenario, expected)

def test_parse_args_leaves_omitted_scenario_unset(self) -> None:
    with patch.object(sys, "argv", ["choose_k.py"]):
        self.assertIsNone(choose_k.parse_args().scenario)

def test_prompt_scenario_retries_and_normalizes(self) -> None:
    with patch("builtins.input", side_effect=["invalid", " tf "]), patch(
        "builtins.print"
    ) as mocked_print:
        self.assertEqual(choose_k.prompt_scenario(), "TF")

    mocked_print.assert_called_once_with("Please enter PF or TF.")
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_choose_k.ChooseKTests.test_parse_args_accepts_case_insensitive_scenario scripts.cluster_preparation.test_choose_k.ChooseKTests.test_parse_args_leaves_omitted_scenario_unset scripts.cluster_preparation.test_choose_k.ChooseKTests.test_prompt_scenario_retries_and_normalizes -v
```

Expected: failures because lowercase `tf` is rejected, the omitted scenario is currently `PF`, and `prompt_scenario` does not exist.

- [ ] **Step 3: Implement normalization and prompt behavior**

Delete the unused `INPUT_FILE` and obsolete `SCENARIO` constants. Add after `normalize_scenario()`:

```python
def prompt_scenario() -> str:
    while True:
        value = input("Select scenario (PF/TF): ")
        try:
            return normalize_scenario(value)
        except ValueError:
            print("Please enter PF or TF.")
```

Change the scenario argument to:

```python
parser.add_argument(
    "-s",
    "--scenario",
    type=normalize_scenario,
    choices=sorted(SCENARIO_CONFIG),
    default=None,
    help="Scenario to analyze: PF or TF. Prompted when omitted.",
)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command again.

Expected: all three tests pass.

### Task 2: Main entry-point selection behavior

**Files:**
- Modify: `scripts/cluster_preparation/test_choose_k.py`
- Modify: `scripts/cluster_preparation/choose_k.py:456-471`

**Interfaces:**
- Consumes: `prompt_scenario() -> str`, `DEFAULT_INPUT_PATHS`, parsed argparse namespace
- Produces: `main()` calls `run_analysis()` with the selected uppercase scenario and matching default input path

- [ ] **Step 1: Write failing tests for main orchestration**

Add `Namespace` and these tests:

```python
def test_main_prompts_when_scenario_is_omitted(self) -> None:
    args = Namespace(
        scenario=None,
        input=None,
        output_root=Path("output"),
        k_min=2,
        k_max=3,
        silhouette_sample_size=10000,
    )
    with patch.object(choose_k, "parse_args", return_value=args), patch.object(
        choose_k, "prompt_scenario", return_value="TF"
    ) as mocked_prompt, patch.object(choose_k, "run_analysis") as mocked_run:
        choose_k.main()

    mocked_prompt.assert_called_once_with()
    mocked_run.assert_called_once_with(
        choose_k.DEFAULT_INPUT_PATHS["TF"],
        "TF",
        args.output_root,
        range(2, 4),
        10000,
    )

def test_main_does_not_prompt_when_scenario_is_supplied(self) -> None:
    args = Namespace(
        scenario="PF",
        input=None,
        output_root=Path("output"),
        k_min=2,
        k_max=3,
        silhouette_sample_size=10000,
    )
    with patch.object(choose_k, "parse_args", return_value=args), patch.object(
        choose_k, "prompt_scenario"
    ) as mocked_prompt, patch.object(choose_k, "run_analysis") as mocked_run:
        choose_k.main()

    mocked_prompt.assert_not_called()
    mocked_run.assert_called_once_with(
        choose_k.DEFAULT_INPUT_PATHS["PF"],
        "PF",
        args.output_root,
        range(2, 4),
        10000,
    )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_choose_k.ChooseKTests.test_main_prompts_when_scenario_is_omitted scripts.cluster_preparation.test_choose_k.ChooseKTests.test_main_does_not_prompt_when_scenario_is_supplied -v
```

Expected: the omitted-scenario test fails because `main()` passes `None` to `normalize_scenario()` rather than prompting.

- [ ] **Step 3: Implement main selection behavior**

Replace the scenario assignment in `main()` with:

```python
scenario = args.scenario if args.scenario is not None else prompt_scenario()
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command again.

Expected: both tests pass.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
python -m unittest scripts.cluster_preparation.test_choose_k -v
python scripts/cluster_preparation/choose_k.py --help
git diff --check -- scripts/cluster_preparation/choose_k.py scripts/cluster_preparation/test_choose_k.py
```

Expected: the full test module passes; help says the scenario is prompted when omitted; diff check exits successfully.

- [ ] **Step 6: Commit implementation**

```powershell
git add -- scripts/cluster_preparation/choose_k.py scripts/cluster_preparation/test_choose_k.py docs/superpowers/plans/2026-07-20-choose-k-scenario-selection.md
git commit -m "feat: prompt for choose-k scenario"
```
