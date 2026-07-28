# PCA Managed Output Overwrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow repeated PCA runs to replace only the ten PCA-managed files in `output_pca/pf` or `output_pca/tf` while preserving unrelated entries and protecting the previous result from generation failures.

**Architecture:** Generate every PCA output in a temporary directory created under the selected output root so staged files and final files are on the same filesystem. Verify the complete managed set, then replace only those filenames in the scenario directory; never clear or recursively replace the scenario directory.

**Tech Stack:** Python 3, `pathlib`, `tempfile.TemporaryDirectory`, pandas, NumPy, Matplotlib, scikit-learn, `unittest`, Windows PowerShell, Conda environment `2026master`.

## Global Constraints

- The default destinations remain `output_pca/pf` and `output_pca/tf`.
- Preserve the existing optional `--output-root` CLI argument.
- Replace exactly ten managed files: `pca_summary.csv`, `pca_loadings.csv`, `pca_scores.csv`, `explained_variance.png`, `cumulative_variance.png`, `pca_scatter.png`, `loading_plot.png`, `correlation_circle.png`, `pca_report.txt`, and `preprocessing_reference.json`.
- Preserve every unmanaged file and subdirectory in the final scenario directory.
- Do not publish any staged file unless all ten managed files were generated as regular files.
- Do not alter PCA inputs, feature selection, structural-null handling, standardization, choose-K, K-means, HDBSCAN, or their output policies.
- Use `D:\anaconda3\Scripts\conda.exe run --no-capture-output -n 2026master python ...` for every Python verification command.
- Do not stage, overwrite, or delete existing user-generated repository outputs while implementing or testing.

---

### Task 1: Stage and publish the complete managed PCA output set

**Files:**

- Modify: `scripts/cluster_preparation/pca_analysis.py:9-27`
- Modify: `scripts/cluster_preparation/pca_analysis.py:349-386`
- Test: `scripts/cluster_preparation/test_pca_analysis.py:15-60`
- Test: `scripts/cluster_preparation/test_pca_analysis.py:136-149`

**Interfaces:**

- Consumes: a verified `PreparedArtifact`, its `PcaResult`, and the requested `Path` output root.
- Produces: `PCA_MANAGED_FILENAMES: tuple[str, ...]`, `_write_pca_outputs(artifact: PreparedArtifact, result: PcaResult, output_dir: Path) -> None`, and `_publish_pca_outputs(staging_dir: Path, output_dir: Path) -> None`.
- Guarantees: generation failures leave the existing final managed files unchanged; successful runs replace only the complete managed set; final scenario paths that are non-directories are rejected.

- [ ] **Step 1: Make patched plot functions create real staged files**

Replace the test-only plot patch helper with a writer that exercises the filesystem boundary:

```python
    def patch_plots(
        self,
        stack: ExitStack,
        failing_function: str | None = None,
    ) -> None:
        def write_placeholder_plot(*args: object) -> None:
            Path(args[-1]).write_bytes(b"test plot")

        for function_name in PLOT_FUNCTIONS:
            side_effect: object
            if function_name == failing_function:
                side_effect = RuntimeError("simulated PCA output failure")
            else:
                side_effect = write_placeholder_plot
            stack.enter_context(
                patch.object(
                    pca,
                    function_name,
                    side_effect=side_effect,
                )
            )
```

This keeps plotting fast while ensuring staging verification observes real files rather than assertions on mocks.

- [ ] **Step 2: Replace the immutability test with a failing repeated-publication test**

Add this hand-derived test constant beside `PLOT_FUNCTIONS`:

```python
EXPECTED_MANAGED_FILENAMES = (
    "pca_summary.csv",
    "pca_loadings.csv",
    "pca_scores.csv",
    "explained_variance.png",
    "cumulative_variance.png",
    "pca_scatter.png",
    "loading_plot.png",
    "correlation_circle.png",
    "pca_report.txt",
    "preprocessing_reference.json",
)
```

Replace `test_run_analysis_rejects_an_existing_final_output_directory` with:

```python
    def test_repeated_run_replaces_managed_outputs_and_preserves_unmanaged_entries(
        self,
    ) -> None:
        output_root = self.root / "output_pca"
        with ExitStack() as stack:
            self.patch_plots(stack)
            output_dir = pca.run_analysis(self.artifact_dir, output_root)

        stale_report = output_dir / "pca_report.txt"
        stale_report.write_text("stale PCA report", encoding="utf-8")
        sentinel = output_dir / "manual_notes.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        manual_directory = output_dir / "manual_review"
        manual_directory.mkdir()
        manual_file = manual_directory / "decision.txt"
        manual_file.write_text("preserve directory", encoding="utf-8")

        with ExitStack() as stack:
            self.patch_plots(stack)
            repeated_output_dir = pca.run_analysis(
                self.artifact_dir,
                output_root,
            )

        self.assertEqual(repeated_output_dir, output_root / "pf")
        self.assertNotEqual(
            stale_report.read_text(encoding="utf-8"),
            "stale PCA report",
        )
        for filename in EXPECTED_MANAGED_FILENAMES:
            self.assertTrue((output_dir / filename).is_file(), filename)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(
            manual_file.read_text(encoding="utf-8"),
            "preserve directory",
        )
```

The production regression caught by this test is either restoring the existing-directory guard or deleting the scenario directory before publishing.

- [ ] **Step 3: Add a failing generation-rollback test**

Add:

```python
    def test_generation_failure_leaves_previous_managed_outputs_unchanged(
        self,
    ) -> None:
        output_root = self.root / "output_pca"
        with ExitStack() as stack:
            self.patch_plots(stack)
            output_dir = pca.run_analysis(self.artifact_dir, output_root)
        before = {
            filename: (output_dir / filename).read_bytes()
            for filename in EXPECTED_MANAGED_FILENAMES
        }

        with ExitStack() as stack:
            self.patch_plots(
                stack,
                failing_function="plot_cumulative_variance",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated PCA output failure",
            ):
                pca.run_analysis(self.artifact_dir, output_root)

        after = {
            filename: (output_dir / filename).read_bytes()
            for filename in EXPECTED_MANAGED_FILENAMES
        }
        self.assertEqual(after, before)
```

The production regression caught by this test is writing directly into the final directory before generation completes.

- [ ] **Step 4: Add a failing non-directory destination test**

Add:

```python
    def test_existing_scenario_path_must_be_a_directory(self) -> None:
        output_root = self.root / "output_pca"
        output_root.mkdir()
        scenario_path = output_root / "pf"
        scenario_path.write_text("not a directory", encoding="utf-8")

        with self.assertRaisesRegex(
            NotADirectoryError,
            "PCA final output path is not a directory",
        ):
            pca.run_analysis(self.artifact_dir, output_root)

        self.assertEqual(
            scenario_path.read_text(encoding="utf-8"),
            "not a directory",
        )
```

The production regression caught by this test is treating any existing path as a publishable scenario directory.

- [ ] **Step 5: Add a failing incomplete-staging publication test**

Add:

```python
    def test_publish_rejects_an_incomplete_staging_set_before_replacement(
        self,
    ) -> None:
        staging_dir = self.root / "staging"
        staging_dir.mkdir()
        output_dir = self.root / "output_pca" / "pf"
        output_dir.mkdir(parents=True)
        old_report = output_dir / "pca_report.txt"
        old_report.write_text("previous report", encoding="utf-8")
        for filename in EXPECTED_MANAGED_FILENAMES:
            if filename != "pca_report.txt":
                (staging_dir / filename).write_bytes(b"new staged output")

        with self.assertRaisesRegex(
            RuntimeError,
            "PCA staging did not produce every managed output",
        ):
            pca._publish_pca_outputs(staging_dir, output_dir)

        self.assertEqual(
            old_report.read_text(encoding="utf-8"),
            "previous report",
        )
        self.assertFalse((output_dir / "pca_summary.csv").exists())
```

The production regression caught by this test is starting the replacement loop before validating the complete managed set.

- [ ] **Step 6: Run the four changed-behavior tests and confirm RED**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.cluster_preparation.test_pca_analysis.PcaAnalysisTests.test_repeated_run_replaces_managed_outputs_and_preserves_unmanaged_entries scripts.cluster_preparation.test_pca_analysis.PcaAnalysisTests.test_generation_failure_leaves_previous_managed_outputs_unchanged scripts.cluster_preparation.test_pca_analysis.PcaAnalysisTests.test_existing_scenario_path_must_be_a_directory scripts.cluster_preparation.test_pca_analysis.PcaAnalysisTests.test_publish_rejects_an_incomplete_staging_set_before_replacement -v
```

Expected: all four tests fail. The first two encounter `FileExistsError` on their repeated run, the third receives `FileExistsError` instead of the required `NotADirectoryError`, and the fourth reports that `_publish_pca_outputs` does not exist.

- [ ] **Step 7: Add the managed filename contract and staging import**

Add the import:

```python
from tempfile import TemporaryDirectory
```

Add beside `DEFAULT_OUTPUT_ROOT`:

```python
PCA_MANAGED_FILENAMES = (
    "pca_summary.csv",
    "pca_loadings.csv",
    "pca_scores.csv",
    "explained_variance.png",
    "cumulative_variance.png",
    "pca_scatter.png",
    "loading_plot.png",
    "correlation_circle.png",
    "pca_report.txt",
    "preprocessing_reference.json",
)
```

- [ ] **Step 8: Extract one complete-output writer**

Add immediately before `run_analysis`:

```python
def _write_pca_outputs(
    artifact: PreparedArtifact,
    result: PcaResult,
    output_dir: Path,
) -> None:
    """Generate the complete managed PCA output set in one directory."""
    save_tables(artifact, result, output_dir)
    plot_scree(result.summary, output_dir / "explained_variance.png")
    plot_cumulative_variance(
        result.summary,
        output_dir / "cumulative_variance.png",
    )
    plot_scatter(result, output_dir / "pca_scatter.png")
    plot_loading(result, output_dir / "loading_plot.png")
    plot_correlation_circle(
        result,
        output_dir / "correlation_circle.png",
    )
    generate_report(artifact, result, output_dir / "pca_report.txt")
    write_preprocessing_reference(
        artifact,
        output_dir / "preprocessing_reference.json",
    )
```

- [ ] **Step 9: Add complete-set validation and managed publication**

Add after `_write_pca_outputs`:

```python
def _publish_pca_outputs(staging_dir: Path, output_dir: Path) -> None:
    """Replace only the complete PCA-managed set in the final directory."""
    missing = [
        filename
        for filename in PCA_MANAGED_FILENAMES
        if not (staging_dir / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            "PCA staging did not produce every managed output: "
            + ", ".join(missing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in PCA_MANAGED_FILENAMES:
        (staging_dir / filename).replace(output_dir / filename)
```

This helper deliberately moves only named files. It does not iterate over or delete final-directory contents.

- [ ] **Step 10: Replace the directory guard and direct writes with staged publication**

Change the beginning and publication section of `run_analysis` to:

```python
def run_analysis(prepared_path: Path, output_root: Path) -> Path:
    print("Loading prepared artifact...")
    artifact = load_pca_input(prepared_path)
    scenario = str(artifact.config["scenario"])
    output_root = Path(output_root)
    output_dir = output_root / scenario.lower()
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(
            f"PCA output root is not a directory: {output_root}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"PCA final output path is not a directory: {output_dir}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    values = artifact.standardized_features.to_numpy(dtype=float, copy=True)
    print("Running PCA...")
    result = run_pca(values, artifact.feature_names)

    with TemporaryDirectory(
        prefix=f".pca-{scenario.lower()}-",
        dir=output_root,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        _write_pca_outputs(artifact, result, staging_dir)
        _publish_pca_outputs(staging_dir, output_dir)
```

Remove the old `if output_dir.exists(): raise FileExistsError(...)` block and the ten direct write calls that follow `run_pca`. Keep the existing summary prints and `return output_dir`.

- [ ] **Step 11: Run the complete PCA test module and confirm GREEN**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest scripts.cluster_preparation.test_pca_analysis -v
```

Expected: all PCA tests pass. The repeated run returns the same `output_root / "pf"` path, the simulated generation failure preserves all ten previous files byte-for-byte, and the non-directory path raises `NotADirectoryError`.

- [ ] **Step 12: Compile and inspect the focused diff**

Run:

```powershell
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m py_compile scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/test_pca_analysis.py
git diff --check
git diff -- scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/test_pca_analysis.py
```

Expected: compilation and diff checks exit `0`; the diff contains only staging, managed publication, and the behavior tests.

- [ ] **Step 13: Commit the staged publication behavior**

```powershell
git add scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/test_pca_analysis.py
git diff --cached --check
git commit -m "feat: safely overwrite managed PCA outputs"
```

---

### Task 2: Document and verify repeatable PCA publication

**Files:**

- Modify: `README.md:24-28`

**Interfaces:**

- Consumes: the managed publication behavior committed in Task 1.
- Produces: a user-facing command and explicit preservation/overwrite semantics.
- Guarantees: users know that omitting `--output-root` writes repeatedly to `output_pca/pf` or `output_pca/tf`, while custom output roots remain supported.

- [ ] **Step 1: Document the default repeated-run behavior**

Add after the confirmed v3 clustering workflow paragraph:

```markdown
PCA may be rerun against the same prepared artifact without choosing a new
output root. By default it republishes its ten managed result files in
`output_pca/pf` or `output_pca/tf`. Other files and subdirectories in those
scenario directories are preserved. Use `--output-root` only when a separate
PCA experiment directory is intentional.
```

- [ ] **Step 2: Run the full regression and compilation checks**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest discover -v
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m py_compile scripts/cluster_preparation/pca_analysis.py scripts/cluster_preparation/test_pca_analysis.py scripts/utils/clustering_preprocessing.py
git diff --check
git status --short
```

Expected: the full suite passes, compilation and diff checks exit `0`, and only `README.md` remains as the intended uncommitted change. Existing user-generated output and data changes remain untouched.

- [ ] **Step 3: Commit the documentation**

```powershell
git add README.md
git diff --cached --check
git commit -m "docs: explain repeatable PCA publication"
```

- [ ] **Step 4: Run final branch verification**

Run:

```powershell
$env:OMP_NUM_THREADS='1'
& 'D:\anaconda3\Scripts\conda.exe' run --no-capture-output -n 2026master python -m unittest discover -v
git log --oneline -4
git status --short
```

Expected: the full suite passes on the committed tree; the two newest implementation commits are `feat: safely overwrite managed PCA outputs` and `docs: explain repeatable PCA publication`; pre-existing user outputs and untracked data remain uncommitted.
