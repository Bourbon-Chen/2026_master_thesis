# PCA Managed Output Overwrite Design

## Goal

Allow repeated PCA runs to publish results to `output_pca/pf` or
`output_pca/tf` without raising `FileExistsError`, while preserving files and
subdirectories that the PCA script does not manage.

## Current Behavior and Root Cause

`scripts/cluster_preparation/pca_analysis.py::run_analysis` resolves the final
directory as `Path(output_root) / scenario.lower()`. It then raises
`FileExistsError` whenever that directory already exists, before running PCA or
writing any result.

The default output root is already the repository-level `output_pca` directory.
Therefore, a run without `--output-root` targets `output_pca/pf` or
`output_pca/tf`. A caller may continue using `--output-root` when a separate
experiment directory is intentionally required.

## Approved Publishing Strategy

Each run generates a complete PCA result in a temporary staging directory. Only
after every table, figure, report, and provenance file has been generated
successfully does the script publish the result to the final scenario
directory.

Publishing replaces only the PCA-managed filenames. It never clears or
recursively replaces the final scenario directory. Consequently, manually
created files and unrelated subdirectories remain untouched.

The ten PCA-managed filenames are:

1. `pca_summary.csv`
2. `pca_loadings.csv`
3. `pca_scores.csv`
4. `explained_variance.png`
5. `cumulative_variance.png`
6. `pca_scatter.png`
7. `loading_plot.png`
8. `correlation_circle.png`
9. `pca_report.txt`
10. `preprocessing_reference.json`

## Data Flow

1. Load and verify the prepared clustering artifact.
2. Run PCA against the artifact's persisted standardized matrix.
3. Create a temporary directory outside the final scenario directory.
4. Generate all ten managed outputs in the temporary directory.
5. Verify that all ten expected files exist as regular files.
6. Create the resolved `output_pca/pf` or `output_pca/tf` directory if it does
   not exist.
7. Replace each corresponding managed destination file with its staged file.
8. Remove the temporary directory automatically.
9. Return and print the final scenario directory.

PF and TF publication remain isolated because each run resolves exactly one
scenario directory.

## Failure and Preservation Semantics

- If loading, PCA computation, plotting, reporting, or staging fails, publishing
  does not begin and the existing final outputs remain unchanged.
- If the final path exists as a non-directory, the run raises a clear
  filesystem error rather than overwriting that path.
- Existing managed files may be replaced.
- Existing unmanaged files and subdirectories are preserved.
- The script does not delete the final scenario directory.
- No overwrite flag is required for normal repeated runs.

Publishing ten files cannot be a single filesystem-wide atomic operation. A
rare failure during the final replacement loop could leave a partially
published managed set. Staging nevertheless prevents the common and more
important failure mode in which PCA generation itself mixes incomplete new
outputs with old outputs.

## Command-Line Contract

The existing CLI remains compatible:

```powershell
python scripts/cluster_preparation/pca_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
```

This writes to the default `output_pca/pf` directory for this PF example and
may replace previous PCA-managed outputs there. A TF artifact similarly writes
to `output_pca/tf`.

The optional `--output-root` argument remains available:

```powershell
python scripts/cluster_preparation/pca_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF --output-root output_pca_run2
```

This intentionally publishes the PF example to `output_pca_run2/pf` using the
same managed-file replacement policy. TF resolves to `output_pca_run2/tf`.

## Testing

Automated tests will verify:

- a first run creates the scenario directory and all managed outputs;
- a repeated run replaces stale managed files without `FileExistsError`;
- an unrelated sentinel file and unrelated subdirectory survive publication;
- a generation failure leaves previously published managed outputs unchanged;
- PCA still consumes the exact standardized prepared matrix without fitting a
  new scaler;
- the default output root remains the singular repository-level `output_pca`;
- the full project regression suite continues to pass in the `2026master`
  environment.

## Scope

This change applies only to PCA output publication. It does not alter prepared
artifacts, feature selection, structural-null handling, standardization,
choose-K, K-means, HDBSCAN, or their output immutability policies.
