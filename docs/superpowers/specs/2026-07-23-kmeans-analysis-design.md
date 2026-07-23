# Interactive K-means Analysis Design

## Goal

Add a directly runnable Python workflow for one-scenario-at-a-time K-means
analysis of prepared street-segment data. The workflow must cluster PF and TF
separately, use the already standardized input values without any further
scaling, produce reproducible tabular and visual outputs, and retain every
original input row and field in the main result.

This is an unsupervised typology experiment. It is not a supervised-learning
training workflow.

## Confirmed Decisions

- The K-means feature set is the existing scenario-specific seven-variable
  list used by the PCA and K-selection workflows.
- PF, TF, and future feature-list changes use one shared configuration so the
  PCA, K-selection, and final K-means scripts cannot drift apart.
- The new workflow lives in `scripts/cluster_kmeans/`.
- A successful run uses this readable final-directory pattern:

  ```text
  {scenario}_k{k}_v{n_features}_kmeanspp_n50_mi500_rs42
  ```

  For example:

  ```text
  tf_k4_v7_kmeanspp_n50_mi500_rs42
  ```

- Final directory names contain no timestamp, feature hash, or run number.
- If the final directory already exists, the workflow stops before fitting
  K-means and tells the user to move or rename the existing result. It never
  overwrites that directory.
- `algorithm="lloyd"` and other complete parameters remain in
  `run_config.json` and the text report rather than the directory name.

## Project Structure

The implementation will add:

```text
scripts/
├── cluster_kmeans/
│   ├── __init__.py
│   ├── kmeans_analysis.py
│   └── test_kmeans_analysis.py
└── utils/
    └── scenario_features.py
```

`scripts/utils/scenario_features.py` will become the single source of truth for:

- the ordered PF feature list;
- the ordered TF feature list;
- scenario metadata/identifier hints needed by the existing PCA and K-selection
  workflows.

`scripts/cluster_preparation/pca_analysis.py` and
`scripts/cluster_preparation/choose_k.py` will import these definitions instead
of maintaining separate copies. Their analytical behavior will otherwise
remain unchanged.

`scripts/cluster_kmeans/kmeans_analysis.py` will contain the complete
interactive workflow. It will remain importable for tests and directly
executable from a VSCode terminal.

## User Interaction

The script prompts in this order.

### 1. Scenario

```text
Select flood scenario:
1. PF
2. TF
```

Accepted values are `1`, `pf`, `2`, and `tf`, ignoring case and surrounding
whitespace. Invalid values print a concise explanation and prompt again.

Internally the scenario is normalized to uppercase for reporting and lowercase
for directory naming.

### 2. K

```text
Enter the number of clusters K:
```

The first validation accepts only an integer greater than or equal to 2. Boolean
values are not relevant to terminal input, and decimal strings such as `2.0`
are rejected.

Because the effective sample count is unknown until the CSV has been loaded and
validated, the script performs the upper-bound check later. If
`K >= n_valid_samples`, it reports the effective sample count and prompts for K
again. One run fits exactly one accepted K.

### 3. CSV path

PF displays:

```text
Enter CSV path [default: pf_for_PCA.csv]:
```

TF displays:

```text
Enter CSV path [default: tf_for_PCA.csv]:
```

Pressing Enter selects the current project defaults:

- `outputs_correlation/pf/pf_for_PCA.csv`
- `outputs_correlation/tf/tf_for_PCA.csv`

The prompt deliberately shows the short filename requested by the research
workflow while the configured default resolves to the existing project file.
An explicitly entered relative path resolves from the terminal's current
working directory; an absolute path remains absolute. Matching single or
double quotes surrounding a pasted path are removed. `pathlib.Path` is used
throughout.

Missing, non-file, or inaccessible paths produce a clear message and prompt
again. CSV parser, decoding, and permission errors are presented without a raw
traceback during normal interactive use.

## Shared Feature Configuration

The initial shared lists are the existing PCA/K-selection variables.

PF:

1. `Per_extent`
2. `N_PFlossR_2kiw`
3. `N_PFlossR_2kmw`
4. `PFResident_lossR`
5. `PFDaynight_lossR`
6. `PFAB5k_NOR`
7. `PFABC_NOR`

TF:

1. `Tem_extent`
2. `N_TFlossR_2kiw`
3. `N_TFlossR_2kmw`
4. `TFResident_lossR`
5. `TFDaynight_lossR`
6. `TFAB5k_NOR`
7. `TFABC_NOR`

The lists are ordered because that order controls the model matrix, exported
centroid columns, profile plots, and feature-list reporting. `n_features` is
calculated from the final usable configured features; it is never hard-coded
to seven.

All input columns remain in `clustered_data.csv`. Being excluded from the model
does not remove a column from the retained source data.

## Column Classification and Validation

Only configured scenario features can enter K-means. Every other input column
is classified as retained-but-excluded and receives a reason for terminal,
JSON, and report output.

Identifier recognition is case-insensitive and uses complete names or explicit
ID patterns, including:

- `id`, `fid`, `objectid`, `ogc_fid`, `gid`, `MS_ID`;
- `street_id`, `segment_id`;
- names ending in `_id` or beginning with `id_`;
- geometry identifier fields.

The implementation must not exclude a normal metric merely because its name
contains the letters `id`.

Other exclusion classifications cover:

- non-numeric fields;
- names containing `index`, `risk`, `cluster`, `label`, `type`, `typology`,
  `intervention`, `score`, or `geometry`, matched case-insensitively;
- existing composite-index, result-label, typology, scenario-selection, and
  explanatory fields;
- numeric fields not present in the shared scenario feature list.

For configured model features:

- a missing column is a fatal schema error that names the missing feature;
- a non-numeric dtype is a fatal schema error that names the feature;
- an all-empty column is excluded and recorded;
- a column with only one unique finite non-missing value is excluded and
  recorded;
- a column containing positive or negative infinity is excluded and recorded;
- finite missing values are not imputed.

After unusable columns are excluded, missing counts are printed for every
remaining feature with missing data and recorded in
`data_quality_issues.csv`, `run_config.json`, and `run_report.txt`. A row is
valid only when all remaining model features are finite and non-missing. The
workflow drops invalid rows from the model matrix, reports the number dropped,
and leaves their `kmeans_cluster` value empty in the retained full dataset.

The final model matrix is checked again for numeric dtype, finite values,
constant columns, at least one feature, and enough observations for the chosen
K. If validation leaves fewer than three effective observations, the workflow
cannot satisfy `2 <= K < n_valid_samples` and stops with a clear error.

If the input already contains a `kmeans_cluster` column, the workflow stops
with a clear reserved-column error rather than silently overwriting original
data.

## No Additional Scaling

The model matrix is created with a direct numeric conversion of the selected
CSV columns. The K-means script must not import or call `StandardScaler`,
`MinMaxScaler`, normalization helpers, or any other scale transformation.

PCA centers values internally as part of its mathematical definition, but PCA
is run only after K-means for visualization and label ordering. PCA output is
never supplied to K-means.

Before fitting, the terminal summary explicitly prints:

- scenario;
- resolved CSV path;
- original sample count;
- valid sample count;
- final feature count;
- the ordered feature names;
- all excluded columns and reasons;
- K;
- a statement that the data will not be standardized again.

## K-means Configuration and Compatibility

User-editable constants at the top of `kmeans_analysis.py` define:

```python
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
KMEANS_RANDOM_STATE = 42
KMEANS_ALGORITHM = "lloyd"
KMEANS_TOL = 1e-4
OUTPUT_DPI = 300
```

The configured model is:

```python
KMeans(
    n_clusters=k,
    init=KMEANS_INIT,
    n_init=KMEANS_N_INIT,
    max_iter=KMEANS_MAX_ITER,
    random_state=KMEANS_RANDOM_STATE,
    algorithm=KMEANS_ALGORITHM,
    tol=KMEANS_TOL,
)
```

The script checks that the installed scikit-learn `KMeans` interface supports
the configured arguments. An incompatible version raises a clear error naming
the unsupported setting and installed version. Core settings are never
silently removed or changed.

The workflow records its start time, end time, elapsed wall-clock seconds,
`model.n_iter_`, inertia, and convergence status. Because scikit-learn does not
expose a definitive boolean convergence flag, `n_iter_ < max_iter` is recorded
as converged; reaching `max_iter` is conservatively recorded as not confirmed
and produces a warning. Captured `ConvergenceWarning` messages are also
retained.

## Stable Cluster Labels

Scikit-learn's original labels are implementation labels with no research
meaning. After fitting:

1. Fit `PCA(n_components=2)` on the same full-dimensional valid model matrix,
   without additional scaling.
2. Project every K-means centroid through that fitted PCA.
3. Sort centroids by PC1 projection from low to high.
4. Map the first centroid to Cluster 1 and continue through Cluster K.

If two ordering values are equal, the original scikit-learn label is the
deterministic tie-breaker.

If two-component PCA is impossible, the workflow warns and falls back to
sorting by each centroid's mean standardized feature value. The chosen rule is
recorded in `run_config.json` and the report.

Reordering changes only displayed labels. It does not change centroid values,
distances, or membership. The complete mapping is exported as
`cluster_label_mapping.csv` with:

- `sklearn_original_label`;
- `reordered_cluster_id`;
- `ordering_value`.

## Output Directory Safety

All final results live under the project-level `output_kmeans/` directory.
After feature validation, the final path is built from lowercase scenario, K,
final feature count, and the readable fixed parameter tags:

```text
output_kmeans/tf_k4_v7_kmeanspp_n50_mi500_rs42
```

The script checks for this final path before fitting. If it exists, execution
stops and explains that the existing result must be moved or renamed.

Exports are first written to a temporary staging directory under
`output_kmeans/`. Only after every required artifact has been written does the
script rename the staging directory to the final readable path. A failed
export removes the staging directory and reports the failed artifact. The
final path is checked again immediately before the rename to prevent an
accidental overwrite.

## Required Tabular and Text Artifacts

All CSV files use UTF-8 encoding and no DataFrame index unless explicitly
represented as a data column. JSON and text files use UTF-8. JSON uses
indentation and `ensure_ascii=False`.

### `clustered_data.csv`

- Preserves all source columns and original row order.
- Adds nullable integer `kmeans_cluster`, numbered from 1.
- Leaves `kmeans_cluster` empty for rows excluded because of missing values.

### `cluster_summary.csv`

Contains one row for each reordered cluster ID, including:

- `cluster_id`;
- `sample_count`;
- `sample_percentage`, expressed from 0 to 100 and using valid samples as the
  denominator;
- `distance_to_centroid_mean`;
- `distance_to_centroid_median`;
- `distance_to_centroid_std`;
- `distance_to_centroid_max`.

Distance standard deviation uses `ddof=0` because the records describe the
complete assigned cluster rather than a sample drawn from it.

### `cluster_centroids_standardized.csv`

Contains `cluster_id` followed by each configured feature's centroid in the
input standardized space. Rows use reordered cluster IDs.

### `cluster_feature_profile.csv`

Contains one row per cluster-feature pair:

- `cluster_id`;
- `feature`;
- `mean`;
- `median`;
- `std`;
- `min`;
- `max`;
- `q25`;
- `q75`.

Profile standard deviation also uses `ddof=0`. Positive, negative, and
near-zero values retain their interpretation relative to the already
standardized overall input space.

### `sample_distances.csv`

Contains valid model samples only, in original row order:

- every recognized available ID field;
- `kmeans_cluster`;
- `distance_to_assigned_centroid`.

Distances are Euclidean distances in the same full standardized feature space
used by K-means, not distances in the PCA plot.

### `model_metrics.csv`

Contains one experiment row with:

- `scenario`;
- `input_csv`;
- `n_original_samples`;
- `n_valid_samples`;
- `n_features`;
- `K`;
- `inertia`;
- `silhouette_score`;
- `calinski_harabasz_score`;
- `davies_bouldin_score`;
- `n_iter`;
- `converged`;
- `init`;
- `n_init`;
- `max_iter`;
- `random_state`;
- `runtime_seconds`;
- `timestamp`.

Silhouette is calculated on the full valid model matrix, not a sample. Each
metric is guarded independently. If label/sample conditions make a metric
undefined, the value is empty and a warning is saved without aborting the run.

### `run_config.json`

Records:

- scenario;
- resolved absolute input and output paths;
- K;
- ordered final feature list;
- excluded-column entries and reasons;
- an explicit `standard_scaler_applied: false`;
- complete K-means parameters, including algorithm and tolerance;
- original, valid, and removed sample counts;
- per-feature missing counts and the drop-row policy;
- cluster-label ordering method;
- PC1 and PC2 explained variance when available;
- Python, pandas, NumPy, and scikit-learn versions;
- start, end, and execution timestamps;
- captured warnings.

### `run_report.txt`

Provides the requested human-readable experiment narrative:

- scenario and input;
- K and ordered features;
- original, valid, and removed sample counts;
- excluded variables and data-quality actions;
- K-means parameters;
- cluster counts and percentages;
- inertia and all three validation metrics;
- iteration and convergence information;
- PC1, PC2, and PC1+PC2 explained variance when available;
- runtime;
- every output filename;
- the label-ordering rule;
- an explicit statement that the input was already standardized and the script
  did not standardize it again.

### `data_quality_issues.csv`

Provides the durable problem record requested for residual data issues. It
contains:

- `issue_type`;
- `column`;
- `count`;
- `action`;
- `details`.

It is still created with headers when no issues are found.

## Visualizations

Matplotlib uses a non-interactive backend and saves every plot at 300 dpi.
Seaborn is not used. Figure objects are explicitly closed.

### `pca_cluster_scatter.png`

- Uses PCA only after full-space K-means.
- Colors points by reordered `kmeans_cluster`.
- Labels axes with PC1 and PC2 explained-variance percentages.
- Shows `Cluster 1` through `Cluster K` in the legend.
- Includes scenario, K, and feature count in the title.
- Includes a figure note that PCA is a visualization projection, not the
  clustering space.

If two-dimensional PCA is unavailable, the script creates a clearly labeled
placeholder figure and records the reason rather than crashing or silently
omitting the required filename.

### `cluster_size_bar.png`

Shows every cluster's valid-sample count and percentage, with both values
annotated.

### `cluster_centroid_heatmap.png`

- Uses `matplotlib.imshow`.
- Places clusters on rows and features on columns.
- Uses a diverging scale symmetric around zero and displays a colorbar.
- Writes formatted centroid values in every cell, with contrasting annotation
  color where needed.
- Expands the figure and rotates labels as feature count or label length grows.

### `cluster_feature_profiles.png`

- Plots one standardized mean line per cluster.
- Uses features on the x-axis.
- Adds a horizontal `y=0` reference.
- Uses readable rotated labels and a cluster legend.

A radar chart is outside the required scope and will not be added initially.

## Terminal Progress and Completion

After the three prompts, progress messages follow the requested sequence:

```text
[1/7] Reading input data...
[2/7] Selecting clustering features...
[3/7] Validating data...
[4/7] Running K-means...
[5/7] Calculating metrics and profiles...
[6/7] Generating visualizations...
[7/7] Exporting results...
```

On success, the terminal prints:

- `K-means analysis completed.`;
- scenario;
- K;
- feature count;
- valid sample count;
- final output folder;
- each cluster's count and percentage;
- silhouette score or `not available`;
- inertia;
- total runtime.

## Error Handling

Expected failures use specific exception handling and user-facing messages:

- `FileNotFoundError`, non-file paths, and `PermissionError` during path input;
- pandas empty-data and parser errors;
- decoding and filesystem errors while reading;
- missing configured features;
- non-numeric configured features;
- no usable model features;
- NaN and infinity validation outcomes;
- K below 2 or not below valid sample count;
- too few effective samples;
- unsupported K-means parameters in the installed scikit-learn version;
- PCA component limitations;
- undefined metrics;
- output permission, disk, and rename failures.

The top-level interactive boundary may convert known workflow exceptions into
concise `ERROR:` messages and a non-zero process exit. It will not use a broad
`except Exception` to suppress unknown programming errors.

## Modular Interfaces

The implementation will provide at least:

- `prompt_scenario()`
- `prompt_k()`
- `prompt_csv_path()`
- `load_dataset()`
- `select_features()`
- `validate_features()`
- `build_output_directory()`
- `run_kmeans()`
- `reorder_cluster_labels()`
- `calculate_metrics()`
- `calculate_cluster_profiles()`
- `generate_pca_plot()`
- `generate_cluster_size_plot()`
- `generate_centroid_heatmap()`
- `generate_profile_plot()`
- `export_results()`
- `main()`

Small dataclasses may carry validated input, fitted-model, metrics, and
artifacts between these functions so that `main()` remains orchestration rather
than a container for analytical logic.

The script ends with:

```python
if __name__ == "__main__":
    main()
```

## Testing Strategy

Tests use `unittest`, NumPy/Pandas synthetic data, `tempfile`, and focused
patching only at interaction or filesystem boundaries. Production analytical
functions are exercised with real scikit-learn objects.

The suite covers:

- scenario aliases, normalization, and retry;
- integer K validation and the post-validation upper bound;
- quoted absolute and relative paths, defaults, missing files, and retries;
- PF/TF feature ordering from the shared configuration;
- synchronization of PCA, K-selection, and K-means feature imports;
- exact ID recognition without substring false positives;
- excluded-column reasons;
- all-empty, constant, infinite, and missing-value behavior;
- retained full source rows and nullable labels for invalid rows;
- absence of any additional scaling;
- K-means parameter values;
- PC1 centroid ordering, deterministic ties, and mean-centroid fallback;
- unchanged membership after remapping;
- centroid distances, summaries, profiles, and metrics;
- metric-unavailable warnings;
- PCA-unavailable placeholder output;
- readable directory naming and existing-directory refusal;
- all required UTF-8 artifacts and JSON/report content;
- an end-to-end run on a small prepared synthetic dataset.

Verification will run focused red-green tests during implementation, the full
new test module, affected existing PCA and K-selection tests, the repository's
complete unittest discovery, Python compilation, and a temporary end-to-end
smoke run.

## Non-goals

- Searching or recommending multiple K values.
- Standardizing, normalizing, or imputing model values.
- Feeding PCA scores into K-means.
- Combining PF and TF in one model.
- Assigning semantic high-risk typology names automatically.
- Adding seaborn or a required radar chart.
- Overwriting, versioning, timestamping, or hashing final output directories.
