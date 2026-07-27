# Unified Clustering Preprocessing Design

## Goal

Create one reproducible preprocessing path for PF and TF clustering
experiments so that PCA, K selection, K-means, and HDBSCAN:

- use the same scenario-specific feature-selection rules;
- interpret structural nulls consistently as semantic zero;
- retain every selected high-risk street object;
- use the same `StandardScaler` transformation for a given scenario dataset;
- preserve the original `[-1, 1]` indicator values for planning
  interpretation and audit;
- record enough provenance to prove which source data, features, null policy,
  and scaler produced an experiment.

This remains an unsupervised street-typology workflow. PCA is a diagnostic
branch; it does not implicitly produce the input for K-means.

## Motivation

The current workflows have drifted:

- PCA fills nulls with zero and standardizes an in-memory copy.
- K selection fills nulls with zero and standardizes independently.
- K-means removes every row with a null in any active feature and does not
  standardize.
- The modular HDBSCAN pipeline fills nulls with zero, while the legacy
  `run_hdbscan_street_typology.py` uses median imputation.
- `scripts/utils/scenario_features.py` contains manually edited PF/TF feature
  lists. Those lists no longer represent all eligible scenario variables.
- PCA output is not passed to K-means, even though the existing filenames and
  workflow can make that relationship appear implicit.

On the current v3 high-risk inputs, complete-case K-means retains only 2,204
of 4,100 PF rows and 5,182 of 13,584 TF rows. Structural nulls therefore
exclude 46.24% of PF objects and 61.85% of TF objects even though those nulls
have a defined planning meaning.

## Confirmed Domain Decisions

1. A null in a clustering indicator is structural, not an unknown observation.
   It occurs when the indicator is not applicable or cannot be calculated,
   commonly because the baseline denominator or baseline facility count is
   zero.
2. For this experiment, a structural null has the same modelling meaning as
   zero: no measurable change.
3. Structural nulls are converted to `0.0` before fitting `StandardScaler`.
4. The raw semantic dataset keeps its nulls for traceability. Null replacement
   occurs in the modelling-preparation layer, not by overwriting the cleaned
   source CSV.
5. All primary PCA, K-selection, K-means, and HDBSCAN analyses use
   `StandardScaler`.
6. A scaler is fitted separately for each final PF or TF analysis dataset,
   after high-risk row filtering and final feature selection.
7. K selection and final K-means consume the exact same standardized matrix.
8. Values must be finite and within their semantic range before
   standardization. Standardized z-scores are not required to remain in
   `[-1, 1]`.
9. PF/TF feature discovery is dynamic. The workflow does not depend on a
   manually maintained complete feature tuple.

## Non-Goals

- Do not overwrite the raw source CSV or the semantic cleaned CSV.
- Do not encode structural null as a separate clustering feature.
- Do not use mean, median, KNN, or iterative imputation.
- Do not cluster composite `Index` or `Risk` columns together with their
  component indicators.
- Do not silently coerce malformed strings, infinity, or out-of-range values
  to zero.
- Do not make PCA scores the K-means input unless a future experiment
  explicitly chooses PCA-score clustering.
- Do not redesign the mathematical definitions of the HEV indicators or risk
  indexes.

## Two-Layer Architecture

### Layer 1: Semantic cleaning

`scripts/clean_mt_update_ms_hev.py` remains responsible for:

- selecting the requested semantic columns;
- producing the cleaned and normalized-cleaned CSVs;
- preserving signed values;
- capping finite `_lossR` values above 1 where the current domain rule
  requires it;
- retaining structural nulls;
- auditing the finite indicator ranges before modelling.

The normalized-cleaned file remains human-interpretable. Its finite indicator
values retain their original `[-1, 1]` meaning.

Layer 1 must not fit or apply `StandardScaler`.

### Layer 2: Modelling preparation

Add a shared importable module:

```text
scripts/utils/clustering_preprocessing.py
```

Add a command-line entry point:

```text
scripts/cluster_preparation/prepare_clustering_inputs.py
```

The module owns all reusable selection, validation, structural-zero,
standardization, audit, and serialization logic. The command-line script
materializes one immutable prepared dataset per scenario so that downstream
analyses consume the same matrix rather than repeating preprocessing.

### Existing utility migration

`scripts/utils/clustering_preprocessing.py` becomes the canonical
preprocessing implementation.

Update `scripts/utils/scenario_features.py` as follows:

- remove the manually maintained `PF_FEATURES` and `TF_FEATURES` tuples;
- replace them with PF/TF marker definitions and metadata-role definitions;
- replace defaults pointing to `pf_for_PCA.csv` and `tf_for_PCA.csv` with
  helpers that resolve prepared scenario directories;
- update tests so they verify dynamic discovery rather than Python object
  identity between hard-coded tuples.

Move the reusable preparation behavior currently in
`scripts/utils/feature_utils.py` into the canonical module. Convert
`feature_utils.py` into a temporary compatibility re-export, update all active
imports to the canonical module, and leave no independent selection,
fill-zero, or range-validation implementation in the compatibility file.

## End-to-End Data Flow

```text
MT_UPDATE_MS_HEV_vX.csv
    |
    v
clean_mt_update_ms_hev.py
    |
    v
MT_UPDATE_MS_HEV_vX_NORcleaned.csv
    |
    +--> filter_high_risk_dataset.py --scenario pf --threshold 0.25
    |       |
    |       v
    |    PF high-risk semantic CSV
    |
    +--> filter_high_risk_dataset.py --scenario tf --threshold 0.25
            |
            v
         TF high-risk semantic CSV

PF/TF high-risk semantic CSV
    |
    v
dynamic candidate discovery + structural null -> 0
    |
    v
correlation audit and explicit exclusion decisions
    |
    v
prepare_clustering_inputs.py
    |
    +--> filled original-scale model matrix
    +--> standardized model matrix
    +--> scaler parameters
    +--> feature/null/range audit
    +--> provenance manifest and hashes
    |
    +--> pca_analysis.py
    +--> choose_k.py
    +--> kmeans_analysis.py
    +--> primary HDBSCAN workflow
```

## Stage 1: Semantic Cleaning

The user supplies the HEV source CSV explicitly or accepts a documented
default. Layer 1 produces:

```text
data/MT_UPDATE_MS_HEV_vX_cleaned.csv
data/MT_UPDATE_MS_HEV_vX_NORcleaned.csv
```

The normalized-cleaned file is the input to risk filtering. It must retain:

- `fid`;
- `MS_ID`;
- `Per_extent`;
- `Tem_extent`;
- selected normalized, signed-loss, and index columns;
- structural nulls.

The cleaning report records row count, retained columns, numeric ranges, and
the number of capped `_lossR` values. It does not claim that the data are
machine-learning standardized.

## Stage 2: High-Risk Row Filtering

Run `scripts/filter_high_risk_dataset.py` twice against the same
normalized-cleaned file.

PF selection:

```text
PF_Index_Risk_equal > 0.25
```

TF selection:

```text
TF_Index_Risk_equal > 0.25
```

The outputs retain every source column so that feature discovery and final
label joins have full context:

```text
data/MT_UPDATE_MS_HEV_vX_NORcleaned_pf_risk_gt_0_25.csv
data/MT_UPDATE_MS_HEV_vX_NORcleaned_tf_risk_gt_0_25.csv
```

The threshold and comparison operator are recorded in downstream provenance.
For the current v3 data, the expected counts are 4,100 PF rows and 13,584 TF
rows. Those counts are regression references, not hard-coded requirements for
future datasets.

## Stage 3: Dynamic Scenario Feature Discovery

### Scenario classification

Column classification is case-insensitive and anchored at the beginning of
the column name.

PF markers:

```text
^PF
^N_PF
^Per_
```

TF markers:

```text
^TF
^N_TF
^Tem_
```

Examples:

| Column | Classification |
| --- | --- |
| `Per_extent` | PF |
| `N_PFlossR_2kiw` | PF |
| `PFResident_lossR` | PF |
| `PFAB5k_NOR` | PF |
| `Tem_extent` | TF |
| `N_TFlossR_2kiw` | TF |
| `TFDaynight_lossR` | TF |
| `TFABC_NOR` | TF |

Anchored markers avoid treating an unrelated name containing the letters
`pf`, `tf`, `per`, or `tem` as a scenario feature.

### Field roles

Every column is assigned one role:

- `PF_FEATURE`;
- `TF_FEATURE`;
- `SHARED_FEATURE`;
- `IDENTIFIER`;
- `COMPOSITE_INDEX`;
- `RESULT_OR_LABEL`;
- `NON_NUMERIC`;
- `EXCLUDED`.

Identifiers include `fid`, `MS_ID`, and the existing explicit ID patterns.
Names containing `Index` or `Risk` are metadata or validation fields and do
not enter the clustering matrix. They remain available for post-cluster
profiles and joins.

Columns such as `Street_type_NOR`, `Resident_NOR`, `DayNight_NOR`, and
`PRM_NOR` are classified as shared rather than silently assigned to both
scenarios. The primary experiment includes only dynamically identified
scenario-specific features. Shared features are excluded and recorded; adding
them is outside this implementation.

### Stable feature order

Selected features retain source-column order. The final order is written to
`selected_features.csv` and the preprocessing manifest. Every downstream
consumer reads that order from the manifest; no consumer re-discovers or
sorts the features.

## Stage 4: Correlation Review

Correlation review uses the same semantic null policy as modelling:

```text
candidate scenario features -> structural nulls filled with 0 -> Pearson
```

Pearson correlation does not require StandardScaler because positive affine
standardization does not change correlation coefficients.

Refactor `scripts/pearson_correlation/correlation_analysis.py` so it consumes
the shared feature-discovery and structural-zero functions instead of fixed
feature groups.

Correlation outputs include:

```text
feature_summary.csv
data_quality_warnings.csv
pearson_correlation_matrix.csv
all_correlation_pairs.csv
high_correlation_pairs.csv
high_correlation_pairs_within_group.csv
high_correlation_pairs_cross_group.csv
```

Feature removal remains an explicit research decision. The decision is stored
as a machine-readable exclusion artifact, for example:

```text
excluded_features.json
```

An empty exclusion list means that all eligible scenario features are kept.
The shared preprocessor validates that every requested exclusion exists and
belongs to the active scenario.

The existing `extract_high_columns.py` manual default-removal lists must not
remain a hidden source of truth. Remove it from all active default paths and
workflow documentation. Keep the command only as a legacy compatibility tool,
make it print a deprecation warning, and require an explicit input path. No
PCA, K-selection, K-means, or HDBSCAN entry point may default to its
`pf_for_PCA.csv` or `tf_for_PCA.csv` outputs.

## Stage 5: Final Modelling Preparation

### Public data structure

The shared module returns a structured result similar to:

```python
@dataclass
class PreparedClusteringData:
    scenario: str
    source_data: pd.DataFrame
    metadata: pd.DataFrame
    original_features: pd.DataFrame
    filled_features: pd.DataFrame
    standardized_features: pd.DataFrame
    feature_names: tuple[str, ...]
    missing_counts: pd.Series
    rows_with_structural_null: pd.Series
    excluded_columns: pd.DataFrame
    audit: pd.DataFrame
    scaler: StandardScaler
```

The implementation uses this `PreparedClusteringData` boundary. Internal
helper names may differ, but downstream consumers receive this single
structured result rather than separate ad hoc DataFrames.

### Validation order

For the active scenario:

1. Require a non-empty input table.
2. Require identifier and scenario-risk metadata needed by the workflow.
3. Discover eligible scenario features dynamically.
4. Apply the explicit correlation exclusion artifact.
5. Preserve source row order and a stable source-row position.
6. Require selected features to be numeric.
7. Reject positive or negative infinity.
8. Reject finite values outside the configured semantic range.
9. Record per-feature null counts and per-row structural-null counts.
10. Copy selected features and replace structural nulls with `0.0`.
11. Reject any remaining non-finite value.
12. Remove features that are constant after structural-zero replacement and
    record the reason.
13. Fail if no usable feature remains.
14. Fit one `StandardScaler` on all retained rows and final features.
15. Verify every standardized value is finite.
16. Verify each retained standardized feature has mean approximately zero and
    population standard deviation (`ddof=0`) approximately one, using
    `numpy.allclose(..., atol=1e-10, rtol=1e-10)`.
17. Verify output row count and row order equal the input high-risk table.

Malformed strings are errors; they are not coerced to zero. Only actual null
values in eligible modelling features receive the semantic-zero treatment.

### StandardScaler scope

Fit separate scalers:

- one on the final PF high-risk rows and PF feature set;
- one on the final TF high-risk rows and TF feature set.

Do not fit a scaler on the 116,959-row full dataset and reuse it for the
high-risk subsets. That would not give the selected PF/TF samples unit
variance.

The scaler uses scikit-learn defaults:

```text
with_mean=True
with_std=True
```

Constant features are removed before fitting. Standardized values may be less
than -1 or greater than 1; that is expected.

## Prepared Artifact Layout

For each source dataset and scenario, materialize:

```text
outputs_preprocessing/
└── {dataset_name}/
    ├── PF/
    │   ├── metadata.csv
    │   ├── features_original.csv
    │   ├── features_filled.csv
    │   ├── features_standardized.csv
    │   ├── selected_features.csv
    │   ├── excluded_features.csv
    │   ├── scaler_parameters.csv
    │   ├── preprocessing_audit.csv
    │   └── preprocessing_config.json
    └── TF/
        └── same files
```

`metadata.csv` contains a stable source-row position, identifiers, and the
active risk index. Feature files preserve exactly the same row order.

`scaler_parameters.csv` records, for each feature:

- fitted mean;
- fitted variance;
- fitted scale;
- original null count;
- original minimum and maximum;
- filled minimum and maximum.

`preprocessing_config.json` records:

- source path and SHA-256;
- source row count;
- scenario and risk-filter definition;
- feature-classification rules;
- explicit exclusion artifact and hash;
- selected feature order and hash;
- structural-null strategy and fill value;
- scaler class, parameters, and fitted statistics;
- output matrix SHA-256;
- package and Python versions;
- creation timestamp.

Prepared artifacts are immutable. If the scenario output directory already
exists, preparation stops before writing and tells the user to select a new
output root or move the existing directory. This design provides no overwrite
mode.

## Stage 6: PCA

Refactor `scripts/cluster_preparation/pca_analysis.py` to consume the prepared
manifest and `features_standardized.csv`.

PCA no longer:

- selects features;
- fills nulls;
- fits its own StandardScaler.

It validates the prepared matrix hash and feature order, then produces PCA
scores, loadings, explained-variance tables, and plots.

PCA is diagnostic. K-means does not consume `pca_scores.csv` in this design.
Clustering on principal-component scores would be a separate, explicitly
configured experiment.

## Stage 7: K Selection

Refactor `scripts/cluster_preparation/choose_k.py` to consume the same prepared
manifest and standardized matrix.

It no longer:

- selects features;
- fills nulls;
- fits StandardScaler.

It evaluates the configured K range using the exact geometry that final
K-means will use and writes:

```text
k_evaluation.csv
elbow_plot.png
silhouette_score.png
calinski_harabasz.png
davies_bouldin.png
recommended_k.txt
```

The recommendation reports metric-specific best values and cluster-size
diagnostics. Final K selection remains a documented research decision rather
than an automatic single-metric choice.

## Stage 8: Final K-means

Refactor `scripts/cluster_kmeans/kmeans_analysis.py` to accept:

- the prepared scenario directory or preprocessing manifest;
- the selected K;
- the output root.

K-means no longer:

- selects features from the source CSV;
- drops rows containing nulls;
- fills nulls;
- fits StandardScaler.

Before fitting, it verifies:

- prepared scenario;
- row count;
- ordered feature names;
- standardized matrix hash;
- finite values;
- `2 <= K < n_samples`.

It fits the existing reproducible K-means configuration against the prepared
standardized matrix, calculates labels and distances, and joins labels back to
metadata by stable row position. Every prepared row receives a cluster label.

## K-means Outputs

The result directory retains the existing reproducible experiment naming
principle and includes:

```text
clustered_data.csv
cluster_summary.csv
cluster_centroids_standardized.csv
cluster_centroids_original_scale.csv
cluster_feature_profile.csv
sample_distances.csv
model_metrics.csv
cluster_label_mapping.csv
preprocessing_audit.csv
run_config.json
run_report.txt
pca_cluster_scatter.png
cluster_size_bar.png
cluster_centroid_heatmap.png
cluster_feature_profiles.png
```

`clustered_data.csv` retains identifiers, relevant original semantic columns,
the risk index, `kmeans_cluster`, `had_structural_null`, and
`structural_null_count`.

`cluster_centroids_standardized.csv` supports relative interpretation:

- positive means above the scenario-sample mean;
- negative means below the scenario-sample mean;
- zero means near the scenario-sample mean.

`cluster_centroids_original_scale.csv` uses
`StandardScaler.inverse_transform()` to return centroids to the filled
semantic scale.

Cluster profiles report, per feature and cluster:

- original non-null count;
- original structural-null count and percentage;
- mean and median among original non-null values;
- mean after semantic-zero replacement;
- standardized centroid;
- inverse-transformed centroid.

This prevents imputed zeros and standardized units from obscuring the planning
meaning.

## HDBSCAN Integration

The modular HDBSCAN runner consumes the same standardized prepared artifact for
its primary experiment. It no longer performs independent feature discovery,
null replacement, or scaling.

The existing raw-scale HDBSCAN Version A may remain as an explicitly labelled
sensitivity experiment. It is not the primary result and must consume
`features_filled.csv`, not recreate preprocessing.

Convert the legacy `scripts/run_hdbscan_street_typology.py` entry point into a
thin compatibility wrapper around the modular HDBSCAN runner. It prints a
deprecation warning, requires a prepared scenario directory, and contains no
feature discovery, imputation, or scaling logic. No active HDBSCAN entry point
may use median imputation.

## Error Handling

Stop before algorithm execution when:

- the input file is absent, unreadable, or empty;
- required identifiers or the active risk index are missing;
- no scenario-specific feature is discovered;
- an exclusion artifact references an absent or opposite-scenario feature;
- a selected feature is non-numeric;
- a selected feature contains infinity;
- a finite semantic value is outside its configured range;
- all selected features are constant after structural-zero replacement;
- a prepared artifact is incomplete;
- source, exclusion, feature-order, or matrix hashes disagree;
- standardized values are non-finite;
- K is invalid for the prepared sample count;
- the final output directory already exists under the existing
  non-overwrite policy.

Warnings, not fatal errors, cover:

- duplicate identifier values when stable source-row position still preserves
  alignment;
- high structural-null rates;
- removal of a constant feature;
- highly imbalanced candidate cluster sizes;
- PCA components that explain little cumulative variance.

## Testing Strategy

### Shared preprocessing unit tests

Test that:

- PF and TF markers classify all expected naming patterns;
- unrelated substrings are not misclassified;
- identifiers, indexes, risks, results, and labels are excluded;
- source-column order becomes model-feature order;
- structural null becomes `0.0`;
- real `-1`, `0`, and `1` remain unchanged before scaling;
- the source DataFrame is not mutated;
- row count, row order, and identifiers are preserved;
- per-feature and per-row null audits are correct;
- malformed strings, infinity, and out-of-range finite values fail;
- constant features are consistently removed and recorded;
- an all-unusable feature set fails clearly;
- standardized features have mean near zero and `ddof=0` standard deviation
  near one;
- inverse transformation recovers the filled matrix within tolerance;
- repeated preparation of identical inputs is deterministic.

### Consumer contract tests

Using one fixture and one prepared artifact, verify that:

- PCA reads the standardized matrix without re-scaling;
- K selection reads the same matrix and feature order;
- K-means reads the same matrix and returns one label per row;
- primary HDBSCAN reads the same matrix;
- no consumer calls independent `fillna`, feature discovery, or
  `StandardScaler.fit`;
- each output records the same preprocessing manifest and matrix hash.

### End-to-end regression tests

For the current v3 high-risk data:

- PF preparation retains 4,100 rows;
- TF preparation retains 13,584 rows;
- rows removed for structural nulls equal zero;
- final K-means label count equals prepared row count;
- `kmeans_cluster` has no missing value;
- centroids can be inverse-transformed;
- output audits reconcile to source null counts.

Future dataset versions assert internal reconciliation rather than hard-coded
v3 counts.

## Intended Run Order

After implementation, the research workflow is:

```text
1. clean_mt_update_ms_hev.py
2. filter_high_risk_dataset.py for PF
3. filter_high_risk_dataset.py for TF
4. correlation_analysis.py for PF and TF
5. review/write explicit exclusion artifacts
6. prepare_clustering_inputs.py for PF and TF
7. pca_analysis.py for diagnostic interpretation
8. choose_k.py for K evaluation
9. kmeans_analysis.py with the selected K
10. optionally run primary HDBSCAN from the same prepared matrix
```

Steps 7 and 8 are independent after preparation. PCA is listed first for the
human research workflow, not because K selection or K-means depends on PCA
scores.

If the source data, threshold, feature rules, or correlation exclusions
change, regenerate the prepared artifact and rerun all downstream analyses.

## Acceptance Criteria

The design is implemented successfully when:

1. PF/TF model features are discovered dynamically from documented markers.
2. No active workflow depends on a manually edited complete PF/TF feature
   tuple.
3. Structural nulls are preserved in semantic source data and converted to
   zero only in model preparation.
4. No PF/TF high-risk row is removed because of a structural null.
5. One scenario-specific StandardScaler is fitted after row and feature
   selection.
6. PCA, K selection, K-means, and primary HDBSCAN consume the same
   standardized matrix and feature order for a scenario run.
7. K selection and final K-means use identical input geometry.
8. Standardized and inverse-transformed centroids are both available.
9. Every output records source, feature, exclusion, scaler, and matrix
   provenance.
10. Unit, consumer-contract, and end-to-end regression tests pass.
