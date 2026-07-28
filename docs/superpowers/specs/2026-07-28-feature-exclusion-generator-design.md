# Feature Exclusion Generator Design

## Goal

Close the workflow gap between Pearson correlation review and PCA by adding a
small command-line tool that converts one user-entered list of feature names
into the JSON artifact already accepted by `prepare_clustering_inputs.py`.

## Current Problem

`extract_high_columns.py` is a legacy compatibility exporter. It creates
`pf_for_PCA.csv` or `tf_for_PCA.csv`, but active PCA, K-selection, K-means, and
HDBSCAN entry points do not consume those files. They consume one verified
prepared artifact.

The active preprocessor already accepts a documented exclusions JSON through
`--exclusions`, but no user-facing tool currently creates that JSON. Users can
review correlation results, but must manually know and reproduce the JSON
schema before the selected variables can be excluded consistently from all
downstream analyses.

## Approved Approach

Add one independent script:

```text
scripts/pearson_correlation/build_feature_exclusions.py
```

The script does not inspect, rank, display, or make decisions from individual
high-correlation pairs. The researcher reviews the correlation outputs
separately and enters the complete exclusion list once.

The generator has one responsibility:

```text
user-entered feature names
    -> validation against feature_roles.csv
    -> minimal exclusions JSON
```

`extract_high_columns.py` remains a legacy exporter and is not restored to the
active modelling path.

## Command-Line Interface

The scenario is required:

```powershell
python scripts/pearson_correlation/build_feature_exclusions.py --scenario PF
```

When `--exclude` is omitted, the script prompts exactly once:

```text
Columns to exclude, separated by commas (blank keeps all):
```

An example response is:

```text
Per_extent, PFAB2k_NOR, PFPRM_lossR, N_PFlossR_2ktw
```

The same decision can be supplied non-interactively:

```powershell
python scripts/pearson_correlation/build_feature_exclusions.py --scenario PF --exclude "Per_extent,PFAB2k_NOR,PFPRM_lossR,N_PFlossR_2ktw"
```

Supported arguments are:

```text
--scenario {PF,TF}    required, case-insensitive
--exclude TEXT        optional comma-separated exclusion list
--roles PATH          optional feature-roles CSV override
--output PATH         optional exclusions JSON override
--overwrite           optional permission to replace an existing output
```

The defaults depend on the canonical scenario:

```text
PF roles:   outputs_correlation/pf/feature_roles.csv
PF output:  outputs_correlation/pf/feature_exclusions.json
TF roles:   outputs_correlation/tf/feature_roles.csv
TF output:  outputs_correlation/tf/feature_exclusions.json
```

## Input Parsing

The exclusion text accepts ASCII commas and Chinese commas. Each item is
trimmed. Empty items are discarded. Duplicate names are removed while
preserving the first occurrence and the user's original order.

A blank prompt response or blank `--exclude` value is valid and produces an
empty exclusion list.

Column-name matching remains exact and case-sensitive because the names are
persisted as source-data column identifiers.

## Feature-Role Validation

The roles CSV must exist and contain:

```text
column
role
reason
```

For PF, an entered name is valid only when its role is `PF_FEATURE`. For TF, it
is valid only when its role is `TF_FEATURE`.

The complete input is validated before anything is written. Invalid names are
reported by category:

- absent from `feature_roles.csv`;
- belongs to the opposite scenario;
- protected identifier, composite index, result, label, or shared field;
- already excluded by a built-in rule;
- non-numeric active-scenario candidate.

When a raw AB lossR column has reason
`raw_ab_lossR_replaced_by_normalized_NOR`, the message explicitly states that
the column is already excluded and should not be added to the user exclusion
list.

The script reports every invalid item in one error so the user can correct the
list in one pass. Interactive invalid input is not repeatedly prompted; the
command exits non-zero and writes nothing. This keeps the interaction to the
single requested prompt.

## Output Contract

The JSON contains only the three fields required by the existing loader:

```json
{
  "scenario": "PF",
  "excluded_features": [
    "Per_extent",
    "PFAB2k_NOR",
    "PFPRM_lossR",
    "N_PFlossR_2ktw"
  ],
  "reason": "User-selected exclusions after Pearson correlation review"
}
```

The field order is stable. The scenario is canonical uppercase. The feature
list preserves user order. The reason is the exact stable value:

```text
User-selected exclusions after Pearson correlation review
```

After writing, the script prints the output path and a compact list of the
selected exclusions. It also prints that the generated path must be passed to
`prepare_clustering_inputs.py --exclusions`.

## File Safety

If the destination already exists, the command fails before prompting unless
`--overwrite` is present. This prevents accidental loss of a previous research
decision.

For a new file or an approved overwrite, JSON is first written to a temporary
file in the destination directory and then published with `Path.replace`.
Parsing, validation, interruption, or temporary-write failures leave an
existing decision file unchanged.

The generator never changes correlation CSVs, high-risk source CSVs, prepared
artifacts, PCA outputs, or clustering outputs.

## Active Workflow

The documented workflow becomes:

```text
correlation_analysis.py
    -> researcher reviews correlation CSVs and plots
build_feature_exclusions.py
    -> feature_exclusions.json
prepare_clustering_inputs.py --exclusions feature_exclusions.json
    -> immutable verified prepared artifact
pca_analysis.py / choose_k.py / kmeans_analysis.py / hdbscan_runner.py
    -> all consume the same selected and standardized matrix
```

Because prepared artifact directories are immutable, a changed exclusion
decision requires a new preparation output root. PCA may then republish its
managed result files to `output_pca/pf` or `output_pca/tf`.

## Testing

Automated tests cover:

- case-insensitive PF/TF scenario parsing;
- one prompt when `--exclude` is omitted;
- non-interactive `--exclude`;
- ASCII and Chinese comma parsing;
- whitespace removal, empty-item removal, stable order, and deduplication;
- blank input producing an empty exclusion list;
- validation of active-scenario roles;
- combined reporting for absent, opposite-scenario, protected, built-in
  excluded, and non-numeric columns;
- the explicit raw AB lossR already-excluded message;
- exact minimal JSON structure and reason;
- default PF and TF paths plus path overrides;
- refusal to overwrite by default;
- successful explicit overwrite;
- interruption or validation failure preserving an existing file;
- direct use of the generated JSON by
  `prepare_clustering_inputs.load_exclusions`;
- end-to-end preparation excluding the selected feature from the prepared
  matrix;
- the full project regression suite in the `2026master` environment.

## Documentation

README workflow instructions will add the generator between correlation
analysis and clustering preparation. The preparation examples will pass the
scenario-specific `feature_exclusions.json` through `--exclusions`.

The documentation will continue to identify `extract_high_columns.py` as
legacy so users do not mistake its CSV export for an active PCA input.

## Scope

This change does not automatically choose variables, inspect pairwise
correlation rows, modify the correlation threshold, delete columns from source
CSVs, run preparation automatically, or run PCA automatically.
