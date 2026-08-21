# Shared Exclusion of Raw AB lossR Features

## Goal

Exclude raw 2 km and 5 km AB loss-ratio variables from both Pearson
correlation candidates and clustering inputs while retaining their normalized
`_NOR` counterparts.

## Scientific rule

The raw AB loss-ratio columns are intermediate values whose normalized
counterparts are the intended modelling variables. Including both would add
redundant representations of the same concept.

The shared classifier must exclude these columns for both scenarios:

```text
PFAB2k_lossR
N_PFAB2k_lossR
PFAB5k_lossR
N_PFAB5k_lossR
TFAB2k_lossR
N_TFAB2k_lossR
TFAB5k_lossR
N_TFAB5k_lossR
```

It must retain:

```text
PFAB2k_NOR
PFAB5k_NOR
TFAB2k_NOR
TFAB5k_NOR
```

## Classification design

`scripts/utils/clustering_preprocessing.py` remains the only owner of this
decision. Add one anchored, case-insensitive built-in exclusion pattern:

```regex
^(?:N_)?(?:PF|TF)AB(?:2k|5k)_lossR$
```

The built-in exclusion is evaluated after identifier, composite Index/Risk,
and result/label precedence checks, but before PF/TF candidate assignment.
Every match receives:

```text
role   = EXCLUDED
reason = raw_ab_lossR_replaced_by_normalized_NOR
```

This built-in rule is independent of optional research-exclusion JSON files.
An exclusion JSON remains available for dataset-specific decisions, but users
do not need to repeat the AB rule in each run.

## Consumer behavior

Both consumer paths already use shared semantic discovery:

- `correlation_analysis.py` calls `prepare_semantic_features(...)`;
- `prepare_clustering_inputs.py` calls `prepare_clustering_data(...)`.

Therefore the shared rule automatically removes raw AB lossR columns from:

- Pearson matrices, pair tables, summaries, and heatmaps;
- `features_original.csv`, `features_filled.csv`, and
  `features_standardized.csv`;
- PCA, choose-K, K-means, and HDBSCAN inputs.

The source high-risk CSV is never modified. `feature_roles.csv` and
`excluded_features.csv` retain the excluded column names and the documented
reason for auditability.

## Validation and tests

Automated tests must prove:

1. all eight exact raw AB lossR names are classified as `EXCLUDED`;
2. the four `_NOR` counterparts remain PF/TF features;
3. matching is case-insensitive and anchored, so incidental names are not
   removed;
4. PF discovery never selects PF raw AB lossR columns;
5. TF discovery never selects TF raw AB lossR columns;
6. correlation preparation and clustering preparation inherit the same
   selected feature set and exclusion reason.

The complete regression suite must continue to pass in the `2026master`
environment.

## Regeneration and non-overwrite behavior

Existing correlation outputs and prepared artifacts do not change
retroactively. After implementation:

1. rerun Pearson correlation into a fresh or cleared correlation output
   location;
2. rerun `prepare_clustering_inputs.py` with a new output root because
   prepared artifacts are immutable;
3. rerun PCA, choose-K, K-means, and HDBSCAN from the new artifact.

No implementation step may delete or overwrite existing user outputs.
