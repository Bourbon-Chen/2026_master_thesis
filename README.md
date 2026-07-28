# GeoAI-based Flood Resilience Assessment

## Project Overview

This project develops a GeoAI workflow for flood resilience assessment.

## Confirmed v3 clustering workflow

Run the v3 workflow from the repository root in this order:

```powershell
python scripts/clean_mt_update_ms_hev.py
python scripts/filter_high_risk_dataset.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned.csv --scenario PF --threshold 0.25
python scripts/filter_high_risk_dataset.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned.csv --scenario TF --threshold 0.25
python scripts/pearson_correlation/correlation_analysis.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF
python scripts/pearson_correlation/correlation_analysis.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF
python scripts/pearson_correlation/build_feature_exclusions.py --scenario PF
python scripts/pearson_correlation/build_feature_exclusions.py --scenario TF
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF --exclusions outputs_correlation/pf/feature_exclusions.json
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF --exclusions outputs_correlation/tf/feature_exclusions.json
python scripts/cluster_preparation/pca_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
python scripts/cluster_preparation/choose_k.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
python scripts/cluster_kmeans/kmeans_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF --k 4
```

The exclusion generator does not choose correlated variables automatically.
Review each scenario's correlation tables or plots first, then enter the full
comma-separated list once. A blank list is valid and keeps every eligible
feature. To replace an existing decision file, rerun the generator with
`--overwrite`.

`feature_exclusions.json` is the active bridge from correlation review to
PCA, K selection, K-means, and HDBSCAN. `extract_high_columns.py` remains a
legacy compatibility exporter; its `pf_for_PCA.csv` and `tf_for_PCA.csv`
files are not active PCA inputs.

For TF, repeat the last three commands with the TF prepared artifact. PCA is
diagnostic only; K-means uses the complete standardized prepared matrix, not
PCA scores. Changing the source file, risk threshold, or feature exclusions
requires a new `--output-root` for `prepare_clustering_inputs.py`, followed by
regeneration of every downstream result from that new prepared artifact.

PCA may be rerun against the same prepared artifact without choosing a new
output root. By default it republishes its ten managed result files in
`output_pca/pf` or `output_pca/tf`. Other files and subdirectories in those
scenario directories are preserved. Use `--output-root` only when a separate
PCA experiment directory is intentional.

Raw `PF/TF AB2k/AB5k lossR` columns, including their `N_` variants, are
built-in exclusions in the shared feature classifier. Their `AB2k_NOR` and
`AB5k_NOR` counterparts remain eligible. Pearson correlation and clustering
preparation therefore use the same feature decision.

## Structure

- data/
- notebooks/
- src/

## Methods

- QGIS
- Random Forest
- Spatial Network Analysis

## Results

...
