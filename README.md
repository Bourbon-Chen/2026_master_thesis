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
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv --scenario PF
python scripts/cluster_preparation/prepare_clustering_inputs.py --input data/MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv --scenario TF
python scripts/cluster_preparation/pca_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
python scripts/cluster_preparation/choose_k.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF
python scripts/cluster_kmeans/kmeans_analysis.py --prepared outputs_preprocessing/MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25/PF --k 4
```

For TF, repeat the last three commands with the TF prepared artifact. PCA is
diagnostic only; K-means uses the complete standardized prepared matrix, not
PCA scores. Changing the source file, risk threshold, or feature exclusions
requires a new output root and regeneration of the prepared artifact and all
downstream results.

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
