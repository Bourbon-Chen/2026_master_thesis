# Modular HDBSCAN Research Pipeline

The original `run_hdbscan_street_typology.py` script and legacy
`outputs_hdbscan/PF` and `outputs_hdbscan/TF` directories remain unchanged.
The modular pipeline writes only under dataset-specific folders inside
`outputs_hdbscan/pipeline_v1`.

This is an unsupervised clustering experiment. It explores whether street
typologies emerge from HEV indicators; it is not supervised training.

## Research Logic

- PF and TF are processed separately.
- `Index_Risk` is included as a clustering feature.
- Other composite `Index` columns are excluded.
- Missing feature values are filled with `0`.
- Version A uses normalized features directly.
- Version B applies `StandardScaler` after fill-zero imputation.
- Each experiment runs one user-selected HDBSCAN parameter pair.
- Version A defines the input file and selected parameter pair.
- Version B reads the last Version A configuration for the same input dataset.
- HDBSCAN keeps the fixed `euclidean` / `eom` settings.

## Run Clustering

Recommended full clustering workflow:

```powershell
python scripts\clustering\run_hdbscan_version_a.py
python scripts\clustering\run_hdbscan_version_b.py
```

Version A prompts for:

```text
input file
min_cluster_size
min_samples
```

Version B prompts only for `input file`, uses it to locate the dataset folder,
and then reuses:

```text
outputs_hdbscan/pipeline_v1/<dataset_name>/logs/last_version_a_config.json
```

Outputs are grouped by input file stem. For example:

```text
outputs_hdbscan/pipeline_v1/MT_UPDATE_MS_HEV_NORcleaned/
```

For reproducible non-interactive Version A execution:

```powershell
python scripts\clustering\run_hdbscan_version_a.py --input data\MT_UPDATE_MS_HEV_NORcleaned.csv --min-cluster-size 100 --min-samples 20
python scripts\clustering\run_hdbscan_version_b.py --input data\MT_UPDATE_MS_HEV_NORcleaned.csv
```

Run one scenario:

```powershell
python scripts\clustering\run_hdbscan_version_a.py --scenario PF
python scripts\clustering\run_hdbscan_version_b.py
```

The convenience wrapper still exists. It runs Version A and Version B in one
process, using the same input and parameters:

```powershell
python scripts\clustering\run_hdbscan_experiment.py
```

Audit feature selection without running HDBSCAN:

```powershell
python scripts\clustering\run_hdbscan_version_a.py --audit-only
```

Completed modular runs with the same scenario, version, and parameter pair are
immutable: the runner stops with an error instead of overwriting or mixing
outputs. Use a new output root to run another experiment.

## Regenerate Outputs Without HDBSCAN

```powershell
python scripts\visualization\generate_pca_plots.py
python scripts\visualization\generate_cluster_maps.py
python scripts\export\export_cluster_summary.py
python scripts\export\export_cluster_tables.py
```

Each post-processing script asks for the input file path, then reads results
from the corresponding `pipeline_v1/<dataset_name>/` folder. You can also pass
`--input` non-interactively.

The current CSV input has no geometry. `generate_cluster_maps.py` therefore
exports GIS join tables for QGIS. Join with the street geometry layer using
`MS_ID` or `fid`.
