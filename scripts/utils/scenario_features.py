"""Compatibility metadata and source paths for PF/TF clustering workflows."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PF_METADATA_COLUMNS = (
    "source_row_position",
    "fid",
    "MS_ID",
    "PF_Index_Risk_equal",
)
TF_METADATA_COLUMNS = (
    "source_row_position",
    "fid",
    "MS_ID",
    "TF_Index_Risk_equal",
)

DEFAULT_INPUT_PATHS = {
    "PF": (
        PROJECT_ROOT
        / "data"
        / "MT_UPDATE_MS_HEV_v3_NORcleaned_pf_risk_gt_0_25.csv"
    ),
    "TF": (
        PROJECT_ROOT
        / "data"
        / "MT_UPDATE_MS_HEV_v3_NORcleaned_tf_risk_gt_0_25.csv"
    ),
}

SCENARIO_CONFIG = {
    "PF": {"metadata": PF_METADATA_COLUMNS},
    "TF": {"metadata": TF_METADATA_COLUMNS},
}
