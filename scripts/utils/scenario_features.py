"""Shared PF/TF feature definitions for cluster preparation and modelling."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PF_FEATURES = (
    "N_PFlossR_2kiw",
    "N_PFlossR_2kmw",
    "PFResident_lossR",
    "PFDaynight_lossR",
    "PFAB5k_NOR",
    "PFABC_NOR",
) # 2. 已经删掉"Per_extent"和"Tem_extent"列，再尝试
TF_FEATURES = (
    "N_TFlossR_2kiw",
    "N_TFlossR_2kmw",
    "TFResident_lossR",
    "TFDaynight_lossR",
    "TFAB5k_NOR",
    "TFABC_NOR",
)

PF_METADATA_COLUMNS = ("fid", "MS_ID", "PF_Index_Risk_equal")
TF_METADATA_COLUMNS = ("fid", "MS_ID", "TF_Index_Risk_equal")

DEFAULT_INPUT_PATHS: Dict[str, Path] = {
    "PF": PROJECT_ROOT / "outputs_correlation" / "pf" / "pf_for_PCA.csv",
    "TF": PROJECT_ROOT / "outputs_correlation" / "tf" / "tf_for_PCA.csv",
}

SCENARIO_CONFIG: Dict[str, Dict[str, Sequence[str]]] = {
    "PF": {"features": PF_FEATURES, "metadata": PF_METADATA_COLUMNS}, # 1. 删掉"Per_extent"和"Tem_extent"列后报错
    "TF": {"features": TF_FEATURES, "metadata": TF_METADATA_COLUMNS},
}
