"""Deprecated compatibility adapters for canonical clustering preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .clustering_preprocessing import (
    FeatureDiscovery,
    FieldRole,
    PreparedArtifact,
    PreparedClusteringData,
    discover_scenario_features,
    load_prepared_artifact,
    prepare_clustering_data,
)


@dataclass
class PreparedFeatures:
    """Deprecated HDBSCAN-facing view of canonical prepared clustering data."""

    values: pd.DataFrame
    feature_names: list[str]
    id_columns: list[str]
    rule_label_column: Optional[str]
    summary: pd.DataFrame
    non_numeric_columns: list[str]
    constant_columns: list[str]
    extreme_columns: pd.DataFrame


def _feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = features.agg(["min", "max", "mean", "std"]).transpose()
    summary["missing_value_count"] = features.isna().sum()
    summary.index.name = "feature"
    return summary.reset_index()


def prepare_features(data: pd.DataFrame, scenario: str) -> PreparedFeatures:
    """Deprecated: adapt canonical preprocessing for legacy HDBSCAN callers."""
    prepared = prepare_clustering_data(data, scenario)
    summary = _feature_summary(prepared.original_features)
    extremes = summary[(summary["min"] < -1) | (summary["max"] > 1)].copy()
    excluded = prepared.excluded_columns
    rule_labels = excluded.loc[
        excluded["role"].eq(FieldRole.RESULT_OR_LABEL.value), "column"
    ].tolist()
    non_numeric_columns = excluded.loc[
        excluded["role"].eq(FieldRole.NON_NUMERIC.value), "column"
    ].tolist()
    constant_columns = excluded.loc[
        excluded["reason"].eq("constant_after_structural_zero"), "column"
    ].tolist()
    return PreparedFeatures(
        values=prepared.filled_features,
        feature_names=list(prepared.feature_names),
        id_columns=[
            name for name in ("fid", "MS_ID") if name in prepared.metadata
        ],
        rule_label_column=rule_labels[0] if rule_labels else None,
        summary=summary,
        non_numeric_columns=non_numeric_columns,
        constant_columns=constant_columns,
        extreme_columns=extremes,
    )


def save_feature_audit(prepared: PreparedFeatures, output_dir: Path) -> None:
    """Deprecated: write audit files expected by legacy HDBSCAN runners."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.summary.to_csv(output_dir / "feature_summary.csv", index=False)
    pd.DataFrame({"feature": prepared.feature_names}).to_csv(
        output_dir / "selected_features.csv", index=False
    )
    pd.DataFrame(
        {"excluded_non_numeric_column": prepared.non_numeric_columns}
    ).to_csv(output_dir / "excluded_non_numeric_columns.csv", index=False)
    pd.DataFrame({"excluded_constant_column": prepared.constant_columns}).to_csv(
        output_dir / "excluded_constant_columns.csv", index=False
    )
    prepared.extreme_columns.to_csv(
        output_dir / "features_outside_minus1_to_1.csv", index=False
    )
