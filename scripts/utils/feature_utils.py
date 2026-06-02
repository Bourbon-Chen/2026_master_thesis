"""Feature selection and preparation shared by clustering and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.config import (
    EXCLUDED_FEATURE_KEYWORDS,
    ID_COLUMN_CANDIDATES,
    RULE_LABEL_CANDIDATES,
)


@dataclass
class PreparedFeatures:
    values: pd.DataFrame
    feature_names: list[str]
    id_columns: list[str]
    rule_label_column: Optional[str]
    summary: pd.DataFrame
    non_numeric_columns: list[str]
    constant_columns: list[str]
    extreme_columns: pd.DataFrame


def find_rule_label_column(columns: pd.Index) -> Optional[str]:
    by_lowercase = {column.lower(): column for column in columns}
    for candidate in RULE_LABEL_CANDIDATES:
        if candidate.lower() in by_lowercase:
            return by_lowercase[candidate.lower()]
    return None


def belongs_to_scenario(column: str, scenario: str) -> bool:
    other_scenario = "TF" if scenario == "PF" else "PF"
    return other_scenario not in column.upper()


def is_excluded_feature(column: str, id_columns: list[str]) -> bool:
    lowercase = column.lower()
    if column in id_columns:
        return True
    if lowercase in {candidate.lower() for candidate in RULE_LABEL_CANDIDATES}:
        return True
    if any(keyword in lowercase for keyword in EXCLUDED_FEATURE_KEYWORDS):
        return True
    # Risk index is intentionally included; other composite indexes are excluded.
    if "index_risk" in lowercase:
        return False
    return "index" in lowercase or "risk" in lowercase


def select_features(
    data: pd.DataFrame, scenario: str
) -> tuple[list[str], list[str], list[str], list[str]]:
    id_columns = [column for column in ID_COLUMN_CANDIDATES if column in data.columns]
    scenario_columns = [
        column for column in data.columns if belongs_to_scenario(column, scenario)
    ]
    candidates = [
        column
        for column in scenario_columns
        if not is_excluded_feature(column, id_columns)
    ]
    numeric_columns = set(data[candidates].select_dtypes(include=[np.number]))
    non_numeric_columns = [
        column for column in candidates if column not in numeric_columns
    ]
    numeric_candidates = [
        column for column in candidates if column in numeric_columns
    ]
    constant_columns = [
        column
        for column in numeric_candidates
        if data[column].nunique(dropna=True) <= 1
    ]
    features = [
        column for column in numeric_candidates if column not in constant_columns
    ]
    return features, id_columns, non_numeric_columns, constant_columns


def build_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = features.agg(["min", "max", "mean", "std"]).transpose()
    summary["missing_value_count"] = features.isna().sum()
    summary.index.name = "feature"
    return summary.reset_index()


def prepare_features(data: pd.DataFrame, scenario: str) -> PreparedFeatures:
    features, id_columns, non_numeric_columns, constant_columns = select_features(
        data, scenario
    )
    if not features:
        raise ValueError(f"No clustering features remain for scenario {scenario}.")
    raw_values = data[features].copy()
    summary = build_feature_summary(raw_values)
    extremes = summary[(summary["min"] < -1) | (summary["max"] > 1)].copy()
    values = raw_values.fillna(0)
    if values.isna().any().any():
        raise ValueError("Missing feature values remain after fill-zero imputation.")
    return PreparedFeatures(
        values=values,
        feature_names=features,
        id_columns=id_columns,
        rule_label_column=find_rule_label_column(data.columns),
        summary=summary,
        non_numeric_columns=non_numeric_columns,
        constant_columns=constant_columns,
        extreme_columns=extremes,
    )


def save_feature_audit(prepared: PreparedFeatures, output_dir: Path) -> None:
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
