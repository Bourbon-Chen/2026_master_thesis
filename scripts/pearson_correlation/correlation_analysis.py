#!/usr/bin/env python3
"""Analyze Pearson correlations among PF and TF clustering candidates.

The script profiles candidate features, reports data-quality concerns, and
calculates scenario-specific Pearson correlations using pairwise complete
finite observations. It never edits the source data or automatically removes
features. Correlation identifies linear association, not causation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIRECTORY / "data" / "MT_UPDATE_MS_HEV_v2_NORcleaned.csv"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs_correlation"
CORRELATION_THRESHOLD = 0.6
IDENTIFIER_COLUMNS = ("fid", "MS_ID")

PF_FEATURE_GROUPS: Dict[str, Sequence[str]] = {
    "flood_exposure": ["Per_extent"],
    "accessibility_loss": [
        "N_PFlossR_2kiw",
        "N_PFlossR_2ktw",
        "N_PFlossR_2kmw",
    ],
    "population_loss": [
        "PFResident_lossR",
        "PFDaynight_lossR",
        "PFPRM_lossR",
    ],
    "centrality_influence": ["PFAB2k_NOR", "PFAB5k_NOR", "PFABC_NOR"],
}

TF_FEATURE_GROUPS: Dict[str, Sequence[str]] = {
    "flood_exposure": ["Tem_extent"],
    "accessibility_loss": [
        "N_TFlossR_2kiw",
        "N_TFlossR_2ktw",
        "N_TFlossR_2kmw",
    ],
    "population_loss": [
        "TFResident_lossR",
        "TFDaynight_lossR",
        "TFPRM_lossR",
    ],
    "centrality_influence": ["TFAB2k_NOR", "TFAB5k_NOR", "TFABC_NOR"],
}

SCENARIO_FEATURE_GROUPS: Dict[str, Mapping[str, Sequence[str]]] = {
    "PF": PF_FEATURE_GROUPS,
    "TF": TF_FEATURE_GROUPS,
}

EXPECTED_RANGES: Dict[str, Tuple[float, float]] = {
    "flood_exposure": (0.0, 1.0),
    "accessibility_loss": (-1.0, 1.0),
    "population_loss": (-1.0, 0.0),
    "centrality_influence": (0.0, 1.0),
}


FEATURE_SUMMARY_COLUMNS = [
    "scenario",
    "feature",
    "theoretical_group",
    "data_type",
    "count",
    "missing_count",
    "missing_percentage",
    "unique_count",
    "min",
    "max",
    "mean",
    "median",
    "standard_deviation",
    "first_quartile",
    "third_quartile",
]

DATA_QUALITY_WARNING_COLUMNS = [
    "scenario",
    "feature",
    "theoretical_group",
    "warning_type",
    "severity",
    "affected_count",
    "details",
]

CORRELATION_PAIR_COLUMNS = [
    "scenario",
    "feature_1",
    "group_1",
    "feature_2",
    "group_2",
    "pearson_r",
    "absolute_r",
    "pairwise_n",
    "same_group",
    "correlation_direction",
    "threshold_exceeded",
]

COMBINED_SUMMARY_COLUMNS = [
    "scenario",
    "number_of_features",
    "number_of_all_pairs",
    "number_of_high_correlation_pairs",
    "number_of_within_group_high_pairs",
    "number_of_cross_group_high_pairs",
    "maximum_absolute_correlation",
    "feature_1_of_max_pair",
    "feature_2_of_max_pair",
    "pearson_r_of_max_pair",
]


def load_data(input_file: Path) -> pd.DataFrame:
    """Load a CSV file without modifying its contents."""
    input_file = Path(input_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_file}")
    try:
        return pd.read_csv(input_file, low_memory=False)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Unable to read input CSV '{input_file}': {error}") from error


def _flatten_features(feature_groups: Mapping[str, Sequence[str]]) -> list:
    """Return configured features in theoretical-group order."""
    return [feature for features in feature_groups.values() for feature in features]


def validate_columns(
    data: pd.DataFrame,
    scenario_feature_groups: Mapping[str, Mapping[str, Sequence[str]]],
    identifier_columns: Sequence[str] = IDENTIFIER_COLUMNS,
) -> None:
    """Print all absent configured columns and raise ``ValueError`` if any are missing."""
    required_columns = list(identifier_columns)
    for feature_groups in scenario_feature_groups.values():
        required_columns.extend(_flatten_features(feature_groups))

    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        print("ERROR: Missing required columns:")
        for column in missing_columns:
            print(f"  - {column}")
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))


def build_feature_group_mapping(
    feature_groups: Mapping[str, Sequence[str]],
) -> Dict[str, str]:
    """Map every feature name to its configured theoretical group."""
    mapping: Dict[str, str] = {}
    for theoretical_group, features in feature_groups.items():
        for feature in features:
            if feature in mapping:
                raise ValueError(f"Feature configured more than once: {feature}")
            mapping[feature] = theoretical_group
    return mapping


def create_feature_summary(
    data: pd.DataFrame,
    scenario: str,
    feature_groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Create descriptive and missingness statistics for scenario features."""
    records = []
    total_rows = len(data)
    for theoretical_group, features in feature_groups.items():
        for feature in features:
            series = data[feature]
            count = int(series.count())
            missing_count = int(series.isna().sum())
            record: Dict[str, Any] = {
                "scenario": scenario,
                "feature": feature,
                "theoretical_group": theoretical_group,
                "data_type": str(series.dtype),
                "count": count,
                "missing_count": missing_count,
                "missing_percentage": (
                    missing_count / total_rows * 100.0 if total_rows else np.nan
                ),
                "unique_count": int(series.nunique(dropna=True)),
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "standard_deviation": np.nan,
                "first_quartile": np.nan,
                "third_quartile": np.nan,
            }
            if is_numeric_dtype(series.dtype):
                numeric = pd.to_numeric(series, errors="coerce")
                record.update(
                    {
                        "min": numeric.min(),
                        "max": numeric.max(),
                        "mean": numeric.mean(),
                        "median": numeric.median(),
                        "standard_deviation": numeric.std(ddof=1),
                        "first_quartile": numeric.quantile(0.25),
                        "third_quartile": numeric.quantile(0.75),
                    }
                )
            records.append(record)

    return pd.DataFrame(records, columns=FEATURE_SUMMARY_COLUMNS)


def _quality_warning(
    scenario: str,
    feature: str,
    theoretical_group: str,
    warning_type: str,
    severity: str,
    affected_count: int,
    details: str,
) -> Dict[str, Any]:
    """Build one stable-schema data-quality warning record."""
    return {
        "scenario": scenario,
        "feature": feature,
        "theoretical_group": theoretical_group,
        "warning_type": warning_type,
        "severity": severity,
        "affected_count": int(affected_count),
        "details": details,
    }


def check_data_quality(
    data: pd.DataFrame,
    scenario: str,
    feature_groups: Mapping[str, Sequence[str]],
    expected_ranges: Mapping[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Identify requested quality problems without changing source values."""
    warning_records = []

    for theoretical_group, features in feature_groups.items():
        expected_minimum, expected_maximum = expected_ranges[theoretical_group]
        for feature in features:
            series = data[feature]
            missing_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))

            if missing_count == len(series):
                warning_records.append(
                    _quality_warning(
                        scenario,
                        feature,
                        theoretical_group,
                        "all_missing",
                        "critical",
                        missing_count,
                        "All observations are missing.",
                    )
                )
            if unique_count == 1:
                warning_records.append(
                    _quality_warning(
                        scenario,
                        feature,
                        theoretical_group,
                        "constant_variable",
                        "high",
                        int(series.notna().sum()),
                        "The feature has exactly one unique non-missing value; "
                        "Pearson correlation is undefined.",
                    )
                )

            if not is_numeric_dtype(series.dtype):
                warning_records.append(
                    _quality_warning(
                        scenario,
                        feature,
                        theoretical_group,
                        "non_numeric",
                        "high",
                        int(series.notna().sum()),
                        f"Configured candidate has non-numeric dtype '{series.dtype}'.",
                    )
                )
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric.to_numpy(dtype=float)
            positive_infinity_count = int(np.isposinf(values).sum())
            negative_infinity_count = int(np.isneginf(values).sum())
            if positive_infinity_count:
                warning_records.append(
                    _quality_warning(
                        scenario,
                        feature,
                        theoretical_group,
                        "positive_infinity",
                        "high",
                        positive_infinity_count,
                        "Positive infinity is excluded from affected Pearson pairs.",
                    )
                )
            if negative_infinity_count:
                warning_records.append(
                    _quality_warning(
                        scenario,
                        feature,
                        theoretical_group,
                        "negative_infinity",
                        "high",
                        negative_infinity_count,
                        "Negative infinity is excluded from affected Pearson pairs.",
                    )
                )

            finite_mask = np.isfinite(values)
            out_of_range_mask = finite_mask & (
                (values < expected_minimum) | (values > expected_maximum)
            )
            out_of_range_count = int(out_of_range_mask.sum())
            if out_of_range_count:
                finite_outliers = values[out_of_range_mask]
                warning_records.append(
                    _quality_warning(
                        scenario,
                        feature,
                        theoretical_group,
                        "out_of_expected_range",
                        "medium",
                        out_of_range_count,
                        (
                            f"Expected [{expected_minimum:g}, {expected_maximum:g}]; "
                            f"observed out-of-range values from "
                            f"{finite_outliers.min():g} to {finite_outliers.max():g}. "
                            "Values were not modified."
                        ),
                    )
                )

    warnings = pd.DataFrame(warning_records, columns=DATA_QUALITY_WARNING_COLUMNS)
    for warning in warning_records:
        print(
            "WARNING "
            f"[{scenario}] {warning['feature']} - {warning['warning_type']}: "
            f"{warning['details']}"
        )
    return warnings


def calculate_pairwise_correlation(
    data: pd.DataFrame, features: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson coefficients and sample counts from identical pair masks."""
    feature_list = list(features)
    correlation_matrix = pd.DataFrame(
        np.nan, index=feature_list, columns=feature_list, dtype=float
    )
    pairwise_counts = pd.DataFrame(
        0, index=feature_list, columns=feature_list, dtype=int
    )
    numeric_data = {
        feature: pd.to_numeric(data[feature], errors="coerce")
        for feature in feature_list
    }

    for left_index, left_feature in enumerate(feature_list):
        left = numeric_data[left_feature]
        for right_index in range(left_index, len(feature_list)):
            right_feature = feature_list[right_index]
            right = numeric_data[right_feature]
            valid_mask = pd.Series(
                np.isfinite(left.to_numpy(dtype=float))
                & np.isfinite(right.to_numpy(dtype=float)),
                index=data.index,
            )
            pairwise_n = int(valid_mask.sum())
            pairwise_counts.loc[left_feature, right_feature] = pairwise_n
            pairwise_counts.loc[right_feature, left_feature] = pairwise_n

            pearson_r = np.nan
            if pairwise_n >= 2:
                left_complete = left.loc[valid_mask]
                right_complete = right.loc[valid_mask]
                if (
                    left_complete.nunique(dropna=True) > 1
                    and right_complete.nunique(dropna=True) > 1
                ):
                    if left_feature == right_feature:
                        pearson_r = 1.0
                    else:
                        pearson_r = float(left_complete.corr(right_complete))

            correlation_matrix.loc[left_feature, right_feature] = pearson_r
            correlation_matrix.loc[right_feature, left_feature] = pearson_r

    return correlation_matrix, pairwise_counts


def build_correlation_pairs_table(
    correlation_matrix: pd.DataFrame,
    pairwise_counts: pd.DataFrame,
    scenario: str,
    feature_group_mapping: Mapping[str, str],
    threshold: float,
) -> pd.DataFrame:
    """Create one record per unique feature pair and classify its correlation."""
    features = list(correlation_matrix.columns)
    records = []
    for left_index, feature_1 in enumerate(features):
        for feature_2 in features[left_index + 1 :]:
            pearson_r = correlation_matrix.loc[feature_1, feature_2]
            absolute_r = abs(float(pearson_r)) if pd.notna(pearson_r) else np.nan
            if pd.isna(pearson_r):
                direction = "undefined"
            elif pearson_r > 0:
                direction = "positive"
            elif pearson_r < 0:
                direction = "negative"
            else:
                direction = "zero"

            group_1 = feature_group_mapping[feature_1]
            group_2 = feature_group_mapping[feature_2]
            records.append(
                {
                    "scenario": scenario,
                    "feature_1": feature_1,
                    "group_1": group_1,
                    "feature_2": feature_2,
                    "group_2": group_2,
                    "pearson_r": pearson_r,
                    "absolute_r": absolute_r,
                    "pairwise_n": int(pairwise_counts.loc[feature_1, feature_2]),
                    "same_group": group_1 == group_2,
                    "correlation_direction": direction,
                    "threshold_exceeded": bool(
                        pd.notna(absolute_r) and absolute_r > threshold
                    ),
                }
            )

    pairs = pd.DataFrame(records, columns=CORRELATION_PAIR_COLUMNS)
    if not pairs.empty:
        pairs = pairs.sort_values(
            "absolute_r", ascending=False, na_position="last", kind="mergesort"
        ).reset_index(drop=True)
    return pairs


def save_correlation_outputs(
    scenario_output_directory: Path,
    feature_summary: pd.DataFrame,
    quality_warnings: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    all_pairs: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Save all requested scenario CSV files and return split pair tables."""
    scenario_output_directory = Path(scenario_output_directory)
    scenario_output_directory.mkdir(parents=True, exist_ok=True)

    high_mask = all_pairs["threshold_exceeded"].fillna(False).astype(bool)
    high_pairs = all_pairs.loc[high_mask].copy().reset_index(drop=True)
    within_high_pairs = high_pairs.loc[
        high_pairs["same_group"].astype(bool)
    ].copy().reset_index(drop=True)
    cross_high_pairs = high_pairs.loc[
        ~high_pairs["same_group"].astype(bool)
    ].copy().reset_index(drop=True)

    feature_summary.to_csv(
        scenario_output_directory / "feature_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality_warnings.to_csv(
        scenario_output_directory / "data_quality_warnings.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlation_matrix.to_csv(
        scenario_output_directory / "pearson_correlation_matrix.csv",
        index=True,
        encoding="utf-8-sig",
    )
    all_pairs.to_csv(
        scenario_output_directory / "all_correlation_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    high_pairs.to_csv(
        scenario_output_directory / "high_correlation_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    within_high_pairs.to_csv(
        scenario_output_directory / "high_correlation_pairs_within_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_high_pairs.to_csv(
        scenario_output_directory / "high_correlation_pairs_cross_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "high_pairs": high_pairs,
        "within_high_pairs": within_high_pairs,
        "cross_high_pairs": cross_high_pairs,
    }


def _draw_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    mask: Optional[np.ndarray] = None,
) -> None:
    """Draw an annotated heatmap with seaborn or a matplotlib fallback."""
    feature_count = len(correlation_matrix)
    figure_size = max(8.0, feature_count * 1.25)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    values = correlation_matrix.to_numpy(dtype=float)
    effective_mask = np.isnan(values)
    if mask is not None:
        effective_mask = effective_mask | np.asarray(mask, dtype=bool)

    try:
        import seaborn as sns

        sns.heatmap(
            correlation_matrix,
            mask=effective_mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Pearson r", "shrink": 0.8},
            ax=axis,
        )
    except ImportError:
        masked_values = np.ma.array(values, mask=effective_mask)
        image = axis.imshow(masked_values, cmap="RdBu_r", vmin=-1, vmax=1)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.8)
        colorbar.set_label("Pearson r")
        axis.set_xticks(np.arange(feature_count))
        axis.set_yticks(np.arange(feature_count))
        axis.set_xticklabels(correlation_matrix.columns)
        axis.set_yticklabels(correlation_matrix.index)
        for row_index in range(feature_count):
            for column_index in range(feature_count):
                if not effective_mask[row_index, column_index]:
                    value = values[row_index, column_index]
                    text_color = "white" if abs(value) >= 0.55 else "black"
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                    )

    axis.set_title(title, pad=18)
    axis.set_xlabel("Candidate feature")
    axis.set_ylabel("Candidate feature")
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.tick_params(axis="y", labelrotation=0)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG stores resolution as an integer number of pixels per metre. Saving
    # at exactly 300 dpi can round-trip as 299.9994 dpi, so use 320 to remain
    # unambiguously above the required 300-dpi minimum.
    figure.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(figure)


def plot_full_heatmap(
    correlation_matrix: pd.DataFrame,
    scenario: str,
    threshold: float,
    output_path: Path,
) -> None:
    """Save the complete annotated Pearson correlation heatmap."""
    _draw_heatmap(
        correlation_matrix,
        f"{scenario} Pearson Correlation Matrix (|r| > {threshold:.2f} threshold)",
        output_path,
    )


def plot_high_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    scenario: str,
    threshold: float,
    output_path: Path,
) -> None:
    """Save a heatmap masking the diagonal and correlations not above threshold."""
    values = correlation_matrix.to_numpy(dtype=float)
    mask = np.isnan(values) | (np.abs(values) <= threshold)
    np.fill_diagonal(mask, True)
    _draw_heatmap(
        correlation_matrix,
        f"{scenario} High Pearson Correlations (|r| > {threshold:.2f})",
        output_path,
        mask=mask,
    )


def create_combined_summary(
    scenario_results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Aggregate PF and TF correlation counts and strongest pairs."""
    records = []
    for scenario, result in scenario_results.items():
        all_pairs = result["all_pairs"]
        finite_pairs = all_pairs.loc[all_pairs["absolute_r"].notna()]
        if finite_pairs.empty:
            maximum_absolute_correlation = np.nan
            max_feature_1 = None
            max_feature_2 = None
            max_pearson_r = np.nan
        else:
            max_pair = finite_pairs.sort_values(
                "absolute_r", ascending=False, kind="mergesort"
            ).iloc[0]
            maximum_absolute_correlation = max_pair["absolute_r"]
            max_feature_1 = max_pair["feature_1"]
            max_feature_2 = max_pair["feature_2"]
            max_pearson_r = max_pair["pearson_r"]

        records.append(
            {
                "scenario": scenario,
                "number_of_features": len(result["features"]),
                "number_of_all_pairs": len(all_pairs),
                "number_of_high_correlation_pairs": len(result["high_pairs"]),
                "number_of_within_group_high_pairs": len(
                    result["within_high_pairs"]
                ),
                "number_of_cross_group_high_pairs": len(result["cross_high_pairs"]),
                "maximum_absolute_correlation": maximum_absolute_correlation,
                "feature_1_of_max_pair": max_feature_1,
                "feature_2_of_max_pair": max_feature_2,
                "pearson_r_of_max_pair": max_pearson_r,
            }
        )

    return pd.DataFrame(records, columns=COMBINED_SUMMARY_COLUMNS)


def _format_pair_lines(pairs: pd.DataFrame) -> list:
    """Format correlation-pair records for the plain-text report."""
    if pairs.empty:
        return ["  None"]
    return [
        (
            f"  - {row.feature_1} ({row.group_1}) <-> "
            f"{row.feature_2} ({row.group_2}): "
            f"r={row.pearson_r:.4f}, |r|={row.absolute_r:.4f}, "
            f"pairwise_n={int(row.pairwise_n)}"
        )
        for row in pairs.itertuples(index=False)
    ]


def write_text_report(
    scenario_results: Mapping[str, Mapping[str, Any]],
    scenario_feature_groups: Mapping[str, Mapping[str, Sequence[str]]],
    output_path: Path,
    threshold: float,
) -> None:
    """Write a human-readable PF/TF correlation analysis report."""
    lines = [
        "Pearson Correlation Analysis Report",
        "=" * 35,
        f"High-correlation rule: absolute Pearson r > {threshold:.2f}",
        "Pairwise complete finite observations were used; no imputation was applied.",
        "Correlation does not imply causation.",
        "",
    ]

    for scenario, result in scenario_results.items():
        lines.extend([scenario, "-" * len(scenario)])
        lines.append(f"Features analyzed: {len(result['features'])}")
        lines.append("Theoretical groups:")
        for theoretical_group, features in scenario_feature_groups[scenario].items():
            lines.append(f"  - {theoretical_group}: {', '.join(features)}")

        feature_summary = result["feature_summary"]
        missing_features = feature_summary.loc[
            feature_summary["missing_count"] > 0
        ]
        lines.append("Missing observations by feature:")
        if missing_features.empty:
            lines.append("  None.")
        else:
            for feature_row in missing_features.itertuples(index=False):
                lines.append(
                    f"  - {feature_row.feature}: {int(feature_row.missing_count)} "
                    f"missing ({feature_row.missing_percentage:.2f}%)"
                )

        warnings = result["quality_warnings"]
        lines.append("Data-quality issues:")
        if warnings.empty:
            lines.append("  None detected by the configured checks.")
        else:
            for warning in warnings.itertuples(index=False):
                lines.append(
                    f"  - {warning.feature} [{warning.warning_type}]: "
                    f"{warning.details}"
                )

        lines.append(
            f"High-correlation pairs: {len(result['high_pairs'])}"
        )
        lines.append(
            f"Within-group high-correlation pairs: {len(result['within_high_pairs'])}"
        )
        lines.extend(_format_pair_lines(result["within_high_pairs"]))
        lines.append(
            f"Cross-group high-correlation pairs: {len(result['cross_high_pairs'])}"
        )
        lines.extend(_format_pair_lines(result["cross_high_pairs"]))

        finite_pairs = result["all_pairs"].loc[
            result["all_pairs"]["absolute_r"].notna()
        ]
        lines.append("Highest absolute correlation pair:")
        if finite_pairs.empty:
            lines.append("  Undefined because no pair has a finite Pearson coefficient.")
        else:
            max_pair = finite_pairs.sort_values(
                "absolute_r", ascending=False, kind="mergesort"
            ).iloc[0]
            lines.extend(_format_pair_lines(max_pair.to_frame().T))
        lines.append("")

    lines.extend(
        [
            "Researcher reminder",
            "-------------------",
            "Do not automatically remove variables based only on correlation.",
            "Cross-group pairs may represent distinct theoretical dimensions even when",
            "their linear correlation is high. Feature removal requires manual review",
            "of theoretical meaning and the intended clustering interpretation.",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_scenario(
    data: pd.DataFrame,
    scenario: str,
    feature_groups: Mapping[str, Sequence[str]],
    scenario_output_directory: Path,
    threshold: float,
    expected_ranges: Mapping[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """Run all quality, correlation, output, and chart steps for one scenario."""
    features = _flatten_features(feature_groups)
    feature_mapping = build_feature_group_mapping(feature_groups)
    feature_summary = create_feature_summary(data, scenario, feature_groups)
    quality_warnings = check_data_quality(
        data, scenario, feature_groups, expected_ranges
    )
    correlation_matrix, pairwise_counts = calculate_pairwise_correlation(
        data, features
    )
    all_pairs = build_correlation_pairs_table(
        correlation_matrix,
        pairwise_counts,
        scenario,
        feature_mapping,
        threshold,
    )
    split_pairs = save_correlation_outputs(
        scenario_output_directory,
        feature_summary,
        quality_warnings,
        correlation_matrix,
        all_pairs,
    )
    plot_full_heatmap(
        correlation_matrix,
        scenario,
        threshold,
        Path(scenario_output_directory) / "pearson_correlation_heatmap.png",
    )
    plot_high_correlation_heatmap(
        correlation_matrix,
        scenario,
        threshold,
        Path(scenario_output_directory) / "high_correlation_heatmap.png",
    )

    return {
        "features": features,
        "feature_summary": feature_summary,
        "quality_warnings": quality_warnings,
        "correlation_matrix": correlation_matrix,
        "pairwise_counts": pairwise_counts,
        "all_pairs": all_pairs,
        **split_pairs,
    }


def main() -> None:
    """Run PF and TF Pearson correlation analyses and save all outputs."""
    print("Loading dataset...")
    data = load_data(INPUT_FILE)
    print(f"Dataset shape: {len(data)} rows × {len(data.columns)} columns")
    validate_columns(data, SCENARIO_FEATURE_GROUPS)

    scenario_results: Dict[str, Mapping[str, Any]] = {}
    for scenario, feature_groups in SCENARIO_FEATURE_GROUPS.items():
        print(f"\nRunning {scenario} correlation analysis...")
        print(f"{scenario} features: {len(_flatten_features(feature_groups))}")
        result = analyze_scenario(
            data,
            scenario,
            feature_groups,
            OUTPUT_DIRECTORY / scenario,
            CORRELATION_THRESHOLD,
            EXPECTED_RANGES,
        )
        scenario_results[scenario] = result
        print(f"{scenario} high-correlation pairs: {len(result['high_pairs'])}")
        print(
            f"{scenario} within-group high-correlation pairs: "
            f"{len(result['within_high_pairs'])}"
        )
        print(
            f"{scenario} cross-group high-correlation pairs: "
            f"{len(result['cross_high_pairs'])}"
        )

    combined_output_directory = OUTPUT_DIRECTORY / "combined_summary"
    combined_output_directory.mkdir(parents=True, exist_ok=True)
    combined_summary = create_combined_summary(scenario_results)
    combined_summary.to_csv(
        combined_output_directory / "correlation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_text_report(
        scenario_results,
        SCENARIO_FEATURE_GROUPS,
        combined_output_directory / "correlation_report.txt",
        CORRELATION_THRESHOLD,
    )

    print("\nAnalysis completed.")
    print("Outputs saved to: outputs_correlation/")
    print("Suggested next step: 人工审查高相关变量对并确定最终特征")


if __name__ == "__main__":
    main()
