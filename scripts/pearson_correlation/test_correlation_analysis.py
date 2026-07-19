"""Tests for the standalone Pearson correlation analysis script."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.pearson_correlation import correlation_analysis as ca


class CorrelationAnalysisTests(unittest.TestCase):
    """Verify correlation, quality, reporting, and plotting behavior."""

    def test_load_data_and_validate_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "sample.csv"
            pd.DataFrame({"fid": [1], "MS_ID": [10], "a": [0.5]}).to_csv(
                csv_path, index=False
            )
            loaded = ca.load_data(csv_path)

        self.assertEqual(loaded.shape, (1, 3))
        ca.validate_columns(
            loaded,
            {"PF": {"group": ["a"]}},
            identifier_columns=("fid", "MS_ID"),
        )
        with self.assertRaisesRegex(ValueError, "missing_feature"):
            ca.validate_columns(
                loaded,
                {"PF": {"group": ["missing_feature"]}},
                identifier_columns=("fid", "MS_ID"),
            )

    def test_build_feature_group_mapping(self) -> None:
        mapping = ca.build_feature_group_mapping(
            {"exposure": ["flood"], "loss": ["loss_a", "loss_b"]}
        )
        self.assertEqual(
            mapping,
            {"flood": "exposure", "loss_a": "loss", "loss_b": "loss"},
        )

    def test_create_feature_summary_has_required_statistics(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "y"]})
        summary = ca.create_feature_summary(
            frame, "PF", {"numeric": ["a"], "text": ["b"]}
        )

        expected_columns = [
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
        self.assertEqual(summary.columns.tolist(), expected_columns)
        numeric = summary.set_index("feature").loc["a"]
        self.assertEqual(numeric["count"], 2)
        self.assertEqual(numeric["missing_count"], 1)
        self.assertAlmostEqual(numeric["missing_percentage"], 100 / 3)
        self.assertAlmostEqual(numeric["median"], 2.0)
        self.assertTrue(np.isnan(summary.set_index("feature").loc["b", "mean"]))

    def test_check_data_quality_detects_every_required_problem(self) -> None:
        frame = pd.DataFrame(
            {
                "empty": [np.nan, np.nan, np.nan],
                "constant": [1.0, 1.0, np.nan],
                "text": ["1", "bad", None],
                "invalid_numeric": [np.inf, -np.inf, 2.0],
            }
        )
        groups = {
            "g": ["empty", "constant", "text", "invalid_numeric"],
        }
        warnings = ca.check_data_quality(
            frame, "PF", groups, expected_ranges={"g": (0.0, 1.0)}
        )

        warning_types = set(warnings["warning_type"])
        self.assertTrue(
            {
                "all_missing",
                "constant_variable",
                "non_numeric",
                "positive_infinity",
                "negative_infinity",
                "out_of_expected_range",
            }.issubset(warning_types)
        )
        out_of_range = warnings.loc[
            warnings["warning_type"] == "out_of_expected_range"
        ].iloc[0]
        self.assertEqual(out_of_range["feature"], "invalid_numeric")
        self.assertEqual(out_of_range["affected_count"], 1)

    def test_calculate_pairwise_correlation_uses_pairwise_finite_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0],
                "b": [2.0, 4.0, 6.0, np.nan],
                "constant": [1.0, 1.0, 1.0, 1.0],
            }
        )
        matrix, counts = ca.calculate_pairwise_correlation(
            frame, ["a", "b", "constant"]
        )

        self.assertEqual(counts.loc["a", "b"], 2)
        self.assertAlmostEqual(matrix.loc["a", "b"], 1.0)
        self.assertEqual(counts.loc["a", "constant"], 3)
        self.assertTrue(np.isnan(matrix.loc["a", "constant"]))
        self.assertTrue(np.isnan(matrix.loc["constant", "constant"]))

    def test_build_pairs_is_unique_sorted_and_uses_strict_threshold(self) -> None:
        features = ["a", "b", "c"]
        matrix = pd.DataFrame(
            [[1.0, 0.8, -0.6], [0.8, 1.0, -0.9], [-0.6, -0.9, 1.0]],
            index=features,
            columns=features,
        )
        counts = pd.DataFrame(10, index=features, columns=features)
        pairs = ca.build_correlation_pairs_table(
            matrix,
            counts,
            "PF",
            {"a": "g1", "b": "g1", "c": "g2"},
            threshold=0.6,
        )

        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs.iloc[0]["feature_1"], "b")
        self.assertEqual(pairs.iloc[0]["feature_2"], "c")
        self.assertEqual(int(pairs["threshold_exceeded"].sum()), 2)
        exact_threshold = pairs.loc[pairs["absolute_r"] == 0.6].iloc[0]
        self.assertFalse(bool(exact_threshold["threshold_exceeded"]))
        self.assertFalse(bool(exact_threshold["same_group"]))
        self.assertEqual(exact_threshold["correlation_direction"], "negative")

    def test_save_outputs_creates_all_scenario_csv_files(self) -> None:
        summary = pd.DataFrame({"feature": ["a"]})
        warnings = pd.DataFrame(columns=ca.DATA_QUALITY_WARNING_COLUMNS)
        matrix = pd.DataFrame([[1.0]], index=["a"], columns=["a"])
        pairs = pd.DataFrame(columns=ca.CORRELATION_PAIR_COLUMNS)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            split = ca.save_correlation_outputs(
                output_directory, summary, warnings, matrix, pairs
            )
            expected_files = {
                "feature_summary.csv",
                "data_quality_warnings.csv",
                "pearson_correlation_matrix.csv",
                "all_correlation_pairs.csv",
                "high_correlation_pairs.csv",
                "high_correlation_pairs_within_group.csv",
                "high_correlation_pairs_cross_group.csv",
            }
            actual_files = {path.name for path in output_directory.glob("*.csv")}

        self.assertEqual(actual_files, expected_files)
        self.assertEqual(len(split["high_pairs"]), 0)

    def test_plots_create_nonempty_png_files(self) -> None:
        matrix = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]], index=["a", "b"], columns=["a", "b"]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            full_path = output_directory / "full.png"
            high_path = output_directory / "high.png"
            ca.plot_full_heatmap(matrix, "PF", 0.6, full_path)
            ca.plot_high_correlation_heatmap(matrix, "PF", 0.6, high_path)
            full_size = full_path.stat().st_size
            high_size = high_path.stat().st_size
            with Image.open(full_path) as image:
                full_dpi = image.info["dpi"]
            with Image.open(high_path) as image:
                high_dpi = image.info["dpi"]

        self.assertGreater(full_size, 0)
        self.assertGreater(high_size, 0)
        self.assertGreaterEqual(full_dpi[0], 300)
        self.assertGreaterEqual(full_dpi[1], 300)
        self.assertGreaterEqual(high_dpi[0], 300)
        self.assertGreaterEqual(high_dpi[1], 300)

    def test_create_combined_summary_and_write_report(self) -> None:
        pairs = pd.DataFrame(
            [
                {
                    "scenario": "PF",
                    "feature_1": "a",
                    "group_1": "g1",
                    "feature_2": "b",
                    "group_2": "g1",
                    "pearson_r": -0.9,
                    "absolute_r": 0.9,
                    "pairwise_n": 10,
                    "same_group": True,
                    "correlation_direction": "negative",
                    "threshold_exceeded": True,
                }
            ],
            columns=ca.CORRELATION_PAIR_COLUMNS,
        )
        results = {
            "PF": {
                "features": ["a", "b"],
                "feature_summary": pd.DataFrame(
                    [
                        {
                            "feature": "a",
                            "missing_count": 1,
                            "missing_percentage": 10.0,
                        },
                        {
                            "feature": "b",
                            "missing_count": 0,
                            "missing_percentage": 0.0,
                        },
                    ]
                ),
                "quality_warnings": pd.DataFrame(
                    columns=ca.DATA_QUALITY_WARNING_COLUMNS
                ),
                "all_pairs": pairs,
                "high_pairs": pairs.copy(),
                "within_high_pairs": pairs.copy(),
                "cross_high_pairs": pairs.iloc[0:0].copy(),
            }
        }
        summary = ca.create_combined_summary(results)

        self.assertEqual(summary.loc[0, "number_of_features"], 2)
        self.assertEqual(summary.loc[0, "number_of_high_correlation_pairs"], 1)
        self.assertEqual(summary.loc[0, "feature_1_of_max_pair"], "a")
        self.assertEqual(summary.loc[0, "pearson_r_of_max_pair"], -0.9)

        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.txt"
            ca.write_text_report(
                results, {"PF": {"g1": ["a", "b"]}}, report_path, 0.6
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("PF", report)
        self.assertIn("a: 1 missing (10.00%)", report)
        self.assertIn("Correlation does not imply causation", report)
        self.assertIn("manual", report.lower())

    def test_analyze_scenario_returns_complete_result(self) -> None:
        frame = pd.DataFrame(
            {
                "a": [0.0, 0.5, 1.0],
                "b": [0.0, 0.5, 1.0],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = ca.analyze_scenario(
                frame,
                "PF",
                {"group": ["a", "b"]},
                Path(temporary_directory),
                threshold=0.6,
                expected_ranges={"group": (0.0, 1.0)},
            )
            files = {path.name for path in Path(temporary_directory).iterdir()}

        self.assertEqual(result["features"], ["a", "b"])
        self.assertEqual(len(result["high_pairs"]), 1)
        self.assertIn("pearson_correlation_heatmap.png", files)
        self.assertIn("high_correlation_heatmap.png", files)


if __name__ == "__main__":
    unittest.main()
