from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.cluster_preparation import pca_analysis as pca


class PcaAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pf_data = pd.DataFrame(
            {
                "fid": [1, 2, 3, 4, 5],
                "MS_ID": ["s1", "s2", "s3", "s4", "s5"],
                "PF_Index_Risk_equal": [0.10, 0.20, 0.30, 0.40, 0.50],
                "Per_extent": [0.0, 0.2, 0.4, 0.6, 0.8],
                "N_PFlossR_2kiw": [0.0, -0.1, np.nan, -0.3, -0.4],
                "N_PFlossR_2kmw": [0.1, 0.0, -0.1, -0.2, -0.3],
                "PFResident_lossR": [0.0, -0.2, -0.3, np.nan, -0.5],
                "PFDaynight_lossR": [0.0, -0.1, -0.2, -0.4, -0.6],
                "PFAB5k_NOR": [0.1, 0.3, 0.2, 0.8, 0.7],
                "PFABC_NOR": [0.2, 0.1, 0.4, 0.6, 0.9],
                "TFAB5k_NOR": [99, 98, 97, 96, 95],
                "unrelated_numeric": [10, 11, 12, 13, 14],
            }
        )

    def test_load_data_uses_strict_pf_whitelist_and_counts_filled_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "pf.csv"
            original = self.pf_data.copy(deep=True)
            original.to_csv(input_path, index=False)

            loaded = pca.load_data(input_path, "PF")
            reread = pd.read_csv(input_path)

        self.assertEqual(loaded.scenario, "PF")
        self.assertEqual(list(loaded.features.columns), list(pca.PF_FEATURES))
        self.assertEqual(list(loaded.metadata.columns), list(pca.PF_METADATA_COLUMNS))
        self.assertEqual(int(loaded.fill_counts["N_PFlossR_2kiw"]), 1)
        self.assertEqual(int(loaded.fill_counts["PFResident_lossR"]), 1)
        self.assertFalse(loaded.features.isna().any().any())
        pd.testing.assert_frame_equal(reread, original)

    def test_load_data_uses_strict_tf_whitelist_and_metadata(self) -> None:
        tf_data = pd.DataFrame(
            {
                "fid": [1, 2, 3],
                "MS_ID": ["t1", "t2", "t3"],
                "TF_Index_Risk_equal": [0.25, 0.50, 0.75],
                "Tem_extent": [0.1, 0.2, 0.3],
                "N_TFlossR_2kiw": [np.nan, -0.2, -0.3],
                "N_TFlossR_2kmw": [-0.1, -0.2, -0.4],
                "TFResident_lossR": [-0.2, -0.3, -0.4],
                "TFDaynight_lossR": [-0.1, np.nan, -0.5],
                "TFAB5k_NOR": [0.2, 0.4, 0.6],
                "TFABC_NOR": [0.1, 0.3, 0.5],
                "PFAB5k_NOR": [99, 98, 97],
                "unrelated_numeric": [10, 11, 12],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "tf.csv"
            tf_data.to_csv(input_path, index=False)

            loaded = pca.load_data(input_path, "TF")

        self.assertEqual(loaded.scenario, "TF")
        self.assertEqual(list(loaded.features.columns), list(pca.TF_FEATURES))
        self.assertEqual(list(loaded.metadata.columns), list(pca.TF_METADATA_COLUMNS))
        self.assertEqual(int(loaded.fill_counts["N_TFlossR_2kiw"]), 1)
        self.assertEqual(int(loaded.fill_counts["TFDaynight_lossR"]), 1)
        self.assertFalse(loaded.features.isna().any().any())

    def test_load_data_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "bad.csv"
            self.pf_data.drop(columns=["PFABC_NOR"]).to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "PFABC_NOR"):
                pca.load_data(input_path, "PF")

    def test_standardize_centers_features(self) -> None:
        loaded = pca.PreparedDataset(
            scenario="PF",
            metadata=self.pf_data[list(pca.PF_METADATA_COLUMNS)].copy(),
            features=self.pf_data[list(pca.PF_FEATURES)].fillna(0).copy(),
            fill_counts=pd.Series(0, index=pca.PF_FEATURES),
        )

        scaled = pca.standardize(loaded.features)

        self.assertEqual(scaled.shape, (5, 7))
        np.testing.assert_allclose(scaled.mean(axis=0), np.zeros(7), atol=1e-12)

    def test_run_pca_returns_full_scores_loadings_summary_and_two_dimensional_scores(self) -> None:
        loaded = pca.PreparedDataset(
            scenario="PF",
            metadata=self.pf_data[list(pca.PF_METADATA_COLUMNS)].copy(),
            features=self.pf_data[list(pca.PF_FEATURES)].fillna(0).copy(),
            fill_counts=pd.Series(0, index=pca.PF_FEATURES),
        )
        scaled = pca.standardize(loaded.features)

        result = pca.run_pca(scaled, pca.PF_FEATURES)

        self.assertEqual(result.summary.shape[0], 5)
        self.assertEqual(result.loadings.shape, (7, 6))
        self.assertEqual(result.full_scores.shape, (5, 5))
        self.assertEqual(result.two_d_scores.shape, (5, 2))
        self.assertIn("PC1", result.loadings.columns)
        self.assertIn("Cumulative Explained Variance", result.summary.columns)
        np.testing.assert_allclose(
            result.summary["Explained Variance Ratio"].sum(), 1.0, atol=1e-12
        )

    def test_save_tables_writes_expected_metadata_and_pca_tables(self) -> None:
        loaded = pca.PreparedDataset(
            scenario="PF",
            metadata=self.pf_data[list(pca.PF_METADATA_COLUMNS)].copy(),
            features=self.pf_data[list(pca.PF_FEATURES)].fillna(0).copy(),
            fill_counts=pd.Series(0, index=pca.PF_FEATURES),
        )
        result = pca.run_pca(pca.standardize(loaded.features), pca.PF_FEATURES)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            pca.save_tables(loaded, result, output_dir)
            scores = pd.read_csv(output_dir / "pca_scores.csv")
            loadings = pd.read_csv(output_dir / "pca_loadings.csv")
            summary = pd.read_csv(output_dir / "pca_summary.csv")

        self.assertEqual(scores.columns[:3].tolist(), list(pca.PF_METADATA_COLUMNS))
        self.assertIn("PC1", scores.columns)
        self.assertEqual(loadings.columns[0], "Variable")
        self.assertEqual(summary.columns[0], "Component")

    def test_plots_and_report_are_generated(self) -> None:
        loaded = pca.PreparedDataset(
            scenario="PF",
            metadata=self.pf_data[list(pca.PF_METADATA_COLUMNS)].copy(),
            features=self.pf_data[list(pca.PF_FEATURES)].fillna(0).copy(),
            fill_counts=pd.Series({"Per_extent": 0, "N_PFlossR_2kiw": 1,
                                   "N_PFlossR_2kmw": 0, "PFResident_lossR": 1,
                                   "PFDaynight_lossR": 0, "PFAB5k_NOR": 0,
                                   "PFABC_NOR": 0}),
        )
        result = pca.run_pca(pca.standardize(loaded.features), pca.PF_FEATURES)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            pca.plot_scree(result.summary, output_dir / "explained_variance.png")
            pca.plot_cumulative_variance(
                result.summary, output_dir / "cumulative_variance.png"
            )
            pca.plot_scatter(result, output_dir / "pca_scatter.png")
            pca.plot_loading(result, output_dir / "loading_plot.png")
            pca.plot_correlation_circle(
                result, output_dir / "correlation_circle.png"
            )
            pca.generate_report(loaded, result, output_dir / "pca_report.txt")

            expected = {
                "explained_variance.png",
                "cumulative_variance.png",
                "pca_scatter.png",
                "loading_plot.png",
                "correlation_circle.png",
                "pca_report.txt",
            }
            actual = {path.name for path in output_dir.iterdir()}
            report = (output_dir / "pca_report.txt").read_text(encoding="utf-8")

        self.assertEqual(actual, expected)
        self.assertIn("Total observations: 5", report)
        self.assertIn("Variables: 7", report)
        self.assertIn("Fill-zero counts by PCA variable:", report)

    def test_run_analysis_writes_lowercase_scenario_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "pf.csv"
            output_root = root / "output_pca"
            self.pf_data.to_csv(input_path, index=False)

            output_dir = pca.run_analysis(input_path, "PF", output_root)

        self.assertEqual(output_dir.name, "pf")

    def test_default_output_root_is_singular_output_pca(self) -> None:
        self.assertEqual(pca.DEFAULT_OUTPUT_ROOT.name, "output_pca")


if __name__ == "__main__":
    unittest.main()
