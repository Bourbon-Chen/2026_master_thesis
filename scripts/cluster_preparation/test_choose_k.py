from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.cluster_preparation import choose_k


class ChooseKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pf_data = pd.DataFrame(
            {
                "fid": range(1, 13),
                "MS_ID": [f"s{i}" for i in range(1, 13)],
                "PF_Index_Risk_equal": np.linspace(0.1, 0.9, 12),
                "Per_extent": [0, 0.1, 0.2, 0.1, 5, 5.1, 5.2, 5.1, 10, 10.1, 10.2, 10.1],
                "N_PFlossR_2kiw": [0, 0.1, np.nan, 0.2, 5, 5.2, 5.1, 5.2, 10, 10.3, 10.1, 10.2],
                "N_PFlossR_2kmw": [0.1, 0, 0.2, 0.1, 4.9, 5.1, 5.2, 5, 10.1, 10, 10.2, 10.3],
                "PFResident_lossR": [0, -0.1, -0.2, -0.1, -5, -5.1, -5.2, -5.1, -10, -10.1, -10.2, -10.1],
                "PFDaynight_lossR": [0, -0.2, -0.1, np.nan, -5, -5.2, -5.1, -5.2, -10, -10.2, -10.1, -10.3],
                "PFAB5k_NOR": [0, 0.1, 0.2, 0.1, 4.8, 5, 5.1, 5.2, 9.9, 10, 10.2, 10.1],
                "PFABC_NOR": [0.2, 0.1, 0, 0.1, 5.1, 5, 4.9, 5.2, 10.2, 10.1, 10, 10.3],
                "unrelated_numeric": np.arange(12),
                "TFAB5k_NOR": np.arange(100, 112),
            }
        )

    def test_load_data_uses_strict_pf_whitelist_and_fill_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "pf.csv"
            original = self.pf_data.copy(deep=True)
            original.to_csv(input_path, index=False)

            loaded = choose_k.load_data(input_path, "PF")
            reread = pd.read_csv(input_path)

        self.assertEqual(loaded.scenario, "PF")
        self.assertEqual(list(loaded.features.columns), list(choose_k.PF_FEATURES))
        self.assertEqual(list(loaded.metadata.columns), list(choose_k.PF_METADATA_COLUMNS))
        self.assertFalse(loaded.features.isna().any().any())
        self.assertEqual(int(loaded.fill_counts["N_PFlossR_2kiw"]), 1)
        pd.testing.assert_frame_equal(reread, original)

    def test_load_data_uses_strict_tf_whitelist(self) -> None:
        tf_data = pd.DataFrame(
            {
                "fid": [1, 2, 3, 4],
                "MS_ID": ["t1", "t2", "t3", "t4"],
                "TF_Index_Risk_equal": [0.1, 0.2, 0.3, 0.4],
                "Tem_extent": [0.1, 0.2, 5.0, 5.2],
                "N_TFlossR_2kiw": [0.0, np.nan, 5.1, 5.2],
                "N_TFlossR_2kmw": [0.2, 0.1, 5.0, 5.1],
                "TFResident_lossR": [-0.1, -0.2, -5.0, -5.1],
                "TFDaynight_lossR": [-0.2, -0.1, -5.1, -5.2],
                "TFAB5k_NOR": [0.2, 0.1, 5.0, 5.2],
                "TFABC_NOR": [0.1, 0.2, 5.1, 5.0],
                "PFAB5k_NOR": [99, 98, 97, 96],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "tf.csv"
            tf_data.to_csv(input_path, index=False)

            loaded = choose_k.load_data(input_path, "TF")

        self.assertEqual(list(loaded.features.columns), list(choose_k.TF_FEATURES))
        self.assertEqual(list(loaded.metadata.columns), list(choose_k.TF_METADATA_COLUMNS))
        self.assertEqual(int(loaded.fill_counts["N_TFlossR_2kiw"]), 1)

    def test_load_data_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "bad.csv"
            self.pf_data.drop(columns=["PFABC_NOR"]).to_csv(input_path, index=False)

            with self.assertRaisesRegex(ValueError, "PFABC_NOR"):
                choose_k.load_data(input_path, "PF")

    def test_standardize_features_centers_values(self) -> None:
        loaded = choose_k.PreparedDataset(
            scenario="PF",
            metadata=self.pf_data[list(choose_k.PF_METADATA_COLUMNS)].copy(),
            features=self.pf_data[list(choose_k.PF_FEATURES)].fillna(0).copy(),
            fill_counts=pd.Series(0, index=choose_k.PF_FEATURES),
        )

        scaled = choose_k.standardize_features(loaded.features)

        self.assertEqual(scaled.shape, (12, 7))
        np.testing.assert_allclose(scaled.mean(axis=0), np.zeros(7), atol=1e-12)

    def test_evaluate_k_returns_metrics_without_cluster_labels(self) -> None:
        loaded = choose_k.PreparedDataset(
            scenario="PF",
            metadata=self.pf_data[list(choose_k.PF_METADATA_COLUMNS)].copy(),
            features=self.pf_data[list(choose_k.PF_FEATURES)].fillna(0).copy(),
            fill_counts=pd.Series(0, index=choose_k.PF_FEATURES),
        )
        values = choose_k.standardize_features(loaded.features)

        result = choose_k.evaluate_k(values, range(2, 5))

        self.assertEqual(result["K"].tolist(), [2, 3, 4])
        self.assertEqual(
            result.columns.tolist(),
            [
                "K",
                "Inertia",
                "Silhouette",
                "Calinski_Harabasz",
                "Davies_Bouldin",
                "Largest_Cluster",
                "Smallest_Cluster",
                "Cluster_Size_SD",
            ],
        )
        self.assertNotIn("Cluster_Label", result.columns)
        self.assertTrue(np.isfinite(result["Silhouette"]).all())

    def test_default_silhouette_sample_size_is_configured(self) -> None:
        self.assertEqual(choose_k.SILHOUETTE_SAMPLE_SIZE, 10000)

    def test_plots_results_and_recommendation_are_saved(self) -> None:
        values = choose_k.standardize_features(
            self.pf_data[list(choose_k.PF_FEATURES)].fillna(0)
        )
        result = choose_k.evaluate_k(values, range(2, 5))

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            choose_k.save_results(result, output_dir)
            recommendation = choose_k.recommend_best_k(result, output_dir / "recommended_k.txt")
            choose_k.plot_elbow(result, output_dir / "elbow_plot.png")
            choose_k.plot_silhouette(result, output_dir / "silhouette_score.png")
            choose_k.plot_calinski(result, output_dir / "calinski_harabasz.png")
            choose_k.plot_davies(result, output_dir / "davies_bouldin.png")

            actual = {path.name for path in output_dir.iterdir()}
            report = (output_dir / "recommended_k.txt").read_text(encoding="utf-8")

        self.assertEqual(
            actual,
            {
                "k_evaluation.csv",
                "recommended_k.txt",
                "elbow_plot.png",
                "silhouette_score.png",
                "calinski_harabasz.png",
                "davies_bouldin.png",
            },
        )
        self.assertIn("Recommended K:", report)
        self.assertIn(str(recommendation), report)

    def test_run_analysis_writes_outputs_choose_k_uppercase_scenario_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "pf.csv"
            output_root = root / "outputs_choose_k"
            self.pf_data.to_csv(input_path, index=False)

            output_dir = choose_k.run_analysis(input_path, "PF", output_root, range(2, 4))

        self.assertEqual(output_dir, output_root / "PF")


if __name__ == "__main__":
    unittest.main()
