import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.cluster_preparation import choose_k
from scripts.utils import clustering_preprocessing as preprocessing


PLOT_FUNCTIONS = (
    "plot_elbow",
    "plot_silhouette",
    "plot_calinski",
    "plot_davies",
)


class ChooseKTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source_path = self.root / "high_risk.csv"
        self.source = pd.DataFrame(
            {
                "fid": [10, 11, 12, 13, 14],
                "MS_ID": ["s0", "s1", "s2", "s3", "s4"],
                "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6, 0.7],
                "Per_extent": [-1.0, 0.0, 1.0, np.nan, 0.5],
                "PFABC_NOR": [-0.8, -0.2, 0.2, 0.8, 1.0],
                "PFResident_lossR": [-0.4, -0.1, 0.0, 0.3, 0.7],
                "TFABC_NOR": [99.0, 98.0, 97.0, 96.0, 95.0],
            }
        )
        self.source.to_csv(self.source_path, index=False)
        prepared = preprocessing.prepare_clustering_data(self.source, "PF")
        self.artifact_dir = preprocessing.save_prepared_artifact(
            prepared=prepared,
            source_path=self.source_path,
            output_root=self.root / "prepared",
            risk_filter={
                "column": "PF_Index_Risk_equal",
                "operator": ">",
                "threshold": 0.25,
            },
        )
        self.artifact = preprocessing.load_prepared_artifact(self.artifact_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def patch_plots(self, stack: ExitStack) -> None:
        for function_name in PLOT_FUNCTIONS:
            stack.enter_context(patch.object(choose_k, function_name))

    def test_run_analysis_passes_exact_prepared_matrix_to_k_evaluation_without_scaling(
        self,
    ) -> None:
        expected = pd.DataFrame(
            {
                "K": [2, 3, 4],
                "Inertia": [3.0, 2.0, 1.0],
                "Silhouette": [0.3, 0.2, 0.1],
                "Calinski_Harabasz": [4.0, 3.0, 2.0],
                "Davies_Bouldin": [0.4, 0.5, 0.6],
                "Largest_Cluster": [3, 2, 2],
                "Smallest_Cluster": [2, 1, 1],
                "Cluster_Size_SD": [0.5, 0.4, 0.3],
            }
        )
        output_root = self.root / "outputs_choose_k"

        with ExitStack() as stack:
            self.patch_plots(stack)
            stack.enter_context(
                patch.object(
                    StandardScaler,
                    "fit",
                    side_effect=AssertionError("choose-K must not fit a scaler"),
                )
            )
            evaluate = stack.enter_context(
                patch.object(choose_k, "evaluate_k", return_value=expected)
            )
            output_dir = choose_k.run_analysis(
                prepared_path=self.artifact_dir,
                output_root=output_root,
                k_values=range(2, 5),
            )

        np.testing.assert_allclose(
            evaluate.call_args.args[0],
            self.artifact.standardized_features.to_numpy(dtype=float),
        )
        self.assertEqual(evaluate.call_args.args[1], [2, 3, 4])
        self.assertEqual(output_dir, output_root / "PF")
        reference = json.loads(
            (output_dir / "preprocessing_reference.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            reference,
            {
                "manifest_path": str(
                    (self.artifact_dir / "preprocessing_config.json").resolve()
                ),
                "manifest_sha256": preprocessing.sha256_file(
                    self.artifact_dir / "preprocessing_config.json"
                ),
                "matrix_sha256": self.artifact.config["matrix_sha256"],
                "scenario": "PF",
                "row_count": 5,
                "feature_names": list(self.artifact.feature_names),
            },
        )

    def test_evaluate_k_returns_metrics_and_candidate_cluster_sizes(self) -> None:
        result = choose_k.evaluate_k(
            self.artifact.standardized_features.to_numpy(dtype=float), range(2, 5)
        )

        self.assertEqual(result["K"].tolist(), [2, 3, 4])
        self.assertEqual(result.columns.tolist(), choose_k.K_EVALUATION_COLUMNS)
        self.assertTrue(np.isfinite(result["Silhouette"]).all())
        self.assertGreaterEqual(result["Smallest_Cluster"].min(), 1)
        self.assertGreaterEqual(
            result["Largest_Cluster"].max(), result["Smallest_Cluster"].min()
        )

    def test_run_analysis_rejects_an_existing_final_output_directory(self) -> None:
        """Reusing a scenario directory would overwrite an earlier K evaluation."""
        output_root = self.root / "outputs_choose_k"
        final_directory = output_root / "PF"
        final_directory.mkdir(parents=True)
        sentinel = final_directory / "existing.txt"
        sentinel.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            choose_k.run_analysis(self.artifact_dir, output_root, range(2, 5))

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_recommendation_reports_cluster_size_range_and_research_choice(self) -> None:
        results = pd.DataFrame(
            {
                "K": [2, 3, 4],
                "Inertia": [9.0, 6.0, 5.0],
                "Silhouette": [0.2, 0.5, 0.4],
                "Calinski_Harabasz": [10.0, 12.0, 11.0],
                "Davies_Bouldin": [1.0, 0.7, 0.8],
                "Largest_Cluster": [4, 3, 2],
                "Smallest_Cluster": [2, 1, 1],
                "Cluster_Size_SD": [1.0, 0.8, 0.5],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "recommended_k.txt"
            recommendation = choose_k.recommend_best_k(results, report_path)
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(recommendation, 3)
        self.assertIn("Minimum candidate cluster size: 1", report)
        self.assertIn("Maximum candidate cluster size: 4", report)
        self.assertIn("final K remains a documented research choice", report)

    def test_plots_and_results_are_saved(self) -> None:
        results = pd.DataFrame(
            {
                "K": [2, 3],
                "Inertia": [9.0, 6.0],
                "Silhouette": [0.2, 0.5],
                "Calinski_Harabasz": [10.0, 12.0],
                "Davies_Bouldin": [1.0, 0.7],
                "Largest_Cluster": [4, 3],
                "Smallest_Cluster": [1, 1],
                "Cluster_Size_SD": [1.0, 0.8],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            choose_k.save_results(results, output_dir)
            choose_k.recommend_best_k(results, output_dir / "recommended_k.txt")
            choose_k.plot_elbow(results, output_dir / "elbow_plot.png")
            choose_k.plot_silhouette(results, output_dir / "silhouette_score.png")
            choose_k.plot_calinski(results, output_dir / "calinski_harabasz.png")
            choose_k.plot_davies(results, output_dir / "davies_bouldin.png")

            actual = {path.name for path in output_dir.iterdir()}

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

    def test_cli_accepts_prepared_artifact_and_defaults_output_root(self) -> None:
        with patch.object(
            __import__("sys"),
            "argv",
            ["choose_k.py", "--prepared", str(self.artifact_dir)],
        ):
            args = choose_k.parse_args()

        self.assertEqual(args.prepared, self.artifact_dir)
        self.assertEqual(args.output_root, choose_k.OUTPUT_ROOT)


if __name__ == "__main__":
    unittest.main()
