import json
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.cluster_preparation import pca_analysis as pca
from scripts.utils import clustering_preprocessing as preprocessing


PLOT_FUNCTIONS = (
    "plot_scree",
    "plot_cumulative_variance",
    "plot_scatter",
    "plot_loading",
    "plot_correlation_circle",
)


class PcaAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
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
            stack.enter_context(patch.object(pca, function_name))

    def test_run_analysis_passes_exact_persisted_matrix_to_pca_without_scaling(
        self,
    ) -> None:
        output_root = self.root / "output_pca"
        with ExitStack() as stack:
            self.patch_plots(stack)
            stack.enter_context(
                patch.object(
                    StandardScaler,
                    "fit",
                    side_effect=AssertionError("PCA must not fit a scaler"),
                )
            )
            run_pca = stack.enter_context(
                patch.object(pca, "run_pca", wraps=pca.run_pca)
            )
            output_dir = pca.run_analysis(self.artifact_dir, output_root)

        actual = run_pca.call_args.args[0]
        np.testing.assert_allclose(
            actual,
            self.artifact.standardized_features.to_numpy(dtype=float),
        )
        self.assertEqual(run_pca.call_args.args[1], self.artifact.feature_names)
        self.assertEqual(output_dir, output_root / "pf")

    def test_run_analysis_preserves_tables_report_and_preprocessing_identity(
        self,
    ) -> None:
        with ExitStack() as stack:
            self.patch_plots(stack)
            output_dir = pca.run_analysis(
                self.artifact_dir / "preprocessing_config.json",
                self.root / "output_pca",
            )

        scores = pd.read_csv(output_dir / "pca_scores.csv")
        summary = pd.read_csv(output_dir / "pca_summary.csv")
        loadings = pd.read_csv(output_dir / "pca_loadings.csv")
        report = (output_dir / "pca_report.txt").read_text(encoding="utf-8")
        reference = json.loads(
            (output_dir / "preprocessing_reference.json").read_text(
                encoding="utf-8"
            )
        )

        pd.testing.assert_frame_equal(
            scores.loc[:, self.artifact.metadata.columns],
            self.artifact.metadata,
        )
        self.assertEqual(summary.columns[0], "Component")
        self.assertEqual(loadings["Variable"].tolist(), list(self.artifact.feature_names))
        self.assertIn("PC1", scores.columns)
        self.assertIn("Fill-zero counts by PCA variable:", report)
        self.assertIn("  - Per_extent: 1", report)
        self.assertIn("PCA is diagnostic", report)
        self.assertIn("K-means does not consume PCA scores", report)
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

    def test_run_analysis_rejects_an_existing_final_output_directory(self) -> None:
        """Reusing a scenario directory would overwrite an earlier PCA run."""
        output_root = self.root / "output_pca"
        final_directory = output_root / "pf"
        final_directory.mkdir(parents=True)
        sentinel = final_directory / "existing.txt"
        sentinel.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            pca.run_analysis(self.artifact_dir, output_root)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_run_pca_preserves_full_component_contract(self) -> None:
        values = self.artifact.standardized_features.to_numpy(dtype=float)

        result = pca.run_pca(values, self.artifact.feature_names)

        self.assertEqual(result.summary.shape, (3, 4))
        self.assertEqual(result.loadings.shape, (3, 4))
        self.assertEqual(result.full_scores.shape, (5, 3))
        self.assertEqual(result.two_d_scores.shape, (5, 2))
        self.assertEqual(
            result.correlation_coordinates.columns.tolist(),
            ["Variable", "PC1", "PC2"],
        )
        np.testing.assert_allclose(
            result.summary["Explained Variance Ratio"].sum(), 1.0, atol=1e-12
        )

    def test_cli_accepts_prepared_artifact_and_defaults_output_root(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["pca_analysis.py", "--prepared", str(self.artifact_dir)],
        ):
            args = pca.parse_args()

        self.assertEqual(args.prepared, self.artifact_dir)
        self.assertEqual(args.output_root, pca.DEFAULT_OUTPUT_ROOT)

    def test_default_output_root_is_singular_output_pca(self) -> None:
        self.assertEqual(pca.DEFAULT_OUTPUT_ROOT.name, "output_pca")


if __name__ == "__main__":
    unittest.main()
