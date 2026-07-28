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
EXPECTED_MANAGED_FILENAMES = (
    "pca_summary.csv",
    "pca_loadings.csv",
    "pca_scores.csv",
    "explained_variance.png",
    "cumulative_variance.png",
    "pca_scatter.png",
    "loading_plot.png",
    "correlation_circle.png",
    "pca_report.txt",
    "preprocessing_reference.json",
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

    def patch_plots(
        self,
        stack: ExitStack,
        failing_function: str | None = None,
    ) -> None:
        def write_placeholder_plot(*args: object) -> None:
            Path(args[-1]).write_bytes(b"test plot")

        for function_name in PLOT_FUNCTIONS:
            side_effect: object
            if function_name == failing_function:
                side_effect = RuntimeError("simulated PCA output failure")
            else:
                side_effect = write_placeholder_plot
            stack.enter_context(
                patch.object(
                    pca,
                    function_name,
                    side_effect=side_effect,
                )
            )

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

    def test_repeated_run_replaces_managed_outputs_and_preserves_unmanaged_entries(
        self,
    ) -> None:
        output_root = self.root / "output_pca"
        with ExitStack() as stack:
            self.patch_plots(stack)
            output_dir = pca.run_analysis(self.artifact_dir, output_root)

        stale_report = output_dir / "pca_report.txt"
        stale_report.write_text("stale PCA report", encoding="utf-8")
        sentinel = output_dir / "manual_notes.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        manual_directory = output_dir / "manual_review"
        manual_directory.mkdir()
        manual_file = manual_directory / "decision.txt"
        manual_file.write_text("preserve directory", encoding="utf-8")

        with ExitStack() as stack:
            self.patch_plots(stack)
            repeated_output_dir = pca.run_analysis(
                self.artifact_dir,
                output_root,
            )

        self.assertEqual(repeated_output_dir, output_root / "pf")
        self.assertNotEqual(
            stale_report.read_text(encoding="utf-8"),
            "stale PCA report",
        )
        for filename in EXPECTED_MANAGED_FILENAMES:
            self.assertTrue((output_dir / filename).is_file(), filename)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertEqual(
            manual_file.read_text(encoding="utf-8"),
            "preserve directory",
        )

    def test_generation_failure_leaves_previous_managed_outputs_unchanged(
        self,
    ) -> None:
        output_root = self.root / "output_pca"
        with ExitStack() as stack:
            self.patch_plots(stack)
            output_dir = pca.run_analysis(self.artifact_dir, output_root)
        before = {
            filename: (output_dir / filename).read_bytes()
            for filename in EXPECTED_MANAGED_FILENAMES
        }

        with ExitStack() as stack:
            self.patch_plots(
                stack,
                failing_function="plot_cumulative_variance",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated PCA output failure",
            ):
                pca.run_analysis(self.artifact_dir, output_root)

        after = {
            filename: (output_dir / filename).read_bytes()
            for filename in EXPECTED_MANAGED_FILENAMES
        }
        self.assertEqual(after, before)

    def test_existing_scenario_path_must_be_a_directory(self) -> None:
        output_root = self.root / "output_pca"
        output_root.mkdir()
        scenario_path = output_root / "pf"
        scenario_path.write_text("not a directory", encoding="utf-8")

        with self.assertRaisesRegex(
            NotADirectoryError,
            "PCA final output path is not a directory",
        ):
            pca.run_analysis(self.artifact_dir, output_root)

        self.assertEqual(
            scenario_path.read_text(encoding="utf-8"),
            "not a directory",
        )

    def test_publish_rejects_an_incomplete_staging_set_before_replacement(
        self,
    ) -> None:
        staging_dir = self.root / "staging"
        staging_dir.mkdir()
        output_dir = self.root / "output_pca" / "pf"
        output_dir.mkdir(parents=True)
        old_report = output_dir / "pca_report.txt"
        old_report.write_text("previous report", encoding="utf-8")
        for filename in EXPECTED_MANAGED_FILENAMES:
            if filename != "pca_report.txt":
                (staging_dir / filename).write_bytes(b"new staged output")

        with self.assertRaisesRegex(
            RuntimeError,
            "PCA staging did not produce every managed output",
        ):
            pca._publish_pca_outputs(staging_dir, output_dir)

        self.assertEqual(
            old_report.read_text(encoding="utf-8"),
            "previous report",
        )
        self.assertFalse((output_dir / "pca_summary.csv").exists())

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
