"""End-to-end identity contracts for prepared clustering consumers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.preprocessing import StandardScaler

from scripts.cluster_kmeans import kmeans_analysis
from scripts.cluster_preparation import choose_k, pca_analysis
from scripts.clustering import hdbscan_runner
from scripts.utils.clustering_preprocessing import (
    prepare_clustering_data,
    save_prepared_artifact,
)


class UnifiedClusteringWorkflowTests(unittest.TestCase):
    """Protect the shared prepared-matrix boundary across all consumers."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source_path = self.root / "pf_high_risk.csv"
        self.source = pd.DataFrame(
            {
                "fid": [10, 11, 12, 13, 14, 15],
                "MS_ID": ["s0", "s1", "s2", "s3", "s4", "s5"],
                "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "Per_extent": [-1.0, -0.6, None, 0.2, 0.6, 1.0],
                "PFABC_NOR": [-0.8, -0.4, -0.1, 0.3, 0.7, 0.9],
                "PFResident_lossR": [-0.7, -0.3, 0.0, 0.4, 0.6, 0.8],
                "PFAB2k_lossR": [-0.9, -0.5, -0.1, 0.2, 0.6, 0.9],
                "N_PFAB2k_lossR": [-0.8, -0.4, 0.0, 0.3, 0.7, 1.0],
                "PFAB5k_lossR": [-1.0, -0.6, -0.2, 0.1, 0.5, 0.8],
                "N_PFAB5k_lossR": [-0.7, -0.3, 0.0, 0.4, 0.8, 0.9],
                "PFAB2k_NOR": [-0.95, -0.55, -0.15, 0.25, 0.65, 0.95],
                "PFAB5k_NOR": [-0.85, -0.45, -0.05, 0.35, 0.75, 1.0],
                "TFABC_NOR": [0.9, 0.7, 0.3, 0.0, -0.4, -0.8],
            }
        )
        self.source.to_csv(self.source_path, index=False)
        self.artifact_dir = save_prepared_artifact(
            prepare_clustering_data(self.source, "PF"),
            self.source_path,
            self.root / "prepared",
            risk_filter={
                "column": "PF_Index_Risk_equal",
                "operator": ">",
                "threshold": 0.25,
            },
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_consumers_share_prepared_matrix_identity_without_refitting_scaler(
        self,
    ) -> None:
        """A consumer that scales or reloads a different matrix breaks this contract."""

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("Downstream consumers must not fit a scaler")

        with patch.object(StandardScaler, "fit", side_effect=forbidden):
            pca_input = pca_analysis.load_pca_input(self.artifact_dir)
            choose_k_input = choose_k.load_prepared_artifact(self.artifact_dir)
            kmeans_input = kmeans_analysis.prepare_data(self.artifact_dir)
            hdbscan_input = hdbscan_runner.load_hdbscan_input(
                self.artifact_dir, "version_b"
            )
            kmeans_labels = kmeans_analysis.run_kmeans(kmeans_input.values, 2).original_labels

        self.assertEqual(
            pca_input.config["matrix_sha256"],
            choose_k_input.config["matrix_sha256"],
        )
        self.assertEqual(
            choose_k_input.config["matrix_sha256"],
            kmeans_input.artifact.config["matrix_sha256"],
        )
        self.assertEqual(
            kmeans_input.artifact.config["matrix_sha256"],
            hdbscan_input.artifact.config["matrix_sha256"],
        )
        self.assertEqual(pca_input.feature_names, choose_k_input.feature_names)
        self.assertEqual(choose_k_input.feature_names, kmeans_input.feature_names)
        self.assertEqual(
            kmeans_input.feature_names, hdbscan_input.artifact.feature_names
        )
        expected_row_count = len(kmeans_input.artifact.metadata)
        self.assertEqual(len(pca_input.metadata), expected_row_count)
        self.assertEqual(len(choose_k_input.metadata), expected_row_count)
        self.assertEqual(len(hdbscan_input.artifact.metadata), expected_row_count)
        self.assertEqual(
            pca_input.config["structural_null"],
            kmeans_input.artifact.config["structural_null"],
        )
        self.assertEqual(
            int(pca_input.metadata["had_structural_null"].sum()),
            int(hdbscan_input.artifact.metadata["had_structural_null"].sum()),
        )
        self.assertEqual(len(kmeans_labels), len(kmeans_input.artifact.metadata))
        self.assertFalse(pd.Series(kmeans_labels).isna().any())

    def test_raw_ab_lossr_never_reaches_any_prepared_consumer(self):
        artifact = pca_analysis.load_pca_input(self.artifact_dir)
        raw_names = {
            "PFAB2k_lossR",
            "N_PFAB2k_lossR",
            "PFAB5k_lossR",
            "N_PFAB5k_lossR",
        }

        self.assertTrue(raw_names.isdisjoint(artifact.feature_names))
        self.assertIn("PFAB2k_NOR", artifact.feature_names)
        self.assertIn("PFAB5k_NOR", artifact.feature_names)
        excluded = artifact.excluded_features.set_index("column")
        for name in raw_names:
            self.assertEqual(excluded.loc[name, "role"], "EXCLUDED")
            self.assertEqual(
                excluded.loc[name, "reason"],
                "raw_ab_lossR_replaced_by_normalized_NOR",
            )


if __name__ == "__main__":
    unittest.main()
