from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.cluster_kmeans import kmeans_analysis
from scripts.utils import clustering_preprocessing as preprocessing
from scripts.utils.clustering_preprocessing import (
    load_prepared_artifact,
    prepare_clustering_data,
    save_prepared_artifact,
)


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fid": [10, 11, 12, 13, 14, 15],
            "MS_ID": ["s0", "s1", "s2", "s3", "s4", "s5"],
            "PF_Index_Risk_equal": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "Per_extent": [-1.0, -0.8, np.nan, 0.7, 0.8, 1.0],
            "PFABC_NOR": [-0.9, -0.7, -0.5, 0.5, 0.7, 0.9],
        }
    )


class PreparedArtifactTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = source_frame()
        self.source_path = self.root / "source.csv"
        self.source.to_csv(self.source_path, index=False)
        self.artifact_dir = save_prepared_artifact(
            prepare_clustering_data(self.source, "PF"),
            self.source_path,
            self.root / "prepared",
            risk_filter={"minimum_risk": 0.0},
        )
        self.artifact = load_prepared_artifact(self.artifact_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()


class PreparedInputTests(PreparedArtifactTestCase):
    def test_prepare_data_uses_the_verified_standardized_matrix_without_local_cleanup(
        self,
    ) -> None:
        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("K-means must not transform prepared input")

        with patch.object(
            kmeans_analysis,
            "load_prepared_artifact",
            return_value=self.artifact,
        ), patch.object(
            pd.DataFrame, "dropna", side_effect=forbidden
        ), patch.object(
            pd.DataFrame, "fillna", side_effect=forbidden
        ), patch(
            "sklearn.preprocessing.StandardScaler.fit", side_effect=forbidden
        ):
            validated = kmeans_analysis.prepare_data(self.artifact_dir)
            fitted = kmeans_analysis.run_kmeans(validated.values, 2)

        self.assertEqual(validated.row_count, len(self.source))
        self.assertEqual(len(fitted.original_labels), len(self.source))
        self.assertEqual(validated.feature_names, self.artifact.feature_names)
        np.testing.assert_allclose(
            validated.values,
            self.artifact.standardized_features.to_numpy(dtype=float),
        )
        self.assertEqual(
            validated.metadata["source_row_position"].tolist(),
            list(range(len(self.source))),
        )

    def test_non_finite_prepared_matrix_is_rejected(self) -> None:
        with patch.object(
            kmeans_analysis,
            "load_prepared_artifact",
            return_value=self.artifact.__class__(
                **{
                    **self.artifact.__dict__,
                    "standardized_features": self.artifact.standardized_features.assign(
                        Per_extent=np.nan
                    ),
                }
            ),
        ):
            with self.assertRaisesRegex(
                kmeans_analysis.DataValidationError,
                "standardized matrix is not finite",
            ):
                kmeans_analysis.prepare_data(self.artifact_dir)


class KMeansPreparedRowTests(PreparedArtifactTestCase):
    def test_preflight_identifies_the_prepared_matrix_and_null_retention(self) -> None:
        validated = kmeans_analysis.prepare_data(self.artifact_dir)
        with patch("builtins.print") as printed:
            kmeans_analysis.print_preflight_summary(
                "PF", self.artifact_dir, 2, validated
            )

        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        for expected in (
            self.artifact.config["source_path"],
            self.artifact.config["source_sha256"],
            self.artifact.config["matrix_sha256"],
            "Prepared rows: 6",
            "Features (2): Per_extent, PFABC_NOR",
            "Rows with structural null: 1",
            "Rows excluded for null = 0",
        ):
            self.assertIn(expected, output)

    def test_kmeans_assigns_a_label_and_distance_to_every_prepared_row(self) -> None:
        validated = kmeans_analysis.prepare_data(self.artifact_dir)
        fitted = kmeans_analysis.run_kmeans(validated.values, 2)
        ordering = kmeans_analysis.reorder_cluster_labels(
            fitted, validated.values, validated.feature_names
        )
        tables = kmeans_analysis.calculate_cluster_profiles(
            validated, fitted, ordering
        )

        self.assertEqual(len(tables.clustered_data), len(self.source))
        self.assertFalse(tables.clustered_data["kmeans_cluster"].isna().any())
        self.assertEqual(len(tables.sample_distances), len(self.source))
        self.assertFalse(
            tables.sample_distances["distance_to_assigned_centroid"].isna().any()
        )
        self.assertEqual(
            tables.clustered_data.loc[
                tables.clustered_data["had_structural_null"],
                "structural_null_count",
            ].min(),
            1,
        )
        self.assertEqual(
            tables.sample_distances["source_row_position"].tolist(),
            list(range(len(self.source))),
        )

    def test_assignments_use_metadata_row_keys_in_matrix_row_order(self) -> None:
        validated = kmeans_analysis.prepare_data(self.artifact_dir)
        metadata = validated.metadata.copy()
        metadata["source_row_position"] = [41, 7, 93, 12, 58, 3]
        lower_boundary = kmeans_analysis.ValidatedData(
            artifact=validated.artifact,
            values=validated.values,
            feature_names=validated.feature_names,
            metadata=metadata,
            row_count=validated.row_count,
        )
        fitted = kmeans_analysis.run_kmeans(lower_boundary.values, 2)
        ordering = kmeans_analysis.reorder_cluster_labels(
            fitted, lower_boundary.values, lower_boundary.feature_names
        )

        tables = kmeans_analysis.calculate_cluster_profiles(
            lower_boundary, fitted, ordering
        )

        self.assertEqual(
            tables.clustered_data["source_row_position"].tolist(),
            [41, 7, 93, 12, 58, 3],
        )
        np.testing.assert_array_equal(
            tables.clustered_data["kmeans_cluster"],
            ordering.reordered_labels,
        )
        np.testing.assert_allclose(
            tables.sample_distances["distance_to_assigned_centroid"],
            fitted.assigned_distances,
        )

    def test_run_analysis_records_preprocessing_identity_and_zero_null_exclusions(
        self,
    ) -> None:
        output_root = self.root / "output"
        with patch("builtins.print"):
            output_path = kmeans_analysis.run_analysis(
                self.artifact_dir, 2, output_root
            )

        config = json.loads((output_path / "run_config.json").read_text("utf-8"))
        self.assertEqual(
            config["preprocessing_reference"]["manifest_sha256"],
            kmeans_analysis.sha256_file(self.artifact_dir / "preprocessing_config.json"),
        )
        self.assertEqual(
            config["preprocessing_reference"]["matrix_sha256"],
            self.artifact.config["matrix_sha256"],
        )
        self.assertEqual(
            config["preprocessing_reference"],
            {
                "manifest_path": str(
                    (self.artifact_dir / "preprocessing_config.json").resolve()
                ),
                "manifest_sha256": kmeans_analysis.sha256_file(
                    self.artifact_dir / "preprocessing_config.json"
                ),
                "matrix_sha256": self.artifact.config["matrix_sha256"],
                "source_path": str(self.source_path.resolve()),
                "source_sha256": self.artifact.config["source_sha256"],
                "scenario": "PF",
                "row_count": 6,
                "feature_names": ["Per_extent", "PFABC_NOR"],
                "feature_count": 2,
                "structural_null_row_count": 1,
                "structural_null_value_count": 1,
                "rows_excluded_for_null": 0,
            },
        )
        self.assertEqual(config["sample_counts"]["removed"], 0)
        self.assertEqual(config["missing_value_policy"], "Prepared structural nulls retained.")

    def test_k_must_be_strictly_smaller_than_every_prepared_row_count(self) -> None:
        with self.assertRaisesRegex(
            kmeans_analysis.DataValidationError,
            "K must satisfy 2 <= K < 6; received 6.",
        ):
            kmeans_analysis.run_analysis(self.artifact_dir, 6, self.root / "output")


class KMeansOriginalScaleOutputTests(PreparedArtifactTestCase):
    def _fitted_tables(self) -> tuple[
        kmeans_analysis.ValidatedData,
        kmeans_analysis.FittedKMeans,
        kmeans_analysis.AnalysisTables,
    ]:
        validated = kmeans_analysis.prepare_data(self.artifact_dir)
        fitted = kmeans_analysis.run_kmeans(validated.values, 2)
        ordering = kmeans_analysis.reorder_cluster_labels(
            fitted, validated.values, validated.feature_names
        )
        return validated, fitted, kmeans_analysis.calculate_cluster_profiles(
            validated, fitted, ordering
        )

    def test_profiles_preserve_centroid_scales_and_structural_nulls(self) -> None:
        validated, fitted, tables = self._fitted_tables()

        expected_original_centroids = preprocessing.inverse_transform_frame(
            fitted.standardized_centroids,
            validated.artifact.scaler_parameters,
        )
        pd.testing.assert_frame_equal(
            tables.cluster_centroids_original_scale,
            expected_original_centroids,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertEqual(
            tuple(tables.cluster_centroids_standardized.columns),
            validated.feature_names,
        )
        self.assertEqual(
            set(tables.cluster_feature_profile.columns),
            {
                "cluster",
                "feature",
                "original_non_null_count",
                "original_structural_null_count",
                "original_structural_null_percent",
                "original_non_null_mean",
                "original_non_null_median",
                "filled_mean",
                "standardized_centroid",
                "inverse_transformed_centroid",
            },
        )

        null_row = validated.artifact.original_features["Per_extent"].isna()
        cluster = int(tables.clustered_data.loc[null_row, "kmeans_cluster"].iloc[0])
        profile = tables.cluster_feature_profile.loc[
            (tables.cluster_feature_profile["cluster"] == cluster)
            & (tables.cluster_feature_profile["feature"] == "Per_extent")
        ].iloc[0]
        member_mask = tables.clustered_data["kmeans_cluster"].eq(cluster)
        original_values = validated.artifact.original_features.loc[
            member_mask, "Per_extent"
        ]
        self.assertEqual(
            int(profile["original_structural_null_count"]),
            int(original_values.isna().sum()),
        )
        self.assertAlmostEqual(
            float(profile["original_non_null_mean"]),
            float(original_values.mean()),
        )
        self.assertAlmostEqual(
            float(profile["filled_mean"]),
            float(
                validated.artifact.filled_features.loc[member_mask, "Per_extent"].mean()
            ),
        )

    def test_all_structural_null_profile_keeps_reporting_nan(self) -> None:
        validated = kmeans_analysis.prepare_data(self.artifact_dir)
        fitted = kmeans_analysis.run_kmeans(validated.values, 2)
        ordering = kmeans_analysis.reorder_cluster_labels(
            fitted, validated.values, validated.feature_names
        )
        member_mask = ordering.reordered_labels == 1
        original_features = validated.artifact.original_features.copy()
        filled_features = validated.artifact.filled_features.copy()
        original_features.loc[member_mask, "Per_extent"] = np.nan
        filled_features.loc[member_mask, "Per_extent"] = 0.0
        artifact = replace(
            validated.artifact,
            original_features=original_features,
            filled_features=filled_features,
        )
        modified = replace(validated, artifact=artifact)

        tables = kmeans_analysis.calculate_cluster_profiles(
            modified, fitted, ordering
        )

        profile = tables.cluster_feature_profile.loc[
            (tables.cluster_feature_profile["cluster"] == 1)
            & (tables.cluster_feature_profile["feature"] == "Per_extent")
        ].iloc[0]
        self.assertEqual(int(profile["original_non_null_count"]), 0)
        self.assertEqual(
            int(profile["original_structural_null_count"]),
            int(member_mask.sum()),
        )
        self.assertTrue(pd.isna(profile["original_non_null_mean"]))
        self.assertTrue(pd.isna(profile["original_non_null_median"]))
        self.assertEqual(float(profile["filled_mean"]), 0.0)
        self.assertTrue(np.isfinite(profile["inverse_transformed_centroid"]))

    def test_run_analysis_exports_original_scale_contract_and_audit_copy(self) -> None:
        output_root = self.root / "output"
        with patch("builtins.print"):
            output_path = kmeans_analysis.run_analysis(
                self.artifact_dir, 2, output_root
            )

        expected_outputs = {
            "clustered_data.csv",
            "cluster_summary.csv",
            "cluster_centroids_standardized.csv",
            "cluster_centroids_original_scale.csv",
            "cluster_feature_profile.csv",
            "sample_distances.csv",
            "model_metrics.csv",
            "cluster_label_mapping.csv",
            "preprocessing_audit.csv",
            "run_config.json",
            "run_report.txt",
            "pca_cluster_scatter.png",
            "cluster_size_bar.png",
            "cluster_centroid_heatmap.png",
            "cluster_feature_profiles.png",
        }
        self.assertEqual({path.name for path in output_path.iterdir()}, expected_outputs)
        for filename in (
            "cluster_centroids_standardized.csv",
            "cluster_centroids_original_scale.csv",
        ):
            centroid_table = pd.read_csv(output_path / filename)
            self.assertEqual(
                centroid_table.columns.tolist(),
                ["cluster", *self.artifact.feature_names],
            )
            self.assertEqual(centroid_table["cluster"].tolist(), [1, 2])
        clustered = pd.read_csv(output_path / "clustered_data.csv")
        self.assertTrue(
            {
                "fid",
                "MS_ID",
                "PF_Index_Risk_equal",
                "Per_extent",
                "PFABC_NOR",
                "kmeans_cluster",
                "had_structural_null",
                "structural_null_count",
            }.issubset(clustered.columns)
        )
        self.assertEqual(
            (output_path / "preprocessing_audit.csv").read_bytes(),
            (self.artifact_dir / "preprocessing_audit.csv").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
