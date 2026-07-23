from __future__ import annotations

import json
import os
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.cluster_preparation import choose_k
from scripts.cluster_preparation import pca_analysis
from scripts.cluster_kmeans import kmeans_analysis


class SharedScenarioConfigurationTests(unittest.TestCase):
    def test_pca_and_choose_k_share_the_same_feature_objects(self) -> None:
        from scripts.utils import scenario_features

        self.assertIs(pca_analysis.PF_FEATURES, scenario_features.PF_FEATURES)
        self.assertIs(pca_analysis.TF_FEATURES, scenario_features.TF_FEATURES)
        self.assertIs(choose_k.PF_FEATURES, scenario_features.PF_FEATURES)
        self.assertIs(choose_k.TF_FEATURES, scenario_features.TF_FEATURES)
        self.assertEqual(len(scenario_features.PF_FEATURES), 7)
        self.assertEqual(len(scenario_features.TF_FEATURES), 7)


class PromptTests(unittest.TestCase):
    def test_prompt_scenario_retries_and_accepts_aliases(self) -> None:
        with patch("builtins.input", side_effect=["bad", " 2 "]), patch(
            "builtins.print"
        ) as mocked_print:
            self.assertEqual(kmeans_analysis.prompt_scenario(), "TF")

        mocked_print.assert_any_call("Invalid selection. Enter 1/pf or 2/tf.")

    def test_prompt_k_retries_for_integer_and_sample_limit(self) -> None:
        with patch("builtins.input", side_effect=["2.0", "1", "5", "3"]), patch(
            "builtins.print"
        ):
            self.assertEqual(kmeans_analysis.prompt_k(n_samples=5), 3)

    def test_prompt_k_accepts_signed_decimal_text_with_leading_zeros(self) -> None:
        with patch("builtins.input", side_effect=["2.0", "-02", "+02"]), patch(
            "builtins.print"
        ):
            self.assertEqual(kmeans_analysis.prompt_k(n_samples=5), 2)

    def test_prompt_csv_path_strips_quotes_and_retries(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            valid_path = Path(temporary_directory) / "prepared data.csv"
            valid_path.write_text("a,b\n1,2\n", encoding="utf-8")
            with patch(
                "builtins.input",
                side_effect=["'missing.csv'", f'"{valid_path}"'],
            ), patch("builtins.print"):
                selected = kmeans_analysis.prompt_csv_path("PF")

        self.assertEqual(selected, valid_path.resolve())

    def test_prompt_csv_path_retries_filesystem_access_errors(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            valid_path = (Path(temporary_directory) / "prepared.csv").resolve()
            valid_path.write_text("a,b\n1,2\n", encoding="utf-8")
            with patch("builtins.input", side_effect=[str(valid_path)] * 2), patch(
                "builtins.print"
            ) as mocked_print, patch.object(
                kmeans_analysis,
                "_resolve_readable_csv_path",
                side_effect=[PermissionError("sharing violation"), valid_path],
                create=True,
            ) as mocked_resolve:
                selected = kmeans_analysis.prompt_csv_path("PF")

        self.assertEqual(selected, valid_path)
        self.assertEqual(mocked_resolve.call_count, 2)
        self.assertTrue(
            any(
                "sharing violation" in str(call.args[0])
                for call in mocked_print.call_args_list
            )
        )


def prepared_pf_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fid": [1, 2, 3, 4, 5, 6],
            "MS_ID": [101, 102, 103, 104, 105, 106],
            "Per_extent": [-1.0, -0.9, -0.8, 0.8, 0.9, 1.0],
            "N_PFlossR_2kiw": [-1.1, -1.0, np.nan, 0.9, 1.0, 1.1],
            "N_PFlossR_2kmw": [-0.9, -0.8, -0.7, 0.7, 0.8, 0.9],
            "PFResident_lossR": [-1.2, -1.0, -0.8, 0.8, 1.0, 1.2],
            "PFDaynight_lossR": [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0],
            "PFAB5k_NOR": [-0.7, -0.6, -0.5, 0.5, 0.6, 0.7],
            "PFABC_NOR": [-0.8, -0.7, -0.6, 0.6, 0.7, 0.8],
            "grid_density": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
            "PF_Index_Risk_equal": [0.2] * 6,
            "street_name": ["a", "b", "c", "d", "e", "f"],
        }
    )


class FeatureValidationTests(unittest.TestCase):
    def test_selects_only_shared_pf_features_and_exact_ids(self) -> None:
        data = prepared_pf_frame()
        selection = kmeans_analysis.select_features(data, "PF")

        self.assertEqual(
            list(selection.feature_names),
            list(kmeans_analysis.PF_FEATURES),
        )
        self.assertEqual(list(selection.id_columns), ["fid", "MS_ID"])
        self.assertNotIn("grid_density", selection.id_columns)
        reasons = {item.column: item.reason for item in selection.exclusions}
        self.assertEqual(reasons["grid_density"], "not in scenario feature configuration")
        self.assertEqual(reasons["PF_Index_Risk_equal"], "excluded keyword: index")

    def test_validate_features_drops_missing_rows_without_imputation(self) -> None:
        data = prepared_pf_frame()
        validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )

        self.assertEqual(len(validated.original_data), 6)
        self.assertEqual(len(validated.features), 5)
        self.assertFalse(validated.valid_mask.iloc[2])
        self.assertEqual(validated.missing_counts["N_PFlossR_2kiw"], 1)
        self.assertNotIn(0.0, validated.features["N_PFlossR_2kiw"].tolist())

    def test_excludes_constant_and_infinite_configured_columns(self) -> None:
        data = prepared_pf_frame()
        data["PFAB5k_NOR"] = 1.0
        data.loc[0, "PFABC_NOR"] = np.inf

        validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )

        self.assertNotIn("PFAB5k_NOR", validated.feature_names)
        self.assertNotIn("PFABC_NOR", validated.feature_names)
        actions = {issue.column: issue.action for issue in validated.issues}
        self.assertEqual(actions["PFAB5k_NOR"], "excluded feature")
        self.assertEqual(actions["PFABC_NOR"], "excluded feature")

    def test_rejects_missing_nonnumeric_and_reserved_columns(self) -> None:
        cases = [
            (
                prepared_pf_frame().drop(columns=["Per_extent"]),
                "Missing configured feature: Per_extent",
            ),
            (
                prepared_pf_frame().assign(Per_extent=["x"] * 6),
                "Configured feature is not numeric: Per_extent",
            ),
            (
                prepared_pf_frame().assign(kmeans_cluster=[1] * 6),
                "Reserved output column already exists: kmeans_cluster",
            ),
        ]
        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    kmeans_analysis.DataValidationError, message
                ):
                    kmeans_analysis.select_features(data, "PF")


class ModelTests(unittest.TestCase):
    def test_fit_parameter_value_error_is_reported_as_compatibility_error(
        self,
    ) -> None:
        class RejectingKMeans:
            def __init__(
                self,
                n_clusters: int,
                init: str,
                n_init: int,
                max_iter: int,
                random_state: int,
                algorithm: str,
                tol: float,
            ) -> None:
                self.algorithm = algorithm

            def fit_predict(self, values: np.ndarray) -> np.ndarray:
                raise ValueError(
                    "Algorithm must be 'auto', 'full' or 'elkan', "
                    "got 'lloyd' instead."
                )

        with patch("sklearn.cluster.KMeans", RejectingKMeans):
            with self.assertRaises(kmeans_analysis.CompatibilityError) as caught:
                kmeans_analysis.run_kmeans(
                    np.array([[0.0], [1.0], [2.0]], dtype=float), 2
                )

        from sklearn import __version__ as sklearn_version

        message = str(caught.exception)
        self.assertIn(sklearn_version, message)
        self.assertIn("algorithm='lloyd'", message)

    def test_nonparameter_fit_value_error_remains_a_data_error(self) -> None:
        class RejectingDataKMeans:
            def __init__(
                self,
                n_clusters: int,
                init: str,
                n_init: int,
                max_iter: int,
                random_state: int,
                algorithm: str,
                tol: float,
            ) -> None:
                pass

            def fit_predict(self, values: np.ndarray) -> np.ndarray:
                raise ValueError("Input X contains NaN.")

        with patch("sklearn.cluster.KMeans", RejectingDataKMeans):
            with self.assertRaises(kmeans_analysis.DataValidationError):
                kmeans_analysis.run_kmeans(
                    np.array([[0.0], [1.0], [2.0]], dtype=float), 2
                )

    def test_run_kmeans_uses_fixed_parameters_and_distances(self) -> None:
        values = np.array(
            [[-1.1, -1.0], [-0.9, -1.2], [0.9, 1.0], [1.1, 1.2]],
            dtype=float,
        )
        fitted = kmeans_analysis.run_kmeans(values, 2)

        params = fitted.model.get_params()
        self.assertEqual(params["init"], "k-means++")
        self.assertEqual(params["n_init"], 50)
        self.assertEqual(params["max_iter"], 500)
        self.assertEqual(params["random_state"], 42)
        self.assertEqual(params["algorithm"], "lloyd")
        self.assertEqual(fitted.assigned_distances.shape, (4,))
        self.assertGreaterEqual(fitted.inertia, 0.0)

    def test_reorder_uses_centroid_pc1_and_preserves_membership(self) -> None:
        values = np.array(
            [[-2.0, -1.8], [-1.8, -2.0], [1.8, 2.0], [2.0, 1.8]],
            dtype=float,
        )
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordered = kmeans_analysis.reorder_cluster_labels(
            fitted, values, ("a", "b")
        )

        original_same = (
            fitted.original_labels[:, None] == fitted.original_labels[None, :]
        )
        reordered_same = (
            ordered.reordered_labels[:, None] == ordered.reordered_labels[None, :]
        )
        np.testing.assert_array_equal(original_same, reordered_same)
        self.assertEqual(
            ordered.mapping["reordered_cluster_id"].tolist(), [1, 2]
        )
        self.assertTrue(
            ordered.mapping["ordering_value"].is_monotonic_increasing
        )
        self.assertEqual(ordered.ordering_method, "centroid PC1 projection")

    def test_reorder_falls_back_to_centroid_mean_for_one_feature(self) -> None:
        values = np.array([[-2.0], [-1.8], [1.8], [2.0]], dtype=float)
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordered = kmeans_analysis.reorder_cluster_labels(fitted, values, ("a",))

        self.assertEqual(ordered.ordering_method, "centroid mean")
        self.assertIsNone(ordered.pca_scores)
        self.assertTrue(any("PCA" in warning for warning in ordered.warnings))


class AnalysisTableTests(unittest.TestCase):
    def setUp(self) -> None:
        data = prepared_pf_frame()
        self.validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )
        values = self.validated.features.to_numpy(dtype=float)
        self.fitted = kmeans_analysis.run_kmeans(values, 2)
        self.ordered = kmeans_analysis.reorder_cluster_labels(
            self.fitted, values, self.validated.feature_names
        )

    def test_profiles_preserve_all_rows_and_blank_invalid_label(self) -> None:
        tables = kmeans_analysis.calculate_cluster_profiles(
            self.validated, self.fitted, self.ordered
        )

        self.assertEqual(len(tables.clustered_data), 6)
        self.assertTrue(pd.isna(tables.clustered_data.loc[2, "kmeans_cluster"]))
        self.assertEqual(len(tables.cluster_summary), 2)
        self.assertAlmostEqual(
            tables.cluster_summary["sample_percentage"].sum(), 100.0
        )
        self.assertEqual(
            len(tables.feature_profile), 2 * len(self.validated.feature_names)
        )
        self.assertEqual(len(tables.sample_distances), 5)
        self.assertEqual(
            list(tables.sample_distances.columns),
            [
                "fid",
                "MS_ID",
                "kmeans_cluster",
                "distance_to_assigned_centroid",
            ],
        )

    def test_calculate_metrics_returns_all_requested_values(self) -> None:
        values = self.validated.features.to_numpy(dtype=float)
        result = kmeans_analysis.calculate_metrics(
            values, self.ordered.reordered_labels
        )

        self.assertEqual(
            set(result.values),
            {
                "silhouette_score",
                "calinski_harabasz_score",
                "davies_bouldin_score",
            },
        )
        self.assertTrue(all(value is not None for value in result.values.values()))

    def test_exact_centroids_distances_inertia_profiles_and_reordered_labels(
        self,
    ) -> None:
        data = pd.DataFrame(
            {
                "segment_id": [1, 2, 3, 4, 5],
                "a": [0.0, 0.0, 0.0, 10.0, 10.0],
                "b": [0.0, 2.0, 4.0, 10.0, 12.0],
            }
        )
        validated = kmeans_analysis.ValidatedData(
            original_data=data.copy(),
            features=data.loc[:, ["a", "b"]].copy(),
            valid_mask=pd.Series([True] * 5, index=data.index),
            feature_names=("a", "b"),
            exclusions=[],
            id_columns=("segment_id",),
            missing_counts={},
            issues=[],
        )
        values = validated.features.to_numpy(dtype=float)
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordering = kmeans_analysis.reorder_cluster_labels(
            fitted, values, validated.feature_names
        )
        tables = kmeans_analysis.calculate_cluster_profiles(
            validated, fitted, ordering
        )

        centers = np.asarray(fitted.model.cluster_centers_)
        np.testing.assert_allclose(
            centers[np.argsort(centers[:, 0])],
            np.array([[0.0, 2.0], [10.0, 11.0]]),
        )
        expected_distances = np.array([2.0, 0.0, 2.0, 1.0, 1.0])
        np.testing.assert_allclose(fitted.assigned_distances, expected_distances)
        self.assertAlmostEqual(fitted.inertia, 10.0)
        remap = dict(
            zip(
                ordering.mapping["sklearn_original_label"],
                ordering.mapping["reordered_cluster_id"],
            )
        )
        np.testing.assert_array_equal(
            ordering.reordered_labels,
            np.array([remap[int(label)] for label in fitted.original_labels]),
        )
        np.testing.assert_array_equal(
            tables.sample_distances["kmeans_cluster"],
            ordering.reordered_labels,
        )
        np.testing.assert_allclose(
            tables.sample_distances["distance_to_assigned_centroid"],
            expected_distances,
        )
        with TemporaryDirectory() as temporary_directory:
            kmeans_analysis.export_data_tables(
                Path(temporary_directory), tables
            )
            exported_distances = pd.read_csv(
                Path(temporary_directory) / "sample_distances.csv"
            )
        np.testing.assert_allclose(
            exported_distances["distance_to_assigned_centroid"],
            expected_distances,
        )
        np.testing.assert_array_equal(
            exported_distances["kmeans_cluster"],
            ordering.reordered_labels,
        )
        low_cluster_id = int(
            tables.centroids.sort_values("a").iloc[0]["cluster_id"]
        )
        high_cluster_id = int(
            tables.centroids.sort_values("a").iloc[1]["cluster_id"]
        )
        low_summary = tables.cluster_summary.loc[
            tables.cluster_summary["cluster_id"] == low_cluster_id
        ].iloc[0]
        high_summary = tables.cluster_summary.loc[
            tables.cluster_summary["cluster_id"] == high_cluster_id
        ].iloc[0]
        self.assertAlmostEqual(low_summary["sample_percentage"], 60.0)
        self.assertAlmostEqual(high_summary["sample_percentage"], 40.0)
        self.assertAlmostEqual(
            low_summary["distance_to_centroid_mean"], 4.0 / 3.0
        )
        self.assertAlmostEqual(
            low_summary["distance_to_centroid_median"], 2.0
        )
        self.assertAlmostEqual(
            low_summary["distance_to_centroid_std"], np.sqrt(8.0) / 3.0
        )
        self.assertNotAlmostEqual(
            low_summary["distance_to_centroid_std"], np.sqrt(4.0 / 3.0)
        )
        self.assertAlmostEqual(low_summary["distance_to_centroid_max"], 2.0)
        self.assertAlmostEqual(
            high_summary["distance_to_centroid_std"], 0.0
        )
        low_b = tables.feature_profile.loc[
            (tables.feature_profile["cluster_id"] == low_cluster_id)
            & (tables.feature_profile["feature"] == "b")
        ].iloc[0]
        np.testing.assert_allclose(
            low_b[
                ["mean", "median", "std", "min", "max", "q25", "q75"]
            ].to_numpy(dtype=float),
            [2.0, 2.0, np.sqrt(8.0 / 3.0), 0.0, 4.0, 1.0, 3.0],
        )

    def test_each_metric_failure_is_independent_and_recorded(self) -> None:
        values = np.array(
            [[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]]
        )
        labels = np.array([1, 1, 2, 2])
        metric_paths = {
            "silhouette_score": "sklearn.metrics.silhouette_score",
            "calinski_harabasz_score": (
                "sklearn.metrics.calinski_harabasz_score"
            ),
            "davies_bouldin_score": "sklearn.metrics.davies_bouldin_score",
        }
        for metric_name, metric_path in metric_paths.items():
            with self.subTest(metric=metric_name), patch(
                metric_path, side_effect=ValueError("forced metric failure")
            ):
                result = kmeans_analysis.calculate_metrics(values, labels)

            self.assertIsNone(result.values[metric_name])
            self.assertTrue(
                all(
                    result.values[name] is not None
                    for name in metric_paths
                    if name != metric_name
                )
            )
            self.assertEqual(len(result.warnings), 1)
            self.assertIn(metric_name, result.warnings[0])
            self.assertIn("forced metric failure", result.warnings[0])

    def test_full_silhouette_uses_bounded_working_memory_at_actual_tf_size(
        self,
    ) -> None:
        values = np.zeros((13_584, 7), dtype=float)
        labels = np.tile(np.array([1, 2, 3, 4]), 3_396)
        context_calls: list[dict[str, object]] = []
        silhouette_shapes: list[tuple[int, ...]] = []

        @contextmanager
        def recording_context(**kwargs: object):
            context_calls.append(kwargs)
            yield

        def record_full_input(
            metric_values: np.ndarray, metric_labels: np.ndarray, **kwargs: object
        ) -> float:
            silhouette_shapes.append(metric_values.shape)
            self.assertEqual(metric_labels.shape, (13_584,))
            self.assertNotIn("sample_size", kwargs)
            return 0.25

        with patch(
            "sklearn.config_context", side_effect=recording_context
        ), patch(
            "sklearn.metrics.silhouette_score", side_effect=record_full_input
        ), patch(
            "sklearn.metrics.calinski_harabasz_score", return_value=2.0
        ), patch(
            "sklearn.metrics.davies_bouldin_score", return_value=0.5
        ):
            result = kmeans_analysis.calculate_metrics(values, labels)

        self.assertEqual(silhouette_shapes, [(13_584, 7)])
        self.assertEqual(
            context_calls,
            [{"working_memory": kmeans_analysis.SILHOUETTE_WORKING_MEMORY_MB}],
        )
        self.assertEqual(result.values["silhouette_score"], 0.25)


class PlotTests(AnalysisTableTests):
    def test_all_required_plots_are_written(self) -> None:
        tables = kmeans_analysis.calculate_cluster_profiles(
            self.validated, self.fitted, self.ordered
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = [
                root / "pca_cluster_scatter.png",
                root / "cluster_size_bar.png",
                root / "cluster_centroid_heatmap.png",
                root / "cluster_feature_profiles.png",
            ]
            kmeans_analysis.generate_pca_plot(
                self.ordered, "PF", 2, self.validated.feature_names, paths[0]
            )
            kmeans_analysis.generate_cluster_size_plot(
                tables.cluster_summary, paths[1]
            )
            kmeans_analysis.generate_centroid_heatmap(
                tables.centroids, self.validated.feature_names, paths[2]
            )
            kmeans_analysis.generate_profile_plot(
                tables.feature_profile, self.validated.feature_names, paths[3]
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_pca_unavailable_still_writes_placeholder(self) -> None:
        values = np.array([[-2.0], [-1.8], [1.8], [2.0]], dtype=float)
        fitted = kmeans_analysis.run_kmeans(values, 2)
        ordered = kmeans_analysis.reorder_cluster_labels(fitted, values, ("a",))
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pca_cluster_scatter.png"
            kmeans_analysis.generate_pca_plot(ordered, "PF", 2, ("a",), path)
            self.assertGreater(path.stat().st_size, 0)


class ExportTests(AnalysisTableTests):
    def test_build_output_directory_is_readable_and_refuses_existing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = kmeans_analysis.build_output_directory(root, "TF", 4, 7)
            self.assertEqual(
                target.name, "tf_k4_v7_kmeanspp_n50_mi500_rs42"
            )
            target.mkdir()
            with self.assertRaisesRegex(
                kmeans_analysis.OutputDirectoryExistsError,
                "move or rename",
            ):
                kmeans_analysis.build_output_directory(root, "TF", 4, 7)

    def test_build_output_directory_refuses_dangling_final_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = (
                root / "tf_k4_v7_kmeanspp_n50_mi500_rs42"
            )
            try:
                target.symlink_to(
                    root / "missing-target", target_is_directory=True
                )
            except OSError:
                with patch("os.path.lexists", return_value=True) as mocked_lexists:
                    with self.assertRaisesRegex(
                        kmeans_analysis.OutputDirectoryExistsError,
                        "move or rename",
                    ):
                        kmeans_analysis.build_output_directory(root, "TF", 4, 7)
                mocked_lexists.assert_called_once_with(os.fspath(target))
            else:
                self.assertFalse(target.exists())
                self.assertTrue(os.path.lexists(target))
                with self.assertRaisesRegex(
                    kmeans_analysis.OutputDirectoryExistsError,
                    "move or rename",
                ):
                    kmeans_analysis.build_output_directory(root, "TF", 4, 7)

    def test_staging_cleanup_refuses_retargeted_directory_identity(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            final_path = root / "final"
            marker: Path | None = None
            with self.assertRaisesRegex(
                kmeans_analysis.KMeansAnalysisError,
                "staging identity changed",
            ):
                with kmeans_analysis.staged_output_directory(final_path) as staging:
                    shutil.rmtree(staging)
                    staging.mkdir()
                    marker = staging / "unrelated-marker.txt"
                    marker.write_text("keep", encoding="utf-8")
                    raise RuntimeError("simulated export failure")

            self.assertIsNotNone(marker)
            self.assertTrue(marker.exists())
            self.assertFalse(os.path.lexists(final_path))

    def test_export_results_writes_required_utf8_files(self) -> None:
        tables = kmeans_analysis.calculate_cluster_profiles(
            self.validated, self.fitted, self.ordered
        )
        metrics = kmeans_analysis.calculate_metrics(
            self.validated.features.to_numpy(dtype=float),
            self.ordered.reordered_labels,
        )
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            config = kmeans_analysis.build_run_config(
                scenario="PF",
                input_path=Path("pf.csv").resolve(),
                output_path=(output / "final").resolve(),
                k=2,
                validated=self.validated,
                fitted=self.fitted,
                ordering=self.ordered,
                warnings_list=[],
                started_at="2026-07-23T12:00:00+02:00",
                ended_at="2026-07-23T12:00:01+02:00",
                runtime_seconds=1.0,
            )
            report = kmeans_analysis.build_run_report(
                config, tables.cluster_summary, metrics.values
            )
            kmeans_analysis.export_results(
                output,
                tables,
                metrics,
                config,
                report,
            )

            required = set(kmeans_analysis.TABULAR_TEXT_OUTPUTS)
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            saved_config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(saved_config["standard_scaler_applied"])
            self.assertEqual(
                saved_config["feature_list"],
                list(self.validated.feature_names),
            )
            self.assertEqual(saved_config["runtime"]["runtime_seconds"], 1.0)
            self.assertEqual(
                saved_config["runtime"]["runtime_scope"],
                kmeans_analysis.PERSISTED_RUNTIME_SCOPE,
            )
            self.assertEqual(
                saved_config["metric_parameters"],
                {
                    "silhouette_sample_size": None,
                    "silhouette_working_memory_mb": (
                        kmeans_analysis.SILHOUETTE_WORKING_MEMORY_MB
                    ),
                },
            )
            model_metrics = pd.read_csv(output / "model_metrics.csv")
            self.assertEqual(model_metrics.loc[0, "runtime_seconds"], 1.0)
            self.assertEqual(
                model_metrics.loc[0, "runtime_scope"],
                kmeans_analysis.PERSISTED_RUNTIME_SCOPE,
            )
            self.assertIn("Persisted staging-complete runtime", report)
            self.assertIn(kmeans_analysis.PERSISTED_RUNTIME_SCOPE, report)


class EndToEndTests(unittest.TestCase):
    def test_timing_separates_artifact_snapshot_from_end_to_end_completion(
        self,
    ) -> None:
        clock = {"seconds": 0.0}
        events: list[str] = []
        metadata_configs: list[dict[str, object]] = []
        real_staging = kmeans_analysis.staged_output_directory
        real_export_data = kmeans_analysis.export_data_tables
        real_export_metadata = kmeans_analysis.export_run_metadata

        @contextmanager
        def timed_staging(final_path: Path):
            with real_staging(final_path) as staging:
                yield staging
            clock["seconds"] = 9.0
            events.append("published")

        def export_data(*args: object, **kwargs: object) -> None:
            real_export_data(*args, **kwargs)
            clock["seconds"] = 5.0
            events.append("bulk")

        def export_metadata(*args: object, **kwargs: object) -> None:
            real_export_metadata(*args, **kwargs)
            metadata_configs.append(json.loads(json.dumps(args[2])))
            if len(metadata_configs) == 1:
                clock["seconds"] = 7.0
                events.append("first_metadata")
            else:
                clock["seconds"] = 8.0
                events.append("final_metadata")

        def write_placeholder_plot(*args: object, **kwargs: object) -> None:
            Path(args[-1]).write_bytes(b"test-png")

        data = prepared_pf_frame().fillna({"N_PFlossR_2kiw": -0.95})
        validated = kmeans_analysis.validate_features(
            data, kmeans_analysis.select_features(data, "PF")
        )
        with TemporaryDirectory() as temporary_directory:
            with patch.object(
                kmeans_analysis,
                "staged_output_directory",
                side_effect=timed_staging,
            ), patch.object(
                kmeans_analysis.time,
                "perf_counter",
                side_effect=lambda: clock["seconds"],
            ), patch.object(
                kmeans_analysis,
                "export_data_tables",
                side_effect=export_data,
            ), patch.object(
                kmeans_analysis,
                "export_run_metadata",
                side_effect=export_metadata,
            ), patch.object(
                kmeans_analysis,
                "generate_pca_plot",
                side_effect=write_placeholder_plot,
            ), patch.object(
                kmeans_analysis,
                "generate_cluster_size_plot",
                side_effect=write_placeholder_plot,
            ), patch.object(
                kmeans_analysis,
                "generate_centroid_heatmap",
                side_effect=write_placeholder_plot,
            ), patch.object(
                kmeans_analysis,
                "generate_profile_plot",
                side_effect=write_placeholder_plot,
            ), patch.object(
                kmeans_analysis, "print_completion_summary"
            ) as mocked_completion, patch(
                "builtins.print"
            ):
                final_path = kmeans_analysis._run_validated_analysis(
                    "PF",
                    2,
                    Path("prepared.csv"),
                    Path(temporary_directory) / "output",
                    validated,
                    kmeans_analysis.datetime.now().astimezone(),
                    0.0,
                )

            self.assertTrue(final_path.exists())
            self.assertEqual(
                {path.name for path in final_path.iterdir()},
                set(kmeans_analysis.ALL_OUTPUTS),
            )
            self.assertEqual(
                list(final_path.parent.glob(".kmeans_staging_*")), []
            )
            persisted_config = json.loads(
                (final_path / "run_config.json").read_text(encoding="utf-8")
            )
            persisted_metrics = pd.read_csv(
                final_path / "model_metrics.csv"
            )
            persisted_report = (final_path / "run_report.txt").read_text(
                encoding="utf-8"
            )

        self.assertEqual(
            events,
            ["bulk", "first_metadata", "final_metadata", "published"],
        )
        self.assertEqual(len(metadata_configs), 2)
        self.assertEqual(
            persisted_config["runtime"]["runtime_seconds"], 7.0
        )
        self.assertEqual(
            persisted_config["runtime"]["runtime_scope"],
            kmeans_analysis.PERSISTED_RUNTIME_SCOPE,
        )
        self.assertEqual(persisted_metrics.loc[0, "runtime_seconds"], 7.0)
        self.assertIn("7.000 seconds", persisted_report)
        self.assertEqual(mocked_completion.call_args.args[6], 9.0)

    def test_small_pf_run_creates_complete_final_directory_without_scaling(
        self,
    ) -> None:
        data = pd.concat(
            [
                prepared_pf_frame().fillna(
                    {"N_PFlossR_2kiw": -0.95}
                ),
                prepared_pf_frame().fillna(
                    {"N_PFlossR_2kiw": -0.85}
                ).assign(fid=lambda frame: frame["fid"] + 10),
            ],
            ignore_index=True,
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "pf_for_PCA.csv"
            output_root = root / "output_kmeans"
            data.to_csv(input_path, index=False, encoding="utf-8")
            with patch("builtins.print"):
                final_path = kmeans_analysis.run_analysis(
                    "PF", 2, input_path, output_root
                )

            self.assertEqual(
                final_path.name, "pf_k2_v7_kmeanspp_n50_mi500_rs42"
            )
            self.assertEqual(
                {path.name for path in final_path.iterdir()},
                set(kmeans_analysis.ALL_OUTPUTS),
            )
            clustered = pd.read_csv(final_path / "clustered_data.csv")
            self.assertEqual(len(clustered), len(data))
            self.assertEqual(
                clustered["kmeans_cluster"].notna().sum(), len(data)
            )
            source_values = data[list(kmeans_analysis.PF_FEATURES)].to_numpy()
            exported_source = clustered[
                list(kmeans_analysis.PF_FEATURES)
            ].to_numpy()
            np.testing.assert_allclose(exported_source, source_values)

    def test_source_contains_no_scaler_import_or_call(self) -> None:
        source = Path(kmeans_analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from sklearn.preprocessing", source)
        self.assertNotIn("StandardScaler(", source)
        self.assertNotIn("MinMaxScaler(", source)

    def test_preflight_and_completion_summaries_include_required_details(
        self,
    ) -> None:
        validated = kmeans_analysis.validate_features(
            prepared_pf_frame(),
            kmeans_analysis.select_features(prepared_pf_frame(), "PF"),
        )
        cluster_summary = pd.DataFrame(
            {
                "cluster_id": [1, 2],
                "sample_count": [2, 3],
                "sample_percentage": [40.0, 60.0],
            }
        )
        with patch("builtins.print") as mocked_print:
            kmeans_analysis.print_preflight_summary(
                "PF", Path("pf.csv"), 2, validated
            )
            kmeans_analysis.print_completion_summary(
                "PF",
                2,
                validated,
                cluster_summary,
                {"silhouette_score": 0.75},
                1.25,
                2.5,
                Path("output"),
            )

        lines = [str(call.args[0]) for call in mocked_print.call_args_list]
        output = "\n".join(lines)
        self.assertIn(
            "No StandardScaler, normalization, or other scaling will be applied.",
            lines,
        )
        for exclusion in validated.exclusions:
            self.assertIn(exclusion.column, output)
            self.assertIn(exclusion.reason, output)
        self.assertIn("K-means analysis completed.", lines)
        for expected in (
            "PF",
            "2",
            str(len(validated.feature_names)),
            str(len(validated.features)),
            "output",
            "Cluster 1",
            "2",
            "40.00%",
            "Cluster 2",
            "3",
            "60.00%",
            "0.750000",
            "1.250000",
            "2.50",
        ):
            self.assertIn(expected, output)

    def test_existing_final_directory_is_rejected_before_kmeans_fit(
        self,
    ) -> None:
        data = prepared_pf_frame().fillna({"N_PFlossR_2kiw": -0.95})
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "pf_for_PCA.csv"
            output_root = root / "output_kmeans"
            input_path.write_text(data.to_csv(index=False), encoding="utf-8")
            existing = (
                output_root / "pf_k2_v7_kmeanspp_n50_mi500_rs42"
            )
            existing.mkdir(parents=True)
            with patch.object(kmeans_analysis, "run_kmeans") as mocked_fit:
                with self.assertRaises(
                    kmeans_analysis.OutputDirectoryExistsError
                ):
                    kmeans_analysis.run_analysis(
                        "PF", 2, input_path, output_root
                    )

        mocked_fit.assert_not_called()

    def test_main_reprompts_k_after_one_preparation_and_reuses_validated(
        self,
    ) -> None:
        validated = kmeans_analysis.validate_features(
            prepared_pf_frame(),
            kmeans_analysis.select_features(prepared_pf_frame(), "PF"),
        )
        input_path = Path("prepared.csv")
        events: list[str] = []

        def scenario_prompt() -> str:
            events.append("scenario")
            return "PF"

        def k_prompt(n_samples: int | None = None) -> int:
            events.append(f"k:{n_samples}")
            return 9 if n_samples is None else 2

        def path_prompt(scenario: str) -> Path:
            events.append(f"path:{scenario}")
            return input_path

        def prepare(scenario: str, path: Path) -> kmeans_analysis.ValidatedData:
            events.append(f"prepare:{scenario}:{path}")
            return validated

        def run_validated(*args: object) -> Path:
            events.append("run")
            return Path("output")

        with patch.object(
            kmeans_analysis, "prompt_scenario", side_effect=scenario_prompt
        ) as mocked_scenario, patch.object(
            kmeans_analysis, "prompt_k", side_effect=k_prompt
        ) as mocked_k, patch.object(
            kmeans_analysis, "prompt_csv_path", side_effect=path_prompt
        ) as mocked_path, patch.object(
            kmeans_analysis, "prepare_data", side_effect=prepare
        ) as mocked_prepare, patch.object(
            kmeans_analysis,
            "_run_validated_analysis",
            side_effect=run_validated,
        ) as mocked_run, patch(
            "builtins.print"
        ):
            kmeans_analysis.main()

        self.assertEqual(
            events,
            [
                "scenario",
                "k:None",
                "path:PF",
                "prepare:PF:prepared.csv",
                f"k:{len(validated.features)}",
                "run",
            ],
        )
        mocked_scenario.assert_called_once_with()
        self.assertEqual(mocked_k.call_count, 2)
        mocked_path.assert_called_once_with("PF")
        mocked_prepare.assert_called_once_with("PF", input_path)
        mocked_run.assert_called_once()
        run_args = mocked_run.call_args.args
        self.assertEqual(run_args[:4], ("PF", 2, input_path, kmeans_analysis.OUTPUT_ROOT))
        self.assertIs(run_args[4], validated)

        source_lines = Path(kmeans_analysis.__file__).read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertEqual(
            source_lines[-2:],
            ['if __name__ == "__main__":', "    main()"],
        )


if __name__ == "__main__":
    unittest.main()
