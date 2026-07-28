from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.utils import clustering_preprocessing as preprocessing


class FeatureDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "fid": [1, 2, 3],
                "MS_ID": ["a", "b", "c"],
                "PF_Index_Risk_equal": [0.4, 0.5, 0.6],
                "Per_extent": [0.0, 0.2, 0.4],
                "N_PFlossR_2kiw": [-1.0, np.nan, 1.0],
                "PFResident_lossR": [0.0, -0.2, -0.4],
                "PFABC_NOR": [0.1, 0.2, 0.3],
                "Tem_extent": [0.3, 0.2, 0.1],
                "TFABC_NOR": [0.4, 0.5, 0.6],
                "Street_type_NOR": [0.2, 0.3, 0.4],
                "PF_result_label": ["x", "y", "z"],
                "unrelated_pf_text": [1.0, 2.0, 3.0],
            }
        )

    def test_classification_is_anchored_case_insensitive_and_role_complete(self):
        discovery = preprocessing.discover_scenario_features(self.data, "pf")

        self.assertEqual(discovery.scenario, "PF")
        self.assertEqual(
            discovery.feature_names,
            ("Per_extent", "N_PFlossR_2kiw", "PFResident_lossR", "PFABC_NOR"),
        )
        roles = discovery.column_roles.set_index("column")["role"].to_dict()
        self.assertEqual(roles["fid"], "IDENTIFIER")
        self.assertEqual(roles["PF_Index_Risk_equal"], "COMPOSITE_INDEX")
        self.assertEqual(roles["Tem_extent"], "TF_FEATURE")
        self.assertEqual(roles["Street_type_NOR"], "SHARED_FEATURE")
        self.assertEqual(roles["PF_result_label"], "RESULT_OR_LABEL")
        self.assertEqual(roles["unrelated_pf_text"], "EXCLUDED")

    def test_explicit_exclusions_must_exist_and_belong_to_active_scenario(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            preprocessing.discover_scenario_features(
                self.data, "PF", exclusions=("PF_missing",)
            )
        with self.assertRaisesRegex(ValueError, "TFABC_NOR"):
            preprocessing.discover_scenario_features(
                self.data, "PF", exclusions=("TFABC_NOR",)
            )

    def test_raw_ab_lossr_is_builtin_excluded_but_normalized_ab_is_retained(self):
        raw_names = (
            "PFAB2k_lossR",
            "N_PFAB2k_lossR",
            "PFAB5k_lossR",
            "N_PFAB5k_lossR",
            "TFAB2k_lossR",
            "N_TFAB2k_lossR",
            "TFAB5k_lossR",
            "N_TFAB5k_lossR",
        )
        normalized_names = (
            "PFAB2k_NOR",
            "PFAB5k_NOR",
            "TFAB2k_NOR",
            "TFAB5k_NOR",
        )
        data = self.data.copy()
        for offset, name in enumerate(raw_names + normalized_names):
            data[name] = [-0.8 + offset * 0.01, 0.0, 0.8 - offset * 0.01]
        data["PFAB2k_lossR_extra"] = [-0.5, 0.0, 0.5]

        pf = preprocessing.discover_scenario_features(data, "PF")
        tf = preprocessing.discover_scenario_features(data, "TF")
        pf_roles = pf.column_roles.set_index("column")
        tf_roles = tf.column_roles.set_index("column")

        for name in raw_names:
            self.assertEqual(pf_roles.loc[name, "role"], "EXCLUDED")
            self.assertEqual(
                pf_roles.loc[name, "reason"],
                "raw_ab_lossR_replaced_by_normalized_NOR",
            )
            self.assertEqual(tf_roles.loc[name, "role"], "EXCLUDED")
        self.assertIn("PFAB2k_NOR", pf.feature_names)
        self.assertIn("PFAB5k_NOR", pf.feature_names)
        self.assertIn("TFAB2k_NOR", tf.feature_names)
        self.assertIn("TFAB5k_NOR", tf.feature_names)
        self.assertIn("PFAB2k_lossR_extra", pf.feature_names)

    def test_normalize_scenario_rejects_noncanonical_numeric_values(self):
        with self.assertRaises(ValueError):
            preprocessing.normalize_scenario(True)
        with self.assertRaises(ValueError):
            preprocessing.normalize_scenario(1.0)


class PrepareClusteringDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = pd.DataFrame(
            {
                "fid": [10, 11, 12, 13],
                "MS_ID": ["s0", "s1", "s2", "s3"],
                "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6],
                "Per_extent": [-1.0, 0.0, 1.0, np.nan],
                "PFABC_NOR": [-0.8, -0.2, 0.2, 0.8],
                "PF_constant": [np.nan, 0.0, np.nan, 0.0],
                "TFABC_NOR": [0.1, 0.2, 0.3, 0.4],
            }
        )

    def test_structural_zero_preserves_rows_extremes_and_source(self):
        original = self.source.copy(deep=True)

        prepared = preprocessing.prepare_clustering_data(self.source, "PF")

        pd.testing.assert_frame_equal(self.source, original)
        self.assertEqual(prepared.feature_names, ("Per_extent", "PFABC_NOR"))
        self.assertEqual(prepared.metadata["source_row_position"].tolist(), [0, 1, 2, 3])
        self.assertEqual(
            prepared.filled_features["Per_extent"].tolist(), [-1.0, 0.0, 1.0, 0.0]
        )
        self.assertEqual(prepared.missing_counts["Per_extent"], 1)
        self.assertEqual(
            prepared.rows_with_structural_null.tolist(), [False, False, False, True]
        )
        np.testing.assert_allclose(
            prepared.standardized_features.mean(axis=0).to_numpy(),
            np.zeros(2),
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            prepared.standardized_features.std(axis=0, ddof=0).to_numpy(),
            np.ones(2),
            atol=1e-10,
            rtol=1e-10,
        )
        recovered = preprocessing.inverse_transform_frame(
            prepared.standardized_features,
            prepared.scaler_parameters,
        )
        pd.testing.assert_frame_equal(
            recovered,
            prepared.filled_features,
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_constant_feature_has_one_excluded_row_without_its_active_role(self):
        """Leaving the discovery row duplicates a constant feature in the audit."""
        prepared = preprocessing.prepare_clustering_data(self.source, "PF")

        constant_rows = prepared.excluded_columns.loc[
            prepared.excluded_columns["column"].eq("PF_constant")
        ]
        self.assertEqual(len(constant_rows), 1)
        self.assertEqual(constant_rows.iloc[0]["role"], "EXCLUDED")
        self.assertEqual(
            constant_rows.iloc[0]["reason"], "constant_after_structural_zero"
        )

    def test_semantic_preparation_fills_nulls_without_scaling_or_constant_removal(self):
        original_source = self.source.copy(deep=True)
        self.source.index = [14, 9, 33, 2]
        original_source.index = self.source.index

        with mock.patch.object(
            preprocessing,
            "StandardScaler",
            side_effect=AssertionError("semantic preparation must not scale"),
        ) as scaler:
            prepared = preprocessing.prepare_semantic_features(self.source, "PF")

        scaler.assert_not_called()
        pd.testing.assert_frame_equal(self.source, original_source)
        self.assertEqual(
            prepared.feature_names,
            ("Per_extent", "PFABC_NOR", "PF_constant"),
        )
        self.assertEqual(prepared.original_features.index.tolist(), [14, 9, 33, 2])
        self.assertTrue(np.isnan(prepared.original_features.loc[2, "Per_extent"]))
        self.assertEqual(prepared.filled_features.loc[2, "Per_extent"], 0.0)
        self.assertEqual(
            prepared.filled_features["PF_constant"].tolist(),
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(prepared.missing_counts.to_dict(), {
            "Per_extent": 1,
            "PFABC_NOR": 0,
            "PF_constant": 2,
        })
        self.assertEqual(prepared.row_null_counts.tolist(), [1, 0, 1, 1])
        self.assertEqual(
            prepared.rows_with_structural_null.tolist(),
            [True, False, True, True],
        )

    def test_invalid_semantic_values_fail_instead_of_becoming_zero(self):
        base = pd.DataFrame(
            {
                "fid": [1, 2],
                "MS_ID": ["a", "b"],
                "PF_Index_Risk_equal": [0.4, 0.5],
                "PFABC_NOR": [0.0, 0.5],
            }
        )
        cases = {
            "non-numeric": ["bad", "0.5"],
            "infinity": [0.0, np.inf],
            "outside": [0.0, 1.01],
        }
        for message, values in cases.items():
            with self.subTest(message=message):
                candidate = base.copy()
                candidate["PFABC_NOR"] = values
                with self.assertRaisesRegex(ValueError, message):
                    preprocessing.prepare_clustering_data(candidate, "PF")

    def test_explicit_exclusion_precedes_non_numeric_active_candidate_rejection(self):
        source = self.source.assign(PF_malformed=["bad", "bad", "bad", "bad"])

        prepared = preprocessing.prepare_clustering_data(
            source, "PF", exclusions=("PF_malformed",)
        )

        excluded = prepared.excluded_columns.set_index("column")
        self.assertEqual(excluded.loc["PF_malformed", "reason"], "explicit_research_exclusion")
        with self.assertRaisesRegex(ValueError, "non-numeric.*PF_malformed"):
            preprocessing.prepare_clustering_data(source, "PF")

    def test_requires_nonempty_input_and_metadata_columns(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            preprocessing.prepare_clustering_data(self.source.iloc[0:0], "PF")
        for required_name in ("fid", "MS_ID", "PF_Index_Risk_equal"):
            with self.subTest(required_name=required_name):
                with self.assertRaisesRegex(ValueError, required_name):
                    preprocessing.prepare_clustering_data(
                        self.source.drop(columns=[required_name]), "PF"
                    )

    def test_excluded_or_post_fill_constant_features_fail_when_none_remain(self):
        with self.assertRaisesRegex(ValueError, "No usable PF features"):
            preprocessing.prepare_clustering_data(
                self.source.drop(columns=["PFABC_NOR", "PF_constant"]),
                "PF",
                exclusions=("Per_extent",),
            )
        with self.assertRaisesRegex(ValueError, "No usable PF features remain"):
            preprocessing.prepare_clustering_data(
                self.source.drop(columns=["Per_extent", "PFABC_NOR"]), "PF"
            )

    def test_metadata_preserves_source_order_and_structural_null_audit(self):
        source = self.source.copy()
        source.index = [14, 9, 33, 2]

        prepared = preprocessing.prepare_clustering_data(source, "PF")

        self.assertEqual(prepared.metadata.index.tolist(), [14, 9, 33, 2])
        self.assertEqual(prepared.metadata["source_row_position"].tolist(), [0, 1, 2, 3])
        self.assertEqual(prepared.metadata["structural_null_count"].tolist(), [0, 0, 0, 1])
        self.assertEqual(prepared.metadata["had_structural_null"].tolist(), [False, False, False, True])

    def test_repeated_preparation_is_deterministic_and_never_uses_opposite_scenario(self):
        first = preprocessing.prepare_clustering_data(self.source, "PF")
        second = preprocessing.prepare_clustering_data(self.source, "PF")

        self.assertNotIn("TFABC_NOR", first.feature_names)
        pd.testing.assert_frame_equal(first.filled_features, second.filled_features)
        pd.testing.assert_frame_equal(first.standardized_features, second.standardized_features)
        pd.testing.assert_frame_equal(first.scaler_parameters, second.scaler_parameters)

    def test_inverse_transform_requires_exact_feature_order(self):
        prepared = preprocessing.prepare_clustering_data(self.source, "PF")
        reordered = prepared.standardized_features.loc[:, ["PFABC_NOR", "Per_extent"]]

        with self.assertRaisesRegex(ValueError, "feature order"):
            preprocessing.inverse_transform_frame(reordered, prepared.scaler_parameters)


class PreparedArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = pd.DataFrame(
            {
                "fid": [10, 11, 12, 13],
                "MS_ID": ["s0", "s1", "s2", "s3"],
                "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6],
                "Per_extent": [-1.0, 0.0, 1.0, np.nan],
                "PFABC_NOR": [-0.8, -0.2, 0.2, 0.8],
                "TFABC_NOR": [0.1, 0.2, 0.3, 0.4],
            }
        )
        self.source_path = self.root / "high_risk.csv"
        self.source.to_csv(self.source_path, index=False)
        self.prepared = preprocessing.prepare_clustering_data(self.source, "PF")
        self.risk_filter = {
            "column": "PF_Index_Risk_equal",
            "operator": ">",
            "threshold": 0.25,
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def save_artifact(self, output_name: str = "outputs") -> Path:
        return preprocessing.save_prepared_artifact(
            self.prepared,
            source_path=self.source_path,
            output_root=self.root / output_name,
            risk_filter=self.risk_filter,
            exclusion_path=None,
        )

    @staticmethod
    def read_config(artifact_dir: Path) -> dict[str, object]:
        return json.loads(
            (artifact_dir / "preprocessing_config.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def write_config(artifact_dir: Path, config: dict[str, object]) -> None:
        (artifact_dir / "preprocessing_config.json").write_text(
            json.dumps(config, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def test_round_trip_preserves_matrix_order_and_complete_provenance(self):
        artifact_dir = self.save_artifact()
        loaded = preprocessing.load_prepared_artifact(artifact_dir)

        self.assertEqual(
            {path.name for path in artifact_dir.iterdir()},
            {
                "metadata.csv",
                "features_original.csv",
                "features_filled.csv",
                "features_standardized.csv",
                "selected_features.csv",
                "excluded_features.csv",
                "scaler_parameters.csv",
                "preprocessing_audit.csv",
                "preprocessing_config.json",
            },
        )
        self.assertEqual(loaded.feature_names, self.prepared.feature_names)
        pd.testing.assert_frame_equal(
            loaded.standardized_features,
            self.prepared.standardized_features.reset_index(drop=True),
            check_exact=False,
            atol=1e-14,
            rtol=1e-14,
        )
        expected_config_keys = {
            "source_path",
            "source_sha256",
            "source_row_count",
            "scenario",
            "risk_filter",
            "marker_rules",
            "exclusion_path",
            "exclusion_sha256",
            "feature_order",
            "feature_order_sha256",
            "structural_null",
            "scaler",
            "matrix_sha256",
            "artifact_sha256",
            "versions",
            "created_at_utc",
        }
        self.assertEqual(set(loaded.config), expected_config_keys)
        self.assertEqual(loaded.config["structural_null"]["fill_value"], 0.0)
        self.assertEqual(
            loaded.config["matrix_sha256"],
            preprocessing.sha256_file(artifact_dir / "features_standardized.csv"),
        )
        self.assertEqual(
            set(loaded.config["artifact_sha256"]),
            {
                "metadata.csv",
                "features_original.csv",
                "features_filled.csv",
                "features_standardized.csv",
                "selected_features.csv",
                "excluded_features.csv",
                "scaler_parameters.csv",
                "preprocessing_audit.csv",
            },
        )

    def test_each_serialized_artifact_file_tampering_is_rejected_by_sha256(self):
        """Removing any per-file integrity hash would silently trust altered data."""
        artifact_dir = self.save_artifact()

        for filename in sorted(loaded_name for loaded_name in preprocessing.ARTIFACT_FILENAMES if loaded_name != "preprocessing_config.json"):
            with self.subTest(filename=filename):
                candidate = artifact_dir / filename
                original = candidate.read_bytes()
                try:
                    candidate.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        preprocessing.load_prepared_artifact(artifact_dir)
                finally:
                    candidate.write_bytes(original)

    def test_artifact_semantics_are_checked_even_when_hash_is_updated(self):
        """A forged filled-file hash must not permit a broken zero-fill contract."""
        artifact_dir = self.save_artifact()
        filled_path = artifact_dir / "features_filled.csv"
        filled = pd.read_csv(filled_path)
        filled.iloc[0, 0] = 0.75
        filled.to_csv(filled_path, index=False)
        config = self.read_config(artifact_dir)
        config["artifact_sha256"]["features_filled.csv"] = preprocessing.sha256_file(
            filled_path
        )
        self.write_config(artifact_dir, config)

        with self.assertRaisesRegex(ValueError, "zero-fill"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_forged_scaler_and_standardized_hashes_cannot_bypass_scaler_semantics(self):
        """Trusting only their mutual transform permits forged StandardScaler state."""
        artifact_dir = self.save_artifact()
        scaler_path = artifact_dir / "scaler_parameters.csv"
        matrix_path = artifact_dir / "features_standardized.csv"
        filled = pd.read_csv(artifact_dir / "features_filled.csv")
        scaler = pd.read_csv(scaler_path)
        scaler["mean"] += 0.25
        forged_matrix = (filled - scaler["mean"].to_numpy()) / scaler[
            "scale"
        ].to_numpy()
        scaler.to_csv(scaler_path, index=False)
        forged_matrix.to_csv(matrix_path, index=False)
        config = self.read_config(artifact_dir)
        config["artifact_sha256"]["scaler_parameters.csv"] = preprocessing.sha256_file(
            scaler_path
        )
        config["artifact_sha256"]["features_standardized.csv"] = (
            preprocessing.sha256_file(matrix_path)
        )
        config["matrix_sha256"] = preprocessing.sha256_file(matrix_path)
        self.write_config(artifact_dir, config)

        with self.assertRaisesRegex(ValueError, "scaler parameters"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_forged_row_alignment_cannot_detach_artifact_from_source(self):
        """Internally consistent row swaps must still match the recorded source."""
        artifact_dir = self.save_artifact()
        filenames = (
            "metadata.csv",
            "features_original.csv",
            "features_filled.csv",
            "features_standardized.csv",
        )
        for filename in filenames:
            path = artifact_dir / filename
            frame = pd.read_csv(path)
            frame.iloc[[0, 1]] = frame.iloc[[1, 0]].to_numpy()
            if filename == "metadata.csv":
                frame["source_row_position"] = np.arange(len(frame))
            frame.to_csv(path, index=False)

        config = self.read_config(artifact_dir)
        for filename in filenames:
            config["artifact_sha256"][filename] = preprocessing.sha256_file(
                artifact_dir / filename
            )
        config["matrix_sha256"] = config["artifact_sha256"][
            "features_standardized.csv"
        ]
        self.write_config(artifact_dir, config)

        with self.assertRaisesRegex(ValueError, "recorded source"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_near_constant_nonconstant_feature_round_trips(self):
        """Tiny non-zero variance must use StandardScaler's actual scale."""
        source = self.source.copy()
        source["PFABC_NOR"] = 0.5 + np.arange(len(source)) * 1e-10
        source_path = self.root / "near_constant.csv"
        source.to_csv(source_path, index=False)
        prepared = preprocessing.prepare_clustering_data(source, "PF")
        artifact_dir = preprocessing.save_prepared_artifact(
            prepared,
            source_path=source_path,
            output_root=self.root / "near_constant_outputs",
            risk_filter=self.risk_filter,
        )

        loaded = preprocessing.load_prepared_artifact(artifact_dir)

        np.testing.assert_allclose(
            loaded.standardized_features,
            prepared.standardized_features,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_existing_scenario_directory_is_immutable(self):
        artifact_dir = self.save_artifact()

        with self.assertRaisesRegex(FileExistsError, "immutable"):
            preprocessing.save_prepared_artifact(
                self.prepared,
                self.source_path,
                self.root / "outputs",
                self.risk_filter,
            )

        self.assertTrue(artifact_dir.is_dir())
        self.assertEqual(len(list(artifact_dir.iterdir())), 9)

    def test_standardized_matrix_tampering_is_rejected(self):
        artifact_dir = self.save_artifact()
        matrix_path = artifact_dir / "features_standardized.csv"
        matrix = pd.read_csv(matrix_path)
        matrix.iloc[0, 0] += 1.0
        matrix.to_csv(matrix_path, index=False)

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_missing_artifact_file_is_rejected(self):
        artifact_dir = self.save_artifact()
        (artifact_dir / "preprocessing_audit.csv").unlink()

        with self.assertRaisesRegex(ValueError, "missing"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_invalid_scenario_is_rejected(self):
        artifact_dir = self.save_artifact()
        config = self.read_config(artifact_dir)
        config["scenario"] = "XX"
        self.write_config(artifact_dir, config)

        with self.assertRaisesRegex(ValueError, "scenario"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_selected_feature_order_must_match_config(self):
        artifact_dir = self.save_artifact()
        selected_path = artifact_dir / "selected_features.csv"
        selected = pd.read_csv(selected_path)
        selected.iloc[::-1].to_csv(selected_path, index=False)

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_feature_csv_column_order_must_match_config(self):
        artifact_dir = self.save_artifact()
        original_path = artifact_dir / "features_original.csv"
        original = pd.read_csv(original_path)
        original.loc[:, list(reversed(original.columns))].to_csv(
            original_path, index=False
        )

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_metadata_and_feature_row_counts_must_match(self):
        artifact_dir = self.save_artifact()
        metadata_path = artifact_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        metadata.iloc[:-1].to_csv(metadata_path, index=False)

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_source_row_positions_must_be_contiguous(self):
        artifact_dir = self.save_artifact()
        metadata_path = artifact_dir / "metadata.csv"
        metadata = pd.read_csv(metadata_path)
        metadata["source_row_position"] += 1
        metadata.to_csv(metadata_path, index=False)

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_feature_order_hash_must_match_config(self):
        artifact_dir = self.save_artifact()
        config = self.read_config(artifact_dir)
        config["feature_order_sha256"] = "0" * 64
        self.write_config(artifact_dir, config)

        with self.assertRaisesRegex(ValueError, "feature-order SHA-256"):
            preprocessing.load_prepared_artifact(artifact_dir)

    def test_non_finite_standardized_values_are_rejected_after_hash_validation(self):
        artifact_dir = self.save_artifact()
        matrix_path = artifact_dir / "features_standardized.csv"
        matrix = pd.read_csv(matrix_path)
        matrix.iloc[0, 0] = np.inf
        matrix.to_csv(matrix_path, index=False)
        config = self.read_config(artifact_dir)
        config["matrix_sha256"] = preprocessing.sha256_file(matrix_path)
        config["artifact_sha256"]["features_standardized.csv"] = (
            preprocessing.sha256_file(matrix_path)
        )
        self.write_config(artifact_dir, config)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            preprocessing.load_prepared_artifact(artifact_dir)
