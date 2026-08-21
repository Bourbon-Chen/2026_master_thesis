from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.clustering import hdbscan_runner
from scripts import run_hdbscan_street_typology as legacy_runner
from scripts.utils.clustering_preprocessing import (
    load_prepared_artifact,
    prepare_clustering_data,
    save_prepared_artifact,
)


def source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fid": [10, 11, 12, 13],
            "MS_ID": ["s0", "s1", "s2", "s3"],
            "PF_Index_Risk_equal": [0.2, 0.3, 0.4, 0.5],
            "Per_extent": [-1.0, np.nan, 0.5, 1.0],
            "PFABC_NOR": [-0.8, -0.2, 0.4, 0.9],
        }
    )


class PreparedHdbscanInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        source = source_frame()
        source_path = self.root / "source.csv"
        source.to_csv(source_path, index=False)
        self.artifact_dir = save_prepared_artifact(
            prepare_clustering_data(source, "PF"),
            source_path,
            self.root / "prepared",
            risk_filter={"minimum_risk": 0.0},
        )
        self.artifact = load_prepared_artifact(self.artifact_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_versions_select_the_committed_prepared_matrices(self) -> None:
        primary = hdbscan_runner.load_hdbscan_input(self.artifact_dir, "version_b")
        sensitivity = hdbscan_runner.load_hdbscan_input(
            self.artifact_dir, "version_a"
        )

        np.testing.assert_allclose(
            primary.values,
            self.artifact.standardized_features.to_numpy(dtype=float),
        )
        np.testing.assert_allclose(
            sensitivity.values,
            self.artifact.filled_features.to_numpy(dtype=float),
        )
        self.assertEqual(primary.scale_label, "standardized_primary")
        self.assertEqual(sensitivity.scale_label, "filled_raw_scale_sensitivity")

    def test_run_experiments_uses_prepared_values_without_local_transforms(self) -> None:
        class FakeHdbscan:
            seen_values: list[np.ndarray] = []

            def __init__(self, **_kwargs: object) -> None:
                pass

            def fit_predict(self, values: np.ndarray) -> np.ndarray:
                self.seen_values.append(values.copy())
                return np.array([0, 0, 1, 1])

        def forbidden(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("HDBSCAN must not transform prepared input")

        args = SimpleNamespace(
            prepared=self.artifact_dir,
            output_root=self.root / "output",
            min_cluster_size=2,
            min_samples=1,
            audit_only=False,
        )
        with patch.object(hdbscan_runner, "parse_args", return_value=args), patch.object(
            hdbscan_runner,
            "load_prepared_artifact",
            return_value=self.artifact,
        ), patch.object(
            hdbscan_runner, "import_sklearn", return_value=(SimpleNamespace(__version__="test"), FakeHdbscan)
        ), patch.object(hdbscan_runner, "prepare_features", side_effect=forbidden, create=True), patch.object(
            pd.DataFrame, "fillna", side_effect=forbidden
        ), patch(
            "sklearn.preprocessing.StandardScaler.fit", side_effect=forbidden
        ), patch("builtins.print"):
            hdbscan_runner.run_experiments(("version_a", "version_b"))

        self.assertEqual(len(FakeHdbscan.seen_values), 2)
        np.testing.assert_allclose(
            FakeHdbscan.seen_values[0],
            self.artifact.filled_features.to_numpy(dtype=float),
        )
        np.testing.assert_allclose(
            FakeHdbscan.seen_values[1],
            self.artifact.standardized_features.to_numpy(dtype=float),
        )
        metadata_path = (
            args.output_root
            / "clustering_results"
            / "PF"
            / "version_b"
            / "min_cluster_size_2_min_samples_1"
            / "cluster_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["scale_label"], "standardized_primary")
        self.assertEqual(
            metadata["structural_null_policy"],
            "structural_zero_from_prepared_artifact",
        )
        self.assertEqual(
            metadata["feature_order"], list(self.artifact.feature_names)
        )
        audit = pd.read_csv(metadata_path.with_name("preprocessing_audit.csv"))
        self.assertEqual(
            json.loads(audit.loc[0, "algorithm_parameters"]),
            {
                "min_cluster_size": 2,
                "min_samples": 1,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            },
        )
        labels = pd.read_csv(metadata_path.with_name("cluster_labels.csv"))
        self.assertEqual(labels["source_row_position"].tolist(), [0, 1, 2, 3])

    def test_existing_hdbscan_run_destination_is_rejected_before_fit(self) -> None:
        """A pre-existing run directory must never be mixed with new artifacts."""
        output_root = self.root / "output"
        destination = (
            output_root
            / "clustering_results"
            / "PF"
            / "version_b"
            / "min_cluster_size_2_min_samples_1"
        )
        destination.mkdir(parents=True)
        sentinel = destination / "existing.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        args = SimpleNamespace(
            prepared=self.artifact_dir,
            output_root=output_root,
            min_cluster_size=2,
            min_samples=1,
            audit_only=False,
        )

        with patch.object(hdbscan_runner, "parse_args", return_value=args), patch.object(
            hdbscan_runner, "import_sklearn", side_effect=AssertionError("must not fit")
        ), patch("builtins.print"):
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                hdbscan_runner.run_experiments(("version_b",))

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_parameter_specific_audit_is_immutable(self) -> None:
        """A second audit write must not overwrite a prior parameter record."""
        hdbscan_input = hdbscan_runner.load_hdbscan_input(
            self.artifact_dir, "version_b"
        )
        parameters = {
            "min_cluster_size": 2,
            "min_samples": 1,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        }
        output_root = self.root / "output"

        hdbscan_runner.save_audit(
            hdbscan_input, "version_b", output_root, parameters
        )

        with self.assertRaisesRegex(FileExistsError, "audit directory already exists"):
            hdbscan_runner.save_audit(
                hdbscan_input, "version_b", output_root, parameters
            )

    def test_cli_no_longer_accepts_overwrite(self) -> None:
        """The removed overwrite flag would allow destructive recomputation."""
        with patch("sys.argv", ["hdbscan_runner.py", "--prepared", str(self.artifact_dir), "--overwrite"]):
            with self.assertRaises(SystemExit) as raised:
                hdbscan_runner.parse_args(("version_b",))

        self.assertEqual(raised.exception.code, 2)


class LegacyWrapperTests(unittest.TestCase):
    def test_legacy_wrapper_requires_prepared_artifact(self) -> None:
        with patch("sys.argv", ["run_hdbscan_street_typology.py"]):
            with self.assertRaises(SystemExit) as raised:
                legacy_runner.parse_args()

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
