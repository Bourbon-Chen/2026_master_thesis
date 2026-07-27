from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scripts.cluster_preparation import prepare_clustering_inputs as prepare_inputs
from scripts.utils import clustering_preprocessing as preprocessing


class PrepareClusteringInputsTests(unittest.TestCase):
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
                "PF_constant": [0.0, 0.0, 0.0, 0.0],
                "TFABC_NOR": [0.1, 0.2, 0.3, 0.4],
            }
        )
        self.input_path = self.root / "already_high_risk.csv"
        self.source.to_csv(self.input_path, index=False)
        self.exclusion_path = self.root / "pf_exclusions.json"
        self.exclusion_path.write_text(
            json.dumps(
                {
                    "scenario": "PF",
                    "excluded_features": ["PFABC_NOR"],
                    "reason": "documented correlation review decision",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_run_preparation_saves_a_loaded_artifact_for_the_input_rows(self) -> None:
        """Removing the delegation or exclusion would change the saved artifact."""
        artifact_dir = prepare_inputs.run_preparation(
            input_path=self.input_path,
            scenario="PF",
            output_root=self.root / "outputs_preprocessing",
            exclusion_path=self.exclusion_path,
            risk_column="PF_Index_Risk_equal",
            risk_operator=">",
            risk_threshold=0.25,
        )

        loaded = preprocessing.load_prepared_artifact(artifact_dir)

        self.assertEqual(len(loaded.metadata), len(self.source))
        self.assertNotIn("PFABC_NOR", loaded.feature_names)
        self.assertEqual(loaded.config["structural_null"]["fill_value"], 0.0)
        self.assertEqual(
            loaded.config["risk_filter"],
            {"column": "PF_Index_Risk_equal", "operator": ">", "threshold": 0.25},
        )
        self.assertEqual(
            loaded.audit.loc[0, ["event", "column"]].tolist(),
            ["constant_feature_removed", "PF_constant"],
        )

    def test_load_exclusions_rejects_a_different_scenario(self) -> None:
        """Dropping scenario validation could apply PF research decisions to TF."""
        self.exclusion_path.write_text(
            json.dumps(
                {
                    "scenario": "TF",
                    "excluded_features": ["TFABC_NOR"],
                    "reason": "documented TF correlation review",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "scenario does not match"):
            prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_rejects_malformed_json(self) -> None:
        """Swallowing parse errors would hide a corrupted research decision artifact."""
        self.exclusion_path.write_text("{not valid JSON", encoding="utf-8")

        with self.assertRaises(json.JSONDecodeError):
            prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_rejects_duplicate_names(self) -> None:
        """Removing duplicate validation would make an exclusion artifact ambiguous."""
        self.exclusion_path.write_text(
            json.dumps(
                {
                    "scenario": "PF",
                    "excluded_features": ["PFABC_NOR", "PFABC_NOR"],
                    "reason": "documented correlation review decision",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_rejects_a_non_object_json_root(self) -> None:
        """Dropping root validation would turn malformed artifacts into attribute errors."""
        self.exclusion_path.write_text(
            json.dumps(["PFABC_NOR"]), encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "JSON object"):
            prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_requires_a_reason(self) -> None:
        """Accepting missing research rationale would make exclusions unauditable."""
        self.exclusion_path.write_text(
            json.dumps({"scenario": "PF", "excluded_features": ["PFABC_NOR"]}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "missing required keys: reason"):
            prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_rejects_non_string_or_blank_reason(self) -> None:
        """Removing reason validation would permit unusable research rationales."""
        for reason in (None, "   "):
            with self.subTest(reason=reason):
                self.exclusion_path.write_text(
                    json.dumps(
                        {
                            "scenario": "PF",
                            "excluded_features": ["PFABC_NOR"],
                            "reason": reason,
                        }
                    ),
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ValueError, "reason must be a non-empty string"):
                    prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_rejects_a_non_string_scenario(self) -> None:
        """Calling string methods on untrusted scenario values would leak AttributeError."""
        self.exclusion_path.write_text(
            json.dumps(
                {
                    "scenario": 1,
                    "excluded_features": ["PFABC_NOR"],
                    "reason": "documented correlation review decision",
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "scenario must be a string"):
            prepare_inputs.load_exclusions(self.exclusion_path, "PF")

    def test_load_exclusions_returns_names_for_a_complete_valid_artifact(self) -> None:
        """Rejecting a complete documented artifact would block valid research exclusions."""
        self.assertEqual(
            prepare_inputs.load_exclusions(self.exclusion_path, "PF"),
            ("PFABC_NOR",),
        )

    def test_load_exclusions_returns_no_names_when_the_artifact_is_absent(self) -> None:
        """Treating absent exclusions as an error would block a valid unexcluded run."""
        self.assertEqual(prepare_inputs.load_exclusions(None, "PF"), ())

    def test_run_preparation_rejects_any_input_row_outside_the_recorded_predicate(self) -> None:
        """Replacing validation with filtering would silently reduce the modelling cohort."""
        source = self.source.copy()
        source.loc[0, "PF_Index_Risk_equal"] = 0.25
        source.to_csv(self.input_path, index=False)

        with self.assertRaisesRegex(ValueError, "recorded risk comparison"):
            prepare_inputs.run_preparation(
                input_path=self.input_path,
                scenario="PF",
                output_root=self.root / "outputs_preprocessing",
                exclusion_path=None,
                risk_column="PF_Index_Risk_equal",
                risk_operator=">",
                risk_threshold=0.25,
            )

    def test_run_preparation_accepts_the_inclusive_recorded_predicate(self) -> None:
        """Implementing only strict comparison would reject valid inclusive inputs."""
        source = self.source.copy()
        source.loc[0, "PF_Index_Risk_equal"] = 0.25
        source.to_csv(self.input_path, index=False)

        artifact_dir = prepare_inputs.run_preparation(
            input_path=self.input_path,
            scenario="PF",
            output_root=self.root / "outputs_preprocessing",
            exclusion_path=None,
            risk_column="PF_Index_Risk_equal",
            risk_operator=">=",
            risk_threshold=0.25,
        )

        self.assertEqual(len(preprocessing.load_prepared_artifact(artifact_dir).metadata), 4)

    def test_run_preparation_rejects_an_unsupported_risk_operator(self) -> None:
        """Accepting another operator would misrepresent the persisted risk filter."""
        with self.assertRaisesRegex(ValueError, "risk operator"):
            prepare_inputs.run_preparation(
                input_path=self.input_path,
                scenario="PF",
                output_root=self.root / "outputs_preprocessing",
                exclusion_path=None,
                risk_column="PF_Index_Risk_equal",
                risk_operator="<",
                risk_threshold=0.25,
            )


if __name__ == "__main__":
    unittest.main()
