from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.cluster_preparation import prepare_clustering_inputs as preparation
from scripts.pearson_correlation import build_feature_exclusions as exclusions
from scripts.utils import clustering_preprocessing


class FeatureExclusionParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roles = pd.DataFrame(
            [
                {"column": "Per_extent", "role": "PF_FEATURE", "reason": ""},
                {"column": "PFAB2k_NOR", "role": "PF_FEATURE", "reason": ""},
                {"column": "Tem_extent", "role": "TF_FEATURE", "reason": ""},
                {"column": "fid", "role": "IDENTIFIER", "reason": ""},
                {
                    "column": "PFAB2k_lossR",
                    "role": "EXCLUDED",
                    "reason": "raw_ab_lossR_replaced_by_normalized_NOR",
                },
                {
                    "column": "PF_text",
                    "role": "NON_NUMERIC",
                    "reason": "active_scenario_feature_is_non_numeric",
                },
                {
                    "column": "TF_text",
                    "role": "NON_NUMERIC",
                    "reason": "active_scenario_feature_is_non_numeric",
                },
            ],
            columns=["column", "role", "reason"],
        )

    def test_parse_accepts_both_commas_and_preserves_first_occurrence(self) -> None:
        parsed = exclusions.parse_exclusion_text(
            " Per_extent，PFAB2k_NOR, ,Per_extent "
        )

        self.assertEqual(parsed, ("Per_extent", "PFAB2k_NOR"))

    def test_parse_blank_input_returns_an_empty_tuple(self) -> None:
        self.assertEqual(exclusions.parse_exclusion_text("   "), ())

    def test_validation_accepts_only_exact_active_scenario_features(self) -> None:
        self.assertEqual(
            exclusions.validate_exclusions(
                self.roles, "pf", ("Per_extent", "PFAB2k_NOR")
            ),
            ("Per_extent", "PFAB2k_NOR"),
        )
        self.assertEqual(
            exclusions.validate_exclusions(
                self.roles, "tf", ("Tem_extent",)
            ),
            ("Tem_extent",),
        )

        with self.assertRaisesRegex(
            exclusions.ExclusionValidationError,
            "absent from feature_roles.csv.*per_extent",
        ):
            exclusions.validate_exclusions(self.roles, "PF", ("per_extent",))

    def test_validation_reports_all_invalid_categories_in_one_error(self) -> None:
        with self.assertRaises(exclusions.ExclusionValidationError) as raised:
            exclusions.validate_exclusions(
                self.roles,
                "PF",
                (
                    "not_present",
                    "Tem_extent",
                    "fid",
                    "PFAB2k_lossR",
                    "PF_text",
                ),
            )

        message = str(raised.exception)
        self.assertIn("absent from feature_roles.csv: not_present", message)
        self.assertIn("opposite scenario: Tem_extent", message)
        self.assertIn("protected or non-feature: fid", message)
        self.assertIn(
            "already excluded by a built-in rule: PFAB2k_lossR", message
        )
        self.assertIn(
            "already excluded and should not be added", message
        )
        self.assertIn(
            "non-numeric active-scenario candidate: PF_text", message
        )

    def test_validation_reports_opposite_non_numeric_candidates_for_pf_and_tf(
        self,
    ) -> None:
        for scenario, name in (("PF", "TF_text"), ("TF", "PF_text")):
            with self.subTest(scenario=scenario):
                with self.assertRaises(exclusions.ExclusionValidationError) as raised:
                    exclusions.validate_exclusions(self.roles, scenario, (name,))

                self.assertIn(f"opposite scenario: {name}", str(raised.exception))

    def test_roles_loader_requires_file_columns_and_unique_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing_path = root / "missing.csv"
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                exclusions.load_feature_roles(missing_path)

            malformed_path = root / "malformed.csv"
            pd.DataFrame({"column": ["Per_extent"]}).to_csv(
                malformed_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "role, reason"):
                exclusions.load_feature_roles(malformed_path)

            duplicate_path = root / "duplicate.csv"
            pd.concat([self.roles.iloc[[0]], self.roles.iloc[[0]]]).to_csv(
                duplicate_path, index=False
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                exclusions.load_feature_roles(duplicate_path)


class FeatureExclusionCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.roles_path = self.root / "feature_roles.csv"
        pd.DataFrame(
            [
                {"column": "Per_extent", "role": "PF_FEATURE", "reason": ""},
                {"column": "PFAB2k_NOR", "role": "PF_FEATURE", "reason": ""},
                {"column": "Tem_extent", "role": "TF_FEATURE", "reason": ""},
            ]
        ).to_csv(self.roles_path, index=False)
        self.output_path = self.root / "feature_exclusions.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_default_paths_are_scenario_specific_and_scenario_is_case_insensitive(self) -> None:
        pf_roles, pf_output = exclusions.default_paths("pf")
        tf_roles, tf_output = exclusions.default_paths("TF")

        self.assertEqual(
            pf_roles,
            exclusions.PROJECT_ROOT / "outputs_correlation/pf/feature_roles.csv",
        )
        self.assertEqual(
            pf_output,
            exclusions.PROJECT_ROOT
            / "outputs_correlation/pf/feature_exclusions.json",
        )
        self.assertEqual(
            tf_roles,
            exclusions.PROJECT_ROOT / "outputs_correlation/tf/feature_roles.csv",
        )
        self.assertEqual(
            tf_output,
            exclusions.PROJECT_ROOT
            / "outputs_correlation/tf/feature_exclusions.json",
        )

    def test_omitted_exclude_prompts_once_and_writes_exact_minimal_json(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch(
                "builtins.input",
                return_value=" Per_extent，PFAB2k_NOR,Per_extent ",
            ) as prompt,
            redirect_stdout(stdout),
        ):
            result = exclusions.main(
                [
                    "--scenario",
                    "pf",
                    "--roles",
                    str(self.roles_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        prompt.assert_called_once_with(exclusions.EXCLUSION_PROMPT)
        self.assertEqual(result, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(list(payload), ["scenario", "excluded_features", "reason"])
        self.assertEqual(
            payload,
            {
                "scenario": "PF",
                "excluded_features": ["Per_extent", "PFAB2k_NOR"],
                "reason": exclusions.EXCLUSION_REASON,
            },
        )
        self.assertIn("Selected exclusions: Per_extent, PFAB2k_NOR", stdout.getvalue())
        self.assertIn("prepare_clustering_inputs.py --exclusions", stdout.getvalue())

    def test_blank_noninteractive_exclude_does_not_prompt(self) -> None:
        with mock.patch(
            "builtins.input",
            side_effect=AssertionError("non-interactive input must not prompt"),
        ):
            result = exclusions.main(
                [
                    "--scenario", "PF", "--exclude", "", "--roles",
                    str(self.roles_path), "--output", str(self.output_path),
                ]
            )

        self.assertEqual(result, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_features"], [])

    def test_blank_prompt_response_writes_an_empty_exclusion_list(self) -> None:
        with mock.patch("builtins.input", return_value="   ") as prompt:
            result = exclusions.main(
                [
                    "--scenario", "PF", "--roles", str(self.roles_path),
                    "--output", str(self.output_path),
                ]
            )

        self.assertEqual(result, 0)
        prompt.assert_called_once_with(exclusions.EXCLUSION_PROMPT)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_features"], [])

    def test_existing_output_is_rejected_before_prompt_without_overwrite(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        stderr = io.StringIO()
        with (
            mock.patch("builtins.input") as prompt,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            exclusions.main(
                [
                    "--scenario", "PF", "--roles", str(self.roles_path),
                    "--output", str(self.output_path),
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        prompt.assert_not_called()
        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "old decision")
        self.assertIn("--overwrite", stderr.getvalue())

    def test_invalid_interactive_input_prompts_once_and_writes_nothing(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch("builtins.input", return_value="Tem_extent") as prompt,
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            exclusions.main(
                [
                    "--scenario", "PF", "--roles", str(self.roles_path),
                    "--output", str(self.output_path),
                ]
            )

        self.assertEqual(raised.exception.code, 2)
        prompt.assert_called_once_with(exclusions.EXCLUSION_PROMPT)
        self.assertFalse(self.output_path.exists())
        self.assertIn("opposite scenario: Tem_extent", stderr.getvalue())

    def test_overwrite_replaces_the_existing_decision(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        result = exclusions.main(
            [
                "--scenario", "PF", "--exclude", "Per_extent", "--roles",
                str(self.roles_path), "--output", str(self.output_path),
                "--overwrite",
            ]
        )

        self.assertEqual(result, 0)
        payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["excluded_features"], ["Per_extent"])

    def test_interrupted_temp_write_preserves_existing_decision(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        payload = exclusions.build_payload("PF", ("Per_extent",))
        with mock.patch.object(exclusions.json, "dump", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                exclusions.write_exclusions_json(self.output_path, payload, overwrite=True)

        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "old decision")
        self.assertEqual(list(self.root.glob(f".{self.output_path.name}.*.tmp")), [])

    def test_validation_failure_with_overwrite_preserves_existing_decision(self) -> None:
        self.output_path.write_text("old decision", encoding="utf-8")
        with self.assertRaises(exclusions.ExclusionValidationError):
            exclusions.generate_exclusions(
                scenario="PF", exclusion_text="Tem_extent", roles_path=self.roles_path,
                output_path=self.output_path, overwrite=True,
            )

        self.assertEqual(self.output_path.read_text(encoding="utf-8"), "old decision")

    def test_direct_script_help_can_import_project_packages(self) -> None:
        script_path = Path(exclusions.__file__)
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"], cwd=self.root,
            capture_output=True, text=True, check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--scenario", completed.stdout)
        self.assertIn("--exclude", completed.stdout)
        self.assertIn("--overwrite", completed.stdout)


class FeatureExclusionPreparationIntegrationTests(unittest.TestCase):
    def test_generated_json_is_loaded_and_excludes_the_selected_prepared_feature(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            roles_path = root / "feature_roles.csv"
            pd.DataFrame(
                [
                    {"column": "Per_extent", "role": "PF_FEATURE", "reason": ""},
                    {"column": "PFABC_NOR", "role": "PF_FEATURE", "reason": ""},
                    {
                        "column": "PF_Index_Risk_equal",
                        "role": "COMPOSITE_INDEX",
                        "reason": "",
                    },
                ]
            ).to_csv(roles_path, index=False)
            exclusions_path = root / "feature_exclusions.json"
            exclusions.generate_exclusions(
                scenario="PF",
                exclusion_text="PFABC_NOR",
                roles_path=roles_path,
                output_path=exclusions_path,
            )

            self.assertEqual(
                preparation.load_exclusions(exclusions_path, "PF"),
                ("PFABC_NOR",),
            )

            input_path = root / "high_risk.csv"
            pd.DataFrame(
                {
                    "fid": [10, 11, 12, 13],
                    "MS_ID": ["s0", "s1", "s2", "s3"],
                    "PF_Index_Risk_equal": [0.3, 0.4, 0.5, 0.6],
                    "Per_extent": [-1.0, 0.0, 1.0, np.nan],
                    "PFABC_NOR": [-0.8, -0.2, 0.2, 0.8],
                    "TFABC_NOR": [0.1, 0.2, 0.3, 0.4],
                }
            ).to_csv(input_path, index=False)
            artifact_directory = preparation.run_preparation(
                input_path=input_path,
                scenario="PF",
                output_root=root / "prepared",
                exclusion_path=exclusions_path,
                risk_column="PF_Index_Risk_equal",
                risk_operator=">",
                risk_threshold=0.25,
            )
            artifact = clustering_preprocessing.load_prepared_artifact(
                artifact_directory
            )

        self.assertEqual(artifact.feature_names, ("Per_extent",))
        self.assertNotIn("PFABC_NOR", artifact.standardized_features.columns)
        self.assertEqual(len(artifact.standardized_features), 4)
        self.assertFalse(artifact.standardized_features.isna().any().any())
        recorded = artifact.excluded_features.set_index("column")
        self.assertEqual(
            recorded.loc["PFABC_NOR", "reason"],
            "explicit_research_exclusion",
        )
