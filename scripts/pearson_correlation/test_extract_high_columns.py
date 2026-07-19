from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.pearson_correlation import extract_high_columns as ehc


SCRIPT_PATH = Path(__file__).with_name("extract_high_columns.py")


class ExtractHighColumnsScriptTests(unittest.TestCase):
    def test_script_exists_at_requested_path(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file())

    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "fid": [1, 2],
                "MS_ID": ["a", "b"],
                "Street_type_NOR": [0.1, 0.2],
                "Index_population": [0.3, 0.4],
                "Per_extent": [0.5, 0.6],
                "Tem_extent": [0.7, 0.8],
                "PFAB2k_NOR": [0.1, 0.2],
                "PFAB5k_NOR": [0.2, 0.3],
                "PFPRM_lossR": [-0.1, -0.2],
                "N_PFlossR_2ktw": [-0.3, -0.4],
                "PF_Index_population": [0.4, 0.5],
                "PF_Index_Risk_equal": [0.5, 0.6],
                "TFAB2k_NOR": [0.6, 0.7],
                "TFAB5k_NOR": [0.7, 0.8],
                "TFPRM_lossR": [-0.5, -0.6],
                "N_TFlossR_2ktw": [-0.7, -0.8],
                "TF_Index_population": [0.8, 0.9],
                "TF_Index_Risk_equal": [0.9, 1.0],
            }
        )

    def test_pf_defaults_remove_selected_pf_and_all_tf_columns(self) -> None:
        result, removed = ehc.filter_dataset(
            self.data,
            "pf",
            ehc.SCENARIO_CONFIG["pf"]["default_removals"],
        )

        self.assertEqual(
            list(result.columns),
            [
                "fid",
                "MS_ID",
                "Street_type_NOR",
                "Index_population",
                "Per_extent",
                "PFAB5k_NOR",
                "PF_Index_population",
                "PF_Index_Risk_equal",
            ],
        )
        self.assertIn("Tem_extent", removed)
        self.assertIn("TF_Index_population", removed)
        self.assertIn("PFAB2k_NOR", removed)

    def test_tf_defaults_are_the_tf_equivalents_and_remove_all_pf_columns(self) -> None:
        result, removed = ehc.filter_dataset(
            self.data,
            "tf",
            ehc.SCENARIO_CONFIG["tf"]["default_removals"],
        )

        self.assertEqual(
            list(result.columns),
            [
                "fid",
                "MS_ID",
                "Street_type_NOR",
                "Index_population",
                "Tem_extent",
                "TFAB5k_NOR",
                "TF_Index_population",
                "TF_Index_Risk_equal",
            ],
        )
        self.assertIn("Per_extent", removed)
        self.assertIn("PF_Index_population", removed)
        self.assertIn("TFAB2k_NOR", removed)

    def test_scenario_detection_does_not_remove_incidental_letter_pairs(self) -> None:
        data = self.data.assign(
            platform_score=[0.1, 0.2],
            mapFeature_score=[0.3, 0.4],
        )

        pf_result, _ = ehc.filter_dataset(data, "pf", [])
        tf_result, _ = ehc.filter_dataset(data, "tf", [])

        self.assertIn("platform_score", pf_result.columns)
        self.assertIn("mapFeature_score", tf_result.columns)

    def test_custom_removal_input_accepts_ascii_and_chinese_commas(self) -> None:
        parsed = ehc.parse_removal_input(
            " PFAB5k_NOR， PFPRM_lossR, N_PFlossR_2kiw "
        )

        self.assertEqual(
            parsed,
            ["PFAB5k_NOR", "PFPRM_lossR", "N_PFlossR_2kiw"],
        )

    def test_custom_removal_replaces_defaults(self) -> None:
        result, removed = ehc.filter_dataset(
            self.data, "pf", ["PFAB5k_NOR"]
        )

        self.assertNotIn("PFAB5k_NOR", result.columns)
        self.assertIn("PFAB5k_NOR", removed)
        self.assertIn("PFAB2k_NOR", result.columns)

    def test_removal_validation_protects_identifiers_and_index_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected"):
            ehc.validate_removal_columns(
                self.data.columns,
                "pf",
                ["fid", "PF_Index_population"],
            )

    def test_removal_validation_rejects_unknown_or_opposite_scenario_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "not removable"):
            ehc.validate_removal_columns(
                self.data.columns,
                "pf",
                ["does_not_exist", "TFAB5k_NOR"],
            )

    def test_filter_dataset_requires_fid_and_ms_id(self) -> None:
        for missing_identifier in ehc.IDENTIFIER_COLUMNS:
            with self.subTest(missing_identifier=missing_identifier):
                data = self.data.drop(columns=[missing_identifier])
                with self.assertRaisesRegex(
                    ValueError, "Missing required identifier columns"
                ):
                    ehc.filter_dataset(data, "pf", [])

    def test_prompt_scenario_retries_until_pf_or_tf_is_entered(self) -> None:
        with patch("builtins.input", side_effect=["other", " TF "]):
            self.assertEqual(ehc.prompt_scenario(), "tf")

    def test_cli_scenario_accepts_uppercase(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["extract_high_columns.py", "--scenario", "PF"],
        ):
            self.assertEqual(ehc.parse_args().scenario, "pf")

    def test_input_prompt_uses_default_or_typed_path(self) -> None:
        self.assertTrue(
            hasattr(ehc, "prompt_input_path"),
            "script must define prompt_input_path",
        )
        with patch("builtins.input", return_value="") as mocked_input:
            self.assertEqual(ehc.prompt_input_path(), ehc.DEFAULT_INPUT)
        mocked_input.assert_called_once_with(
            f"Enter input CSV path [{ehc.DEFAULT_INPUT}]: "
        )

        with patch("builtins.input", return_value="  custom.csv  "):
            self.assertEqual(ehc.prompt_input_path(), Path("custom.csv"))

    def test_input_argument_is_optional_and_accepts_path(self) -> None:
        with patch.object(sys, "argv", ["extract_high_columns.py"]):
            self.assertIsNone(ehc.parse_args().input)
        with patch.object(
            sys,
            "argv",
            ["extract_high_columns.py", "--input", "custom.csv"],
        ):
            self.assertEqual(ehc.parse_args().input, Path("custom.csv"))

    def test_main_prompts_when_input_is_omitted(self) -> None:
        prompted_path = Path("prompted.csv")
        resolved_path = Path("resolved.csv").resolve()
        arguments = [
            "extract_high_columns.py",
            "--scenario",
            "pf",
            "--remove",
            "PFAB5k_NOR",
        ]
        with patch.object(sys, "argv", arguments):
            with patch.object(
                ehc,
                "prompt_input_path",
                return_value=prompted_path,
                create=True,
            ) as mocked_prompt:
                with patch.object(
                    ehc, "resolve_input_path", return_value=resolved_path
                ) as mocked_resolve:
                    with patch.object(
                        ehc.pd, "read_csv", return_value=self.data.iloc[:0]
                    ):
                        with patch.object(ehc, "extract_dataset") as mocked_extract:
                            ehc.main()

        mocked_prompt.assert_called_once_with()
        mocked_resolve.assert_called_once_with(prompted_path)
        mocked_extract.assert_called_once_with(
            resolved_path,
            "pf",
            ["PFAB5k_NOR"],
            ehc.DEFAULT_OUTPUT_ROOT,
        )

    def test_main_explicit_input_bypasses_prompt(self) -> None:
        cli_path = Path("cli.csv")
        resolved_path = Path("resolved.csv").resolve()
        arguments = [
            "extract_high_columns.py",
            "--input",
            str(cli_path),
            "--scenario",
            "pf",
            "--remove",
            "PFAB5k_NOR",
        ]
        with patch.object(sys, "argv", arguments):
            with patch.object(
                ehc,
                "prompt_input_path",
                side_effect=AssertionError("input prompt must be bypassed"),
                create=True,
            ):
                with patch.object(
                    ehc, "resolve_input_path", return_value=resolved_path
                ) as mocked_resolve:
                    with patch.object(
                        ehc.pd, "read_csv", return_value=self.data.iloc[:0]
                    ):
                        with patch.object(ehc, "extract_dataset") as mocked_extract:
                            ehc.main()

        mocked_resolve.assert_called_once_with(cli_path)
        mocked_extract.assert_called_once_with(
            resolved_path,
            "pf",
            ["PFAB5k_NOR"],
            ehc.DEFAULT_OUTPUT_ROOT,
        )

    def test_extract_dataset_writes_scenario_specific_for_pca_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.csv"
            output_root = root / "outputs_correlation"
            self.data.to_csv(input_path, index=False)

            for scenario in ("pf", "tf"):
                with self.subTest(scenario=scenario):
                    defaults = ehc.SCENARIO_CONFIG[scenario][
                        "default_removals"
                    ]
                    expected, _ = ehc.filter_dataset(
                        self.data, scenario, defaults
                    )
                    output_path = ehc.extract_dataset(
                        input_path,
                        scenario,
                        defaults,
                        output_root,
                    )
                    saved = pd.read_csv(output_path)

                    self.assertEqual(
                        output_path,
                        output_root
                        / scenario
                        / f"{scenario}_for_PCA.csv",
                    )
                    pd.testing.assert_frame_equal(
                        saved, expected, check_dtype=False
                    )


if __name__ == "__main__":
    unittest.main()
