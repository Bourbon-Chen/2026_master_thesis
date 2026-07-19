import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FILTER = load_module("filter_high_risk_dataset", "scripts/filter_high_risk_dataset.py")
CORRELATION = load_module(
    "correlation_analysis",
    "scripts/pearson_correlation/correlation_analysis.py",
)


class InteractiveInputTests(unittest.TestCase):
    def assert_prompt_behavior(self, module, default_constant):
        self.assertTrue(
            hasattr(module, "prompt_input_path"),
            "script must define prompt_input_path",
        )
        with patch("builtins.input", return_value="") as mocked_input:
            self.assertEqual(module.prompt_input_path(default_constant), default_constant)
        self.assertEqual(
            mocked_input.call_args.args[0],
            f"Enter input CSV path [{default_constant}]: ",
        )

        custom = Path("data/custom.csv")
        with patch("builtins.input", return_value=f"  {custom}  "):
            self.assertEqual(module.prompt_input_path(default_constant), custom)

    def assert_argument_behavior(self, module):
        self.assertTrue(
            hasattr(module, "parse_args"),
            "script must define parse_args",
        )
        with patch.object(sys, "argv", ["script.py"]):
            self.assertIsNone(module.parse_args().input)
        with patch.object(sys, "argv", ["script.py", "--input", "data/custom.csv"]):
            self.assertEqual(module.parse_args().input, Path("data/custom.csv"))

    def test_filter_input_behavior(self):
        self.assertEqual(
            FILTER.DEFAULT_INPUT.name,
            "MT_UPDATE_MS_HEV_v2_NORcleaned.csv",
        )
        self.assert_prompt_behavior(FILTER, FILTER.DEFAULT_INPUT)
        self.assert_argument_behavior(FILTER)

    def test_correlation_input_behavior(self):
        self.assertEqual(
            CORRELATION.INPUT_FILE.name,
            "MT_UPDATE_MS_HEV_v2_NORcleaned.csv",
        )
        self.assert_prompt_behavior(CORRELATION, CORRELATION.INPUT_FILE)
        self.assert_argument_behavior(CORRELATION)

    def test_correlation_scenario_prompt_retries_and_normalizes(self):
        self.assertTrue(
            hasattr(CORRELATION, "prompt_scenario"),
            "script must define prompt_scenario",
        )
        with patch("builtins.input", side_effect=["invalid", "tf"]) as mocked_input:
            with patch("builtins.print") as mocked_print:
                self.assertEqual(CORRELATION.prompt_scenario(), "TF")

        self.assertEqual(mocked_input.call_count, 2)
        mocked_input.assert_called_with("Enter correlation scenario (pf/tf): ")
        mocked_print.assert_called_once_with("Scenario must be 'pf' or 'tf'.")

    def test_correlation_scenario_prompt_accepts_pf_and_tf_case_insensitively(self):
        for entered, expected in (("pf", "PF"), ("PF", "PF"), ("tf", "TF"), ("TF", "TF")):
            with self.subTest(entered=entered):
                with patch("builtins.input", return_value=entered):
                    self.assertEqual(CORRELATION.prompt_scenario(), expected)

    def test_correlation_main_runs_only_selected_scenario(self):
        data = CORRELATION.pd.DataFrame()
        result = {
            "features": [],
            "high_pairs": [],
            "within_high_pairs": [],
            "cross_high_pairs": [],
        }
        arguments = [
            "correlation_analysis.py",
            "--input",
            "data/custom.csv",
        ]

        for scenario in ("PF", "TF"):
            with self.subTest(scenario=scenario):
                summary = Mock()
                selected = {
                    scenario: CORRELATION.SCENARIO_FEATURE_GROUPS[scenario]
                }
                with TemporaryDirectory() as temporary_directory:
                    output_directory = Path(temporary_directory)
                    with patch.object(sys, "argv", arguments):
                        with patch.object(
                            CORRELATION, "OUTPUT_DIRECTORY", output_directory
                        ):
                            with patch.object(
                                CORRELATION,
                                "prompt_scenario",
                                return_value=scenario,
                            ) as mocked_scenario_prompt:
                                with patch.object(
                                    CORRELATION, "load_data", return_value=data
                                ) as mocked_load:
                                    with patch.object(
                                        CORRELATION, "validate_columns"
                                    ) as mocked_validate:
                                        with patch.object(
                                            CORRELATION,
                                            "analyze_scenario",
                                            return_value=result,
                                        ) as mocked_analyze:
                                            with patch.object(
                                                CORRELATION,
                                                "create_combined_summary",
                                                return_value=summary,
                                            ) as mocked_summary:
                                                with patch.object(
                                                    CORRELATION, "write_text_report"
                                                ) as mocked_report:
                                                    with patch("builtins.print"):
                                                        CORRELATION.main()

                    mocked_scenario_prompt.assert_called_once_with()
                    mocked_load.assert_called_once_with(
                        Path("data/custom.csv").resolve()
                    )
                    mocked_validate.assert_called_once_with(data, selected)
                    mocked_analyze.assert_called_once_with(
                        data,
                        scenario,
                        CORRELATION.SCENARIO_FEATURE_GROUPS[scenario],
                        output_directory / scenario,
                        CORRELATION.CORRELATION_THRESHOLD,
                        CORRELATION.EXPECTED_RANGES,
                    )
                    mocked_summary.assert_called_once_with({scenario: result})
                    summary.to_csv.assert_called_once_with(
                        output_directory
                        / "combined_summary"
                        / "correlation_summary.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    mocked_report.assert_called_once_with(
                        {scenario: result},
                        selected,
                        output_directory
                        / "combined_summary"
                        / "correlation_report.txt",
                        CORRELATION.CORRELATION_THRESHOLD,
                    )

    def test_filter_omitted_input_uses_prompt_result(self):
        custom = Path("data/custom.csv")
        arguments = [
            "filter_high_risk_dataset.py",
            "--scenario",
            "pf",
            "--threshold",
            "0.25",
        ]
        with patch.object(sys, "argv", arguments):
            with patch.object(
                FILTER, "prompt_input_path", return_value=custom
            ) as mocked_prompt:
                with patch.object(FILTER, "filter_dataset") as mocked_filter:
                    FILTER.main()

        mocked_prompt.assert_called_once_with()
        self.assertEqual(mocked_filter.call_args.args[0], custom.resolve())

    def test_correlation_omitted_input_uses_prompt_result(self):
        custom = Path("data/custom.csv")
        stop_after_load = RuntimeError("stop after input selection")
        with patch.object(sys, "argv", ["correlation_analysis.py"]):
            with patch.object(
                CORRELATION, "prompt_input_path", return_value=custom
            ) as mocked_prompt:
                with patch.object(CORRELATION, "prompt_scenario", return_value="PF"):
                    with patch.object(
                        CORRELATION,
                        "load_data",
                        side_effect=stop_after_load,
                    ) as mocked_load:
                        with patch("builtins.print"):
                            with self.assertRaisesRegex(
                                RuntimeError, "stop after input selection"
                            ):
                                CORRELATION.main()

        mocked_prompt.assert_called_once_with()
        mocked_load.assert_called_once_with(custom.resolve())

    def test_filter_explicit_input_bypasses_prompt(self):
        custom = Path("data/custom.csv")
        arguments = [
            "filter_high_risk_dataset.py",
            "--input",
            str(custom),
            "--scenario",
            "pf",
            "--threshold",
            "0.25",
        ]
        with patch.object(sys, "argv", arguments):
            with patch.object(
                FILTER,
                "prompt_input_path",
                side_effect=AssertionError("input prompt must be bypassed"),
            ):
                with patch.object(FILTER, "filter_dataset") as mocked_filter:
                    FILTER.main()

        self.assertEqual(mocked_filter.call_args.args[0], custom.resolve())

    def test_correlation_explicit_input_bypasses_prompt(self):
        custom = Path("data/custom.csv")
        stop_after_load = RuntimeError("stop after input selection")
        arguments = ["correlation_analysis.py", "--input", str(custom)]
        with patch.object(sys, "argv", arguments):
            with patch.object(
                CORRELATION,
                "prompt_input_path",
                side_effect=AssertionError("input prompt must be bypassed"),
            ):
                with patch.object(CORRELATION, "prompt_scenario", return_value="PF"):
                    with patch.object(
                        CORRELATION,
                        "load_data",
                        side_effect=stop_after_load,
                    ) as mocked_load:
                        with patch("builtins.print"):
                            with self.assertRaisesRegex(
                                RuntimeError, "stop after input selection"
                            ):
                                CORRELATION.main()

        mocked_load.assert_called_once_with(custom.resolve())


if __name__ == "__main__":
    unittest.main()
