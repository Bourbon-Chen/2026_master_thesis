import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


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
