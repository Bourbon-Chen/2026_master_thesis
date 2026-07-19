# Single Correlation Scenario Design

## Goal

Change `scripts/pearson_correlation/correlation_analysis.py` so every run asks
the user to choose PF or TF and analyzes only the selected scenario.

## Interaction

- After selecting the input path, show an English prompt:
  `Enter correlation scenario (pf/tf): `.
- Accept PF or TF case-insensitively and return the normalized uppercase value.
- For any other value, print an English validation message and prompt again.
- Always ask for the scenario; do not add a command-line option that bypasses
  this prompt.

## Analysis flow

- Build a one-entry scenario mapping for the selected scenario.
- Validate identifiers and feature columns for the selected scenario only.
- Call the existing `analyze_scenario()` flow once with the selected scenario's
  existing feature groups, threshold, expected ranges, and output directory.
- Build the existing combined summary and text report from the one selected
  result. These files therefore contain one scenario for the current run.
- Do not delete or overwrite the other scenario's existing output directory.

## Compatibility

Input-path prompting and `-i/--input` behavior remain unchanged. Correlation
calculations, quality checks, tables, charts, naming, thresholds, and output
formats remain unchanged.

## Verification

- PF, lowercase or uppercase, selects only PF.
- TF, lowercase or uppercase, selects only TF.
- Invalid values retry with an English error message.
- `main()` validates and analyzes only the selected scenario.
- Existing correlation and interactive-input tests continue to pass.
