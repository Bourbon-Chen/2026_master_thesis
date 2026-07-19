# Interactive Dataset Input Design

## Goal

Allow `scripts/filter_high_risk_dataset.py` to request the input CSV path in the terminal instead of relying only on an embedded path, while preserving the existing default path and all other behavior.

## Input behavior

- Keep `data/MT_UPDATE_MS_HEV_v2_NORcleaned.csv` as the default input path.
- If `-i/--input` is supplied, use that path without prompting.
- If `-i/--input` is omitted, display an English terminal prompt that includes the default path.
- If the user presses Enter without typing a path, use the default path.
- If the user types a path, use that path instead of the default.

## Implementation

Add a small input-path prompt function that returns a `Path`. Change the command-line parser's input default to `None` so `main()` can distinguish an explicit `-i/--input` value from an omitted option. Resolve the selected path exactly as the script currently does.

The output-path naming, PF/TF selection, threshold parsing, CSV filtering, reporting, and explicit `-o/--output` behavior remain unchanged. The user's existing default-file update to `MT_UPDATE_MS_HEV_v2_NORcleaned.csv` must be preserved.

## Error handling

Path opening and CSV validation continue to use the script's existing behavior. The prompt accepts surrounding whitespace and treats a blank value as selection of the default path.

## Verification

Verify these cases:

1. Omitted `-i/--input` plus an empty terminal response selects the default path.
2. Omitted `-i/--input` plus a typed path selects that path.
3. An explicit `-i/--input` value bypasses the input-path prompt.
4. Existing output-path, scenario, threshold, and filtering behavior remains intact.
