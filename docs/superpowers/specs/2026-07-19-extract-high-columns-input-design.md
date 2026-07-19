# Extract High Columns Interactive Input Design

## Goal

Allow `scripts/pearson_correlation/extract_high_columns.py` to prompt for its
input CSV path when `-i/--input` is omitted, while preserving the existing
default path and all extraction behavior.

## Input behavior

- Keep `data/MT_UPDATE_MS_HEV_v2_NORcleaned.csv` as `DEFAULT_INPUT`.
- Preserve `-i/--input`; an explicit value bypasses the prompt.
- When the option is omitted, show the English prompt
  `Enter input CSV path [<default path>]: `.
- Trim surrounding whitespace from interactive input.
- An empty response selects `DEFAULT_INPUT`; a non-empty response is passed to
  the existing `resolve_input_path()` behavior.

## Implementation

Add `prompt_input_path(default_path: Path = DEFAULT_INPUT) -> Path`. Change the
argument parser's input default to `None` so `main()` can distinguish an
omitted option from an explicit value. Select the raw path in `main()` and then
continue through the existing `resolve_input_path()` validation.

Scenario selection, removal-column prompts and validation, relative-path
resolution, extraction logic, output paths, and reporting remain unchanged.

## Verification

- Empty interactive input returns the default path.
- A typed interactive path is used after whitespace trimming.
- Explicit `-i/--input` bypasses the prompt.
- Omitted `-i/--input` uses the prompt result.
- Existing extraction tests continue to pass.
