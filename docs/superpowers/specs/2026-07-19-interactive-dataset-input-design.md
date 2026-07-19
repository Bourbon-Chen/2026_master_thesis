# Interactive Dataset Input Design

## Goal

Allow both `scripts/filter_high_risk_dataset.py` and
`scripts/pearson_correlation/correlation_analysis.py` to request the input CSV
path in the terminal instead of relying only on an embedded path, while
preserving each script's existing default path and all other behavior.

## Input behavior

- Keep `data/MT_UPDATE_MS_HEV_v2_NORcleaned.csv` as the default input path in
  both scripts.
- Support `-i/--input` in both scripts. Preserve the existing option in the
  filtering script and add it to the correlation script.
- If `-i/--input` is supplied, use that path without prompting.
- If `-i/--input` is omitted, display an English terminal prompt that includes
  the applicable default path.
- If the user presses Enter without typing a path, use the default path.
- If the user types a path, use that path instead of the default.

## Implementation

Add a small input-path prompt function that returns a `Path` to each script.
In the filtering script, change the existing command-line parser's input
default to `None`. In the correlation script, add a minimal command-line parser
with an optional `-i/--input` argument whose default is also `None`. Each
`main()` can then distinguish an explicit argument from an omitted option and
prompt only when needed. Resolve the selected path before loading the CSV.

The filtering script's output-path naming, PF/TF selection, threshold parsing,
CSV filtering, reporting, and explicit `-o/--output` behavior remain unchanged.
The correlation script's feature configuration, analysis logic, output
directory, threshold, reports, and charts remain unchanged. The user's existing
default-file update to `MT_UPDATE_MS_HEV_v2_NORcleaned.csv` in the filtering
script must be preserved.

## Error handling

Path opening and CSV validation continue to use each script's existing
behavior. The prompts accept surrounding whitespace and treat a blank value as
selection of the applicable default path.

## Verification

Verify these cases for both scripts:

1. Omitted `-i/--input` plus an empty terminal response selects the default path.
2. Omitted `-i/--input` plus a typed path selects that path.
3. An explicit `-i/--input` value bypasses the input-path prompt.
4. Existing filtering and correlation-analysis behavior remains intact.
