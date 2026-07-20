# Choose K Scenario Selection Design

## Goal

Make `scripts/cluster_preparation/choose_k.py` support the same scenario-selection workflow in terminals, automation, and IDE run configurations:

- accept `--scenario PF` or `--scenario TF`;
- accept lowercase `--scenario pf` or `--scenario tf`;
- prompt for PF or TF when `--scenario` is omitted.

## Selected Approach

Use one normalization function for both command-line and interactive input. Configure argparse with `type=normalize_scenario`, retain the explicit PF/TF choices, and change the argument default from the hard-coded PF scenario to `None`. Add a retrying `prompt_scenario()` function. In `main()`, use the normalized command-line value when present and otherwise call the prompt.

This preserves non-interactive automation while making direct script execution interactive. It also prevents command-line and prompt behavior from diverging.

## Data Flow

1. `parse_args()` reads `--scenario`.
2. If supplied, argparse normalizes whitespace and letter case before validating the value against PF and TF.
3. If omitted, `main()` calls `prompt_scenario()` until the user enters PF or TF, case-insensitively.
4. The resulting uppercase scenario selects the existing path in `DEFAULT_INPUT_PATHS` and the existing feature/metadata configuration in `SCENARIO_CONFIG`.
5. The remaining K-evaluation workflow is unchanged.

## Error Handling

- Invalid command-line values are rejected by argparse with a usage error.
- Invalid interactive values print a concise message and prompt again.
- Existing file, column, numeric-data, K-range, and clustering validation remains unchanged.

## Cleanup

Remove the unused PF-only `INPUT_FILE` constant and the obsolete hard-coded `SCENARIO` default so the configuration does not imply that PF is still selected automatically.

## Tests

Add focused tests proving that:

- command-line PF and TF values are accepted;
- lowercase command-line values normalize to uppercase;
- omitting `--scenario` returns `None` from parsing;
- the interactive prompt retries after invalid input and normalizes valid input;
- `main()` prompts only when the command-line scenario is absent;
- `main()` does not prompt when the scenario is supplied.

Run the complete `scripts.cluster_preparation.test_choose_k` test module after the focused red-green cycle.

## Non-goals

Do not change clustering features, K-means parameters, evaluation metrics, recommendation logic, input CSV contents, or output directory structure.
