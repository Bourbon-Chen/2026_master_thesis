"""Configuration for the modular HDBSCAN research experiment pipeline."""

from pathlib import Path


# This is an unsupervised clustering experiment.
# It explores whether street typologies emerge from HEV indicators. It is not
# supervised training, and rule-based labels are excluded from clustering.
CONDA_ENVIRONMENT = "2026master"
SCENARIOS = ("PF", "TF")
METRIC = "euclidean"
CLUSTER_SELECTION_METHOD = "eom"
STRUCTURAL_NULL_POLICY = "structural_zero_from_prepared_artifact"

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_ROOT.parent
INPUT_CSV = PROJECT_ROOT / "data" / "MT_UPDATE_MS_HEV_NORcleaned.csv"
PIPELINE_ROOT = PROJECT_ROOT / "outputs_hdbscan" / "pipeline_v1"
OUTPUT_ROOT = PIPELINE_ROOT

ID_COLUMN_CANDIDATES = ("fid", "MS_ID")
RULE_LABEL_CANDIDATES = (
    "rule_type",
    "type_label",
    "typology",
    "intervention_type",
)
EXCLUDED_FEATURE_KEYWORDS = (
    "score_b1",
    "score_b2",
    "label",
    "intervention",
    "result",
)


def dataset_name_from_input(input_path: Path) -> str:
    return input_path.stem


def dataset_output_root(input_path: Path, pipeline_root: Path = PIPELINE_ROOT) -> Path:
    return pipeline_root / dataset_name_from_input(input_path)


def clustering_results_root(output_root: Path) -> Path:
    return output_root / "clustering_results"


def logs_root(output_root: Path) -> Path:
    return output_root / "logs"


def tables_root(output_root: Path) -> Path:
    return output_root / "tables"


def visualizations_root(output_root: Path) -> Path:
    return output_root / "visualizations"


def run_name(min_cluster_size: int, min_samples: int) -> str:
    return f"min_cluster_size_{min_cluster_size}_min_samples_{min_samples}"


def run_dir(
    output_root: Path,
    scenario: str,
    version: str,
    min_cluster_size: int,
    min_samples: int,
) -> Path:
    return (
        clustering_results_root(output_root)
        / scenario
        / version
        / run_name(min_cluster_size, min_samples)
    )
