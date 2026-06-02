"""Filesystem and serialization helpers for reproducible experiment outputs."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import INPUT_CSV, dataset_output_root


def ensure_pipeline_dirs(output_root: Path) -> None:
    for relative in (
        "clustering_results",
        "visualizations/pca",
        "visualizations/maps",
        "visualizations/future_visualizations",
        "tables/audits",
        "tables/summaries",
        "tables/comparisons",
        "tables/noise_analysis",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def resolve_input_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")
    return resolved


def prompt_input_path(default: Path = INPUT_CSV) -> Path:
    raw = input(f"Enter input file path [default: {default}]: ").strip()
    return resolve_input_path(Path(raw) if raw else default)


def resolve_dataset_output_root(
    input_path: Path, pipeline_root: Path
) -> Path:
    return dataset_output_root(input_path, pipeline_root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(content, destination, indent=2, ensure_ascii=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def write_last_version_a_config(output_root: Path, config: dict[str, Any]) -> Path:
    path = output_root / "logs" / "last_version_a_config.json"
    write_json(path, config)
    return path


def read_last_version_a_config(output_root: Path) -> dict[str, Any]:
    path = output_root / "logs" / "last_version_a_config.json"
    if not path.exists():
        raise FileNotFoundError(
            "No previous Version A configuration found. "
            f"Run run_hdbscan_version_a.py first for this dataset: {path}"
        )
    return read_json(path)


def build_label_table(
    data: pd.DataFrame, labels, scenario: str, version: str
) -> pd.DataFrame:
    if "MS_ID" in data.columns:
        street_id = data["MS_ID"]
    elif "fid" in data.columns:
        street_id = data["fid"]
    else:
        street_id = pd.Series(data.index, index=data.index)
    output = pd.DataFrame({"street_id": street_id})
    for column in ("fid", "MS_ID"):
        if column in data.columns:
            output[column] = data[column]
    output["scenario"] = scenario
    output["version"] = version
    output["cluster_id"] = labels
    return output


def python_runtime_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
