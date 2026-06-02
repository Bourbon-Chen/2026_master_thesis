"""Discovery helpers for modular clustering result files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.io_utils import read_json


def discover_metadata(
    output_root: Path,
    scenario: Optional[str] = None,
    version: Optional[str] = None,
) -> list[tuple[Path, dict]]:
    results = []
    root = output_root / "clustering_results"
    if not root.exists():
        return results
    for path in sorted(root.rglob("cluster_metadata.json")):
        metadata = read_json(path)
        if scenario and metadata["scenario"] != scenario:
            continue
        if version and metadata["version"] != version:
            continue
        results.append((path, metadata))
    return results
