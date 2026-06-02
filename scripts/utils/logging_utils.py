"""Runtime logging and human-readable progress reporting."""

from __future__ import annotations

import csv
from pathlib import Path


RUNTIME_LOG_FIELDS = (
    "scenario",
    "version",
    "min_cluster_size",
    "min_samples",
    "start_time",
    "end_time",
    "duration_seconds",
    "n_clusters",
    "noise_count",
    "noise_ratio",
    "status",
    "error_message",
)


def append_runtime_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RUNTIME_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RUNTIME_LOG_FIELDS})


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def print_progress(
    current: int,
    total: int,
    scenario: str,
    version: str,
    min_cluster_size: int,
    min_samples: int,
    elapsed_seconds: float,
    completed_durations: list[float],
) -> None:
    average = (
        sum(completed_durations) / len(completed_durations)
        if completed_durations
        else 0
    )
    remaining = average * (total - current + 1)
    print()
    print(f"[{current} / {total}] Progress: {current / total * 100:.1f}%")
    print(f"Scenario: {scenario}")
    print(f"Version: {version}")
    print(f"min_cluster_size: {min_cluster_size}")
    print(f"min_samples: {min_samples}")
    print(f"Elapsed Time: {format_duration(elapsed_seconds)}")
    print(f"Average Runtime: {format_duration(average)}")
    print(f"Estimated Remaining: {format_duration(remaining)}")
