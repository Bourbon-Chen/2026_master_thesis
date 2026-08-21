#!/usr/bin/env python3
"""Deprecated compatibility entry point for prepared-artifact HDBSCAN."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.clustering import hdbscan_runner
except ModuleNotFoundError:  # Direct execution keeps scripts/ on sys.path.
    from clustering import hdbscan_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper for the modular prepared-artifact HDBSCAN runner."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Directory containing the prepared clustering artifact.",
    )
    return parser.parse_known_args()[0]


def main() -> None:
    parse_args()
    print(
        "DEPRECATED: run_hdbscan_street_typology.py now delegates to the modular\n"
        "prepared-artifact HDBSCAN runner. Median imputation is no longer supported."
    )
    hdbscan_runner.main("version_a", "version_b")


if __name__ == "__main__":
    main()
