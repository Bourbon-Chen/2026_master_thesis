#!/usr/bin/env python3
"""Run Version A and Version B with one shared interactive HDBSCAN configuration."""

from hdbscan_runner import main


if __name__ == "__main__":
    main("version_a", "version_b")
