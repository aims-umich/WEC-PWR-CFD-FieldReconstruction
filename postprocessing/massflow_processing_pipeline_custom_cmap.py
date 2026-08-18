#!/usr/bin/env python3
"""Backward-compatible wrapper for legacy custom-cmap pipeline entrypoint.

This script now delegates to massflow_processing_pipeline.py with --custom-cmap.
"""

from __future__ import annotations

import sys

from massflow_processing_pipeline import main as pipeline_main


if __name__ == "__main__":
    if "--custom-cmap" not in sys.argv:
        sys.argv.append("--custom-cmap")
    pipeline_main()
