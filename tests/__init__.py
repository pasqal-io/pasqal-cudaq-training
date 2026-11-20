# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Make the Slurm workflow scripts importable for the tests."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "04-slurm"))
