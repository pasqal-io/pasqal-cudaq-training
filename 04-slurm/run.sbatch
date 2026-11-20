#!/bin/bash
# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
# Run from the repository root with the Python environment already loaded.
#SBATCH --ntasks=1
set -euo pipefail
srun python 04-slurm/run.py "$@"
