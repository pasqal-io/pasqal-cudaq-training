#!/bin/bash
# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
#SBATCH --ntasks=1
set -euo pipefail
srun python 04-slurm/compare.py "$1" "$2" --plot "$3"
