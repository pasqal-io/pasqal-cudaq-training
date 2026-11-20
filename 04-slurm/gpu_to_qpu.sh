#!/bin/bash
# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
# GPU_REQUEST, QPU_REQUEST and ANALYSIS_REQUEST contain sbatch options without shell quoting.
# Usage: QPU_REQUEST='--resources=PASQAL_LOCAL:1' bash 04-slurm/gpu_to_qpu.sh program.json runs/demo
set -euo pipefail
program=$(realpath "${1:?usage: gpu_to_qpu.sh PROGRAM NEW_OUTPUT_DIRECTORY}")
output=$(realpath -m "${2:?provide a new output directory}")
read -r -a gpu_options <<< "${GPU_REQUEST:---gres=gpu:1}"
read -r -a qpu_options <<< "${QPU_REQUEST:?set the site QPU sbatch options}"
read -r -a analysis_options <<< "${ANALYSIS_REQUEST:-}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$(dirname "$output")"
mkdir "$output"
cp "$program" "$output/program.json"

gpu=$(sbatch --parsable "${gpu_options[@]}" --job-name=cudaq-gpu \
    --output="$output/gpu.out" 04-slurm/run.sbatch "$output/program.json" \
    --emulate --output "$output/gpu.json")
printf '%s\n' "$gpu" > "$output/job-ids.txt"
printf 'Submitted GPU job %s; results: %s\n' "$gpu" "$output"
qpu=$(sbatch --parsable "${qpu_options[@]}" --job-name=cudaq-qpu \
    --dependency="afterok:$gpu" --kill-on-invalid-dep=yes --output="$output/qpu.out" \
    04-slurm/run.sbatch "$output/program.json" --output "$output/qpu.json")
printf '%s\n' "$qpu" >> "$output/job-ids.txt"
printf 'Submitted QPU job %s\n' "$qpu"
analysis=$(sbatch --parsable "${analysis_options[@]}" --job-name=cudaq-analysis --dependency="afterok:$qpu" \
    --kill-on-invalid-dep=yes --output="$output/analysis.out" \
    04-slurm/analyze.sbatch "$output/gpu.json" "$output/qpu.json" "$output/comparison.png")
printf '%s\n' "$analysis" >> "$output/job-ids.txt"
printf 'GPU %s -> QPU %s -> analysis %s\nResults: %s\n' "$gpu" "$qpu" "$analysis" "$output"
