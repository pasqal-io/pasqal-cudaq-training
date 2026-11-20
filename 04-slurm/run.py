# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Execute a prepared Rydberg program on a GPU, through QRMI, or on Pasqal Cloud."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np


def hamiltonian(point: dict):
    """Construct the analog Hamiltonian from SI waveform samples.

CUDA-Q evaluates each callback only at the schedule times; the target joins the samples linearly
(amplitude, detuning) or holds them (phase), so the lambdas only need to return the stored samples.

Args:
    point (dict): Coordinates, schedule times and waveform samples.

Returns:
    (RydbergHamiltonian): Program consumed by the Pasqal target.
"""
    from cudaq.operators import RydbergHamiltonian, ScalarOperator

    def sample(values):
        return ScalarOperator(lambda t: float(np.interp(t.real, point["times"], values)))

    return RydbergHamiltonian(atom_sites=point["atom_sites"], amplitude=sample(point["amplitude"]),
                              delta_global=sample(point["detuning"]), phase=sample(point["phase"]))


def execute(program: dict, emulate: bool = False, machine: str = "qrmi", seed: int | None = None) -> dict:
    """Run every prepared point and retain its parameters with the counts.

Args:
    program (dict): Prepared program including C6 and shots.
    emulate (bool, optional): Execute with the local GPU emulator.
    machine (str, optional): QRMI or a direct Pasqal Cloud machine name.
    seed (int | None, optional): GPU sampling seed.

Returns:
    (dict): Run record with atom-order counts and execution metadata.
"""
    # CUDA-Q 0.17 announces changes to sample/observe on import; these examples only use evolve.
    warnings.filterwarnings("ignore", message="The CUDA-Q `sample` and `observe`", category=FutureWarning)
    import cudaq
    from cudaq.dynamics import Schedule

    target = {"emulate": True, "emulation_rydberg_c6": str(program["c6"])} if emulate else {"machine": machine}
    cudaq.set_target("pasqal", **target)
    if emulate and seed is not None:
        cudaq.set_random_seed(seed)
    record = {
        "program": program, "backend": "GPU emulation" if emulate else machine,
        "label": os.environ.get("PASQAL_QRMI_LABEL") if not emulate and machine == "qrmi" else None,
        "cudaq_version": cudaq.__version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resources": os.environ.get("QRMI_JOB_QPU_RESOURCES") if not emulate else None,
        "seed": seed if emulate else None,
    }
    start = time.monotonic()
    # Submit every point first, then wait: remote jobs queue together instead of one after another.
    futures = [cudaq.evolve_async(hamiltonian(p), schedule=Schedule(p["times"], ["t"]), shots_count=program["shots"])
               for p in program["points"]]
    record["counts"] = [dict(sorted(f.get().items())) for f in futures]
    record["elapsed_seconds"] = time.monotonic() - start
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--emulate", action="store_true")
    target.add_argument("--machine", default="qrmi", help="qrmi or a direct Cloud machine such as EMU_FREE")
    parser.add_argument("--seed", type=int, help="reproducible GPU samples")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = execute(json.loads(args.program.read_text()), args.emulate, args.machine, args.seed)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {args.output}; execution took {record['elapsed_seconds']:.2f} s")
