# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Prepare the two-atom blockade sweep of notebook 1 as a program file for GPU and QPU jobs."""

import argparse
import json
import math
from pathlib import Path

OMEGA = math.pi * 1e6  # Rabi frequency in rad/s: one full oscillation every 2 µs.


def prepare(distances: list[float], c6: float, shots: int) -> dict:
    """Build the same collective pi pulse for two atoms at each separation.

Args:
    distances (list[float]): Atom separations in micrometers.
    c6 (float): C6/hbar in rad m^6/s of the intended device; used by the GPU emulation only.
    shots (int): Samples per point.

Returns:
    (dict): Prepared sweep with SI coordinates and waveform samples.
"""
    duration = round(math.pi / (math.sqrt(2) * OMEGA) / 4e-9) * 4e-9
    return {"example": "blockade", "c6": c6, "shots": shots, "points": [{
        "distance_um": d, "atom_sites": [(-d * 0.5e-6, 0.0), (d * 0.5e-6, 0.0)], "times": [0.0, duration],
        "amplitude": [OMEGA, OMEGA], "detuning": [0.0, 0.0], "phase": [0.0, 0.0],
    } for d in distances]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=float, nargs="+", default=[5, 8, 10, 12, 14])
    parser.add_argument("--c6", type=float, default=8.6572302e-25,
                        help="device C6/hbar in rad m^6/s; default: level-60 reference")
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("blockade.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(prepare(args.distances, args.c6, args.shots), indent=2) + "\n")
    print(f"Prepared {len(args.distances)} distances in {args.output}")
