# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Check the blockade program, observables and comparison plot on a CPU."""

import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from cudaq_pasqal import analyze
import blockade


def reference(point: dict, c6: float) -> dict[str, float]:
    """Integrate the zero-phase spin Hamiltonian independently in atom order.

Args:
    point (dict): Prepared coordinates and piecewise-linear fields.
    c6 (float): Interaction coefficient in rad m^6/s.

Returns:
    (dict[str, float]): Exact-state probabilities indexed by atom-order strings.
"""
    sites = point["atom_sites"]
    size = len(sites)
    strings = ["".join(b) for b in itertools.product("01", repeat=size)]
    occupations = np.array([[int(b) for b in bits] for bits in strings])
    interaction = sum(c6 / math.dist(sites[i], sites[j]) ** 6 * occupations[:, i] * occupations[:, j]
                      for i, j in itertools.combinations(range(size), 2))
    flips = [np.arange(2**size) ^ (1 << (size - 1 - i)) for i in range(size)]
    numbers = occupations.sum(axis=1)

    def derivative(t, state):
        omega = np.interp(t, point["times"], point["amplitude"])
        delta = np.interp(t, point["times"], point["detuning"])
        return -1j * ((interaction - delta * numbers) * state + omega / 2 * sum(state[f] for f in flips))

    state = np.zeros(2**size, dtype=complex)
    state[0] = 1
    for start, end in zip(point["times"][:-1], point["times"][1:]):
        solution = solve_ivp(derivative, (start, end), state, method="DOP853", rtol=1e-10, atol=1e-12)
        assert solution.success, solution.message
        state = solution.y[:, -1]
    return dict(zip(strings, np.abs(state) ** 2))


class ScientificChecks(unittest.TestCase):
    def test_blockade_limit(self):
        """Check the collective pi pulse and independent-atom limit."""
        program = blockade.prepare([5, 50], 8.6572302e-25, 100)
        blocked, free = [reference(p, program["c6"]) for p in program["points"]]
        self.assertLess(blocked["11"], 0.02)
        self.assertGreater(blocked["01"] + blocked["10"], 0.95)
        duration = program["points"][1]["times"][-1]
        expected = math.sin(math.pi * 1e6 * duration / 2) ** 4
        self.assertAlmostEqual(free["11"], expected, places=5)

    def test_observables(self):
        """Check observable normalization and shot uncertainty on known states."""
        self.assertEqual(analyze.probability({"00": 50, "11": 50}, ["11"]), (0.5, 0.05))
        self.assertEqual(analyze.magnetization({"000": 100}), (1.0, 0.0))
        self.assertEqual(analyze.magnetization({"111": 100}), (-1.0, 0.0))
        self.assertEqual(analyze.magnetization({"000": 50, "111": 50}), (0.0, 0.1))
        self.assertEqual(analyze.structure_factor({"00": 100}, [[0, 0], [1, 0]], 1), 0)
        self.assertEqual(analyze.structure_factor({"00": 25, "01": 25, "10": 25, "11": 25},
                                                  [[0, 0], [1, 0]], 1), 1)

    def test_compare_plot(self):
        """Compare two saved blockade records and write the plot."""
        program = blockade.prepare([5, 14], 8.6572302e-25, 100)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, counts in (("gpu", [{"01": 50, "10": 50}, {"11": 60, "00": 40}]),
                                 ("qpu", [{"01": 45, "10": 50, "11": 5}, {"11": 55, "00": 45}])):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps({"program": program, "backend": name, "cudaq_version": "test",
                                            "elapsed_seconds": 1.0, "counts": counts}))
                paths.append(str(path))
            output = Path(directory) / "comparison.png"
            result = subprocess.run([sys.executable, str(root / "04-slurm/compare.py"), *paths, "--plot", str(output)],
                                    capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND="Agg"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("5 um: P(11) = 0.050 +/- 0.022", result.stdout)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
