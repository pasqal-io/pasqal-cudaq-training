# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Compare CUDA-Q GPU results with an independent CPU reference and execute the notebooks; requires a GPU."""

import contextlib
import io
import json
from pathlib import Path
import unittest

from run import execute
from tests.test_examples import reference

ROOT = Path(__file__).resolve().parents[1]


class GPUChecks(unittest.TestCase):
    def test_register_order_and_c6(self):
        """Detect reversed counts or a mismatched interaction coefficient."""
        point = {"atom_sites": [(0.0, 0.0), (7e-6, 0.0), (3e-6, 6e-6)], "times": [0.0, 0.5e-6, 1.5e-6, 2e-6],
                 "amplitude": [0.0, 6.3e6, 6.3e6, 0.0], "detuning": [-6e6, -2e6, 4e6, 8e6], "phase": [0.0] * 4}
        levels = (8.6572302e-25, 5.42015853e-24)
        exact = {c6: reference(point, c6) for c6 in levels}

        def tvd(counts, probabilities, flip=False):
            return sum(abs(p - counts.get(b[::-1] if flip else b, 0) / 20000) for b, p in probabilities.items()) / 2

        for c6, other in (levels[::-1], levels):  # level 60 last: it is not reversal-symmetric
            counts = execute({"c6": c6, "shots": 20000, "points": [point]}, emulate=True, seed=19)["counts"][0]
            print(f"C6={c6:g}: TVD={tvd(counts, exact[c6]):.4f}; other C6 {tvd(counts, exact[other]):.4f}; "
                  f"reversed {tvd(counts, exact[c6], flip=True):.4f}")
            self.assertLess(tvd(counts, exact[c6]), 0.02)
            self.assertGreater(tvd(counts, exact[other]), 0.05)
        self.assertGreater(tvd(counts, exact[c6], flip=True), 0.05)

    def test_notebooks(self):
        """Execute every notebook and check its headline result."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        expected = {"mis.ipynb": lambda ns: ns["p_opt"] > 0.8, "dynamics.ipynb": lambda ns: ns["tvd"] < 0.02,
                    "blockade.ipynb": lambda ns: ns["both"][0] < 0.02, "materials.ipynb": lambda ns: ns["mz"][-1] > 0.9}
        for path in sorted(ROOT.glob("0[1-5]-*/*.ipynb")):
            with self.subTest(notebook=path.name), contextlib.chdir(path.parent), \
                    contextlib.redirect_stdout(io.StringIO()):
                namespace = {}
                for cell in json.loads(path.read_text())["cells"]:
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), path.name, "exec"), namespace)
                self.assertTrue(expected[path.name](namespace))
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
