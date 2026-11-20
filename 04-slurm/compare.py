# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Compare saved blockade runs, e.g. GPU and QPU, without executing quantum work."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cudaq_pasqal.analyze import probability

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("records", nargs="+", type=Path)
parser.add_argument("--plot", type=Path, required=True)
args = parser.parse_args()

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for path in args.records:
    record = json.loads(path.read_text())
    label = record.get("label") or record["backend"]
    distances = [p["distance_um"] for p in record["program"]["points"]]
    values, errors = zip(*(probability(c, ["11"]) for c in record["counts"]))
    print(f"\n{label} | {record['cudaq_version']} | C6 {record['program']['c6']:g} | {record['elapsed_seconds']:.2f} s")
    for d, v, e in zip(distances, values, errors):
        print(f"{d:5g} um: P(11) = {v:.3f} +/- {e:.3f}")
    ax.errorbar(distances, values, yerr=errors, marker="o", capsize=3, label=label)
ax.set(xlabel="Atom separation (µm)", ylabel="P(11)", title="Rydberg blockade")
ax.legend()
ax.grid(alpha=0.2)
fig.savefig(args.plot, dpi=160)
print(f"\nSaved {args.plot}")
