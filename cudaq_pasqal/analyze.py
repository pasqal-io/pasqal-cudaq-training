# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Compute observables and their sampling uncertainty from atom-order counts."""

import math

import numpy as np


def probability(counts: dict, accepted: list[str]) -> tuple[float, float]:
    """Calculate an event frequency and its binomial standard error.

Args:
    counts (dict): Atom-order bitstring counts.
    accepted (list[str]): Bitstrings counted as the event.

Returns:
    probability (float): Observed event frequency.
    error (float): One-standard-error sampling uncertainty.
"""
    shots = sum(counts.values())
    p = sum(counts.get(b, 0) for b in accepted) / shots
    return p, math.sqrt(p * (1 - p) / shots)


def magnetization(counts: dict) -> tuple[float, float]:
    """Calculate mean spin per site and its sampling uncertainty.

Args:
    counts (dict): Atom-order bitstrings, zero for ground and one for Rydberg.

Returns:
    mean (float): Magnetization with ground state assigned positive spin.
    error (float): Standard error across sampled shots.
"""
    shots = sum(counts.values())
    values = {bits: 1 - 2 * bits.count("1") / len(bits) for bits in counts}
    mean = sum(counts[b] * values[b] for b in counts) / shots
    variance = sum(counts[b] * (values[b] - mean) ** 2 for b in counts) / shots
    return mean, math.sqrt(variance / shots)


def rydberg_density(counts: dict) -> np.ndarray:
    """Return the probability that each atom was measured in the Rydberg state.

Args:
    counts (dict): Atom-order bitstring counts.

Returns:
    (np.ndarray): One probability per atom.
"""
    bits = np.array([[int(b) for b in s] for s in counts])
    return np.array(list(counts.values())) @ bits / sum(counts.values())


def structure_factor(counts: dict, sites: list, spacing: float) -> float:
    """Calculate the connected structure factor at the three-sublattice wavevector.

Args:
    counts (dict): Atom-order bitstring counts.
    sites (list): Atom coordinates in meters.
    spacing (float): Nearest-neighbor separation in meters.

Returns:
    (float): Connected structure factor per site, without detection correction.
"""
    z = np.array([[1 - 2 * int(b) for b in bits] for bits in counts])
    weights = np.array(list(counts.values())) / sum(counts.values())
    mean = weights @ z
    connected = (z.T * weights) @ z - np.outer(mean, mean)
    r = np.asarray(sites) / spacing
    q = 2 * math.pi / 3 * np.array([1, math.sqrt(3)])
    return float(np.real(np.exp(1j * ((r[:, None] - r[None]) @ q)) * connected).sum() / len(r))
