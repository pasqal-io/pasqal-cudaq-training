# Copyright 2026 Pasqal
# SPDX-License-Identifier: Apache-2.0
"""Draw registers, pulses and measured counts in the notebooks."""

import matplotlib.pyplot as plt
import numpy as np

from cudaq_pasqal.analyze import probability


def blockade_radius(c6: float, omega: float) -> float:
    """Return the distance where the interaction equals the drive.

Args:
    c6 (float): C6/hbar in rad m^6/s.
    omega (float): Rabi frequency in rad/s.

Returns:
    (float): Blockade radius in micrometers.
"""
    return (c6 / omega) ** (1 / 6) * 1e6


def register(sites: list, radius_um: float | None = None, edges: list = (), values: list | None = None,
             ax=None, title: str = "Register"):
    """Draw atoms in micrometers, optionally with half-blockade-radius discs, graph edges and per-atom values.

Args:
    sites (list): Atom coordinates in meters, in atom order.
    radius_um (float | None, optional): Blockade radius; discs of half this radius touch when two atoms blockade.
    edges (list, optional): Pairs of atom indices drawn as lines.
    values (list | None, optional): Per-atom value between 0 and 1, e.g. Rydberg probability, used as colour.
    ax (Axes, optional): Axes to draw on.
    title (str, optional): Axes title.

Returns:
    (Axes): Axes with the drawing.
"""
    ax = ax or plt.subplots(figsize=(4.5, 4.5))[1]
    xy = np.asarray(sites) * 1e6
    for i, j in edges:
        ax.plot(*xy[[i, j]].T, color="0.6", zorder=1)
    if radius_um:
        for x, y in xy:
            ax.add_patch(plt.Circle((x, y), radius_um / 2, color="tab:blue", alpha=0.12, zorder=0))
    colors = "tab:blue" if values is None else plt.cm.viridis(np.asarray(values, dtype=float))
    ax.scatter(*xy.T, s=320, c=colors, edgecolors="k", zorder=2)
    light = [False] * len(xy) if values is None else [v > 0.6 for v in values]
    for index, (x, y) in enumerate(xy):
        ax.annotate(str(index), (x, y), color="k" if light[index] else "w", ha="center", va="center",
                    fontsize=9, zorder=3)
    margin = max(radius_um / 2 if radius_um else 0, 2)
    ax.set(xlabel="x (µm)", ylabel="y (µm)", title=title, aspect="equal",
           xlim=(xy[:, 0].min() - margin, xy[:, 0].max() + margin),
           ylim=(xy[:, 1].min() - margin, xy[:, 1].max() + margin))
    return ax


def pulse(times: list, amplitude: list, detuning: list, title: str = "Pulse"):
    """Draw the global amplitude and detuning as the target applies them: linear between samples.

Args:
    times (list): Schedule times in seconds.
    amplitude (list): Rabi frequency samples in rad/s.
    detuning (list): Detuning samples in rad/s.
    title (str, optional): Figure title.

Returns:
    (Figure): Figure with amplitude and detuning panels.
"""
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 4), layout="constrained")
    t = np.asarray(times) * 1e6
    top.fill_between(t, np.asarray(amplitude) / 1e6, color="tab:green", alpha=0.4)
    top.plot(t, np.asarray(amplitude) / 1e6, "o-", color="tab:green")
    bottom.plot(t, np.asarray(detuning) / 1e6, "o-", color="tab:purple")
    bottom.axhline(0, color="0.7", lw=0.8)
    if not any(detuning):
        bottom.set_ylim(-1, 1)
    top.set(ylabel="Ω (rad/µs)", title=title)
    bottom.set(xlabel="Time (µs)", ylabel="Δ (rad/µs)")
    return fig


def counts(counts: dict, highlight: list = (), top: int = 12, ax=None, title: str = "Measured bitstrings"):
    """Draw the most frequent bitstrings with one-standard-error bars.

Args:
    counts (dict): Atom-order bitstring counts.
    highlight (list, optional): Bitstrings drawn in a contrasting colour, e.g. exact optima.
    top (int, optional): Number of bitstrings shown.
    ax (Axes, optional): Axes to draw on.
    title (str, optional): Axes title.

Returns:
    (Axes): Axes with the bar chart.
"""
    ax = ax or plt.subplots(figsize=(7, 3.5))[1]
    labels = sorted(counts, key=counts.get, reverse=True)[:top]
    values, errors = zip(*(probability(counts, [b]) for b in labels))
    ax.bar(labels, values, yerr=errors, capsize=3,
           color=["tab:orange" if b in highlight else "tab:blue" for b in labels])
    ax.set(xlabel="Atom-order bitstring (atom 0 first)", ylabel="Probability", title=title)
    ax.tick_params(axis="x", rotation=45)
    return ax


def snapshots(sites: list, counts: dict, number: int = 4, title: str = "Measured snapshots"):
    """Draw the most frequent measured configurations on the register.

Args:
    sites (list): Atom coordinates in meters.
    counts (dict): Atom-order bitstring counts.
    number (int, optional): Number of configurations.
    title (str, optional): Figure title.

Returns:
    (Figure): One panel per configuration; yellow atoms are in the Rydberg state.
"""
    labels = sorted(counts, key=counts.get, reverse=True)[:number]
    fig, axes = plt.subplots(1, number, figsize=(3 * number, 3), layout="constrained")
    for ax, bits in zip(np.atleast_1d(axes), labels):
        register(sites, values=[int(b) for b in bits], ax=ax,
                 title=f"{bits}\n{counts[bits] / sum(counts.values()):.1%}")
        ax.set(xlabel="", ylabel="", xticks=[], yticks=[])
    fig.suptitle(title)
    return fig

