# From GPU emulation to Pasqal QPUs with CUDA-Q

Develop an analog neutral-atom program on a GPU, run the same program on a Pasqal
QPU, and compare the measurements. These examples accompany Pasqal and NVIDIA
training on CUDA-Q. They use CUDA-Q's `pasqal` target and
`RydbergHamiltonian`; the programs are analog pulses, not gate circuits.

| Step | What you learn | Needs |
|---|---|---|
| [01-blockade](01-blockade/blockade.ipynb) | Write an analog program in CUDA-Q: $C_6$, schedules, `evolve_async`, the Rydberg blockade, switching targets | Notebook |
| [02-mis](02-mis/mis.ipynb) | Solve a graph problem (maximum independent set) and tune the pulse on the GPU before the QPU | Notebook |
| [03-materials](03-materials/materials.ipynb) | Prepare a frustrated antiferromagnet on a triangular lattice, polarise it with a field and look at its spin snapshots | Notebook |
| [04-slurm](04-slurm/README.md) | Run a basic HPC-QC workflow: GPU → QPU → analysis as dependent Slurm jobs | A cluster with QRMI |
| [05-dynamics](05-dynamics/dynamics.ipynb) | Optional: write your own simulator of the Pasqal backend on CUDA-Q's `dynamics` target, with noise | Notebook |
| [06-cpp](06-cpp/README.md) | Optional: the same program in C++ with `nvq++` | CUDA-Q container |

**Prerequisites:** Python and NumPy, the idea of a qubit, and of a Hamiltonian as what
drives a system's evolution in time. No CUDA-Q, Pulser or neutral-atom experience is
needed: notebook 1 introduces the physics.

The notebooks build on each other and call CUDA-Q directly; every result is
computed live on an NVIDIA GPU. The repository keeps the executed notebooks, so you
can read the outputs without running them.

## Install

Use Python 3.11 or newer. From the repository root:

```bash
nvidia-smi                         # "CUDA Version" in the header: 13.x → gpu-cu13, 12.x → gpu-cu12
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu-cu13,notebook]'   # or '.[gpu-cu12,notebook]': CUDA-Q 0.17, CuPy, widgets, JupyterLab
jupyter lab
```

The notebooks need Linux with an NVIDIA GPU: the `pasqal` emulator and the
`dynamics` target run on NVIDIA cuDensityMat. Pick the extra that matches the
highest CUDA version your driver supports. The `pasqal` target is part of both
CUDA-Q wheels. To check the installation, run the GPU tests (see Tests below).

With an existing Jupyter server, such as a centre's JupyterHub, leave out `notebook`
and register the environment as a kernel instead:

```bash
pip install -e '.[gpu-cu13]'
python -m ipykernel install --user --name cudaq-pasqal --display-name "CUDA-Q Pasqal"
```

## The same program on every target

Notebook 1 shows that only `cudaq.set_target` changes between the GPU, a QPU and
the cloud. For batch jobs, [`04-slurm/run.py`](04-slurm/run.py) reads a program file
(atom coordinates and pulse samples in SI units, plus shots) and runs it on the
chosen target:

```bash
python 04-slurm/blockade.py --output blockade.json
python 04-slurm/run.py blockade.json --emulate --output blockade-gpu.json                  # GPU
python 04-slurm/run.py blockade.json --output blockade-qpu.json                            # QPU through QRMI, inside a Slurm job
python 04-slurm/run.py blockade.json --machine EMU_FREE --output blockade-cloud.json       # Pasqal Cloud
python 04-slurm/compare.py blockade-gpu.json blockade-qpu.json --plot comparison.png
```

The QRMI route needs a job that the scheduler assigned a Pasqal resource
([04-slurm](04-slurm/README.md)). The Cloud route uses `PASQAL_AUTH_TOKEN` and
`PASQAL_PROJECT_ID` from your Pasqal Cloud login; `EMU_FREE` is a Cloud emulator,
not a physical QPU. Keep credentials outside the code.

The examples use the level-60 interaction coefficient,
`c6 = 8.6572302e-25` rad m⁶/s (notebook 1 explains it). Before comparing with a QPU,
emulate with that device's coefficient and respect its layout and pulse limits.
`c6` affects emulation only.

## Repository layout

| Path | Contents |
|---|---|
| `01-*/` … `03-*/`, `05-*/` | One executed notebook each |
| `04-slurm/` | Program file (`blockade.py`), runner (`run.py`), comparison (`compare.py`) and job scripts |
| `06-cpp/` | The blockade sweep in C++ |
| `cudaq_pasqal/` | Plots (`draw.py`) and observables (`analyze.py`) for the notebooks; no CUDA-Q code |
| `tests/` | Independent numerical checks |

## Tests

```bash
pip install -e '.[test]'
python -m unittest tests.test_examples tests.test_workflow -v   # CPU: physics reference, observables, Slurm arguments
python -m unittest tests.test_gpu -v                             # GPU: CUDA-Q against an exact calculation, all notebooks
```

The CPU checks need neither CUDA-Q nor Slurm and run in GitHub Actions. The GPU
checks take about four minutes. Changes to pulse conversion or observables should
come with an independent numerical check. After changing a notebook, execute it on a
GPU (`jupyter nbconvert --to notebook --execute --inplace`) so the saved outputs match.
In an issue, include the command, the Python and CUDA-Q versions, your GPU setup and
the error text, with credentials removed.

## Sources and license

Blockade and workflow code adapt the
[Pasqal HRES example](https://github.com/pasqal-io/qrmi-hres-pasqal-blog/tree/a7b1e0fc520d351fd26707e39021700e9c454d19).
The MIS example follows the [Pulser graph tutorial](https://docs.pasqal.com/pulser/tutorials/mwis/)
using [CUDA-Q's neutral-atom backend](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware/neutralatom.html).
The antiferromagnet example adapts the
[Pasqal SC26 OpenQSE demonstration](https://github.com/pasqal-io/sc26-openqse-demo/blob/3248fa538b5f4e228edbce1500a6be3af80c83bf/tmgo/tmgo.py),
based on [Leclerc et al.](https://arxiv.org/abs/2603.20372).

Copyright 2026 Pasqal. Licensed under the [Apache License, Version 2.0](LICENSE), like
CUDA-Q and Pulser.
