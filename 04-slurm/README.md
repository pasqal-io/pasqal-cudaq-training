# 04 · GPU → QPU workflow with Slurm

Run a prepared program as three dependent Slurm jobs, the flow from the
[Pasqal HRES blog](https://www.pasqal.com/blog/enabling-hybrid-gpu-qpu-workflows-with-slurm-hres-and-pasqal-qpus/):

```text
prepared program -> GPU job -> QPU job -> CPU comparison
                     GPU       QPU          neither
```

Each job holds only the resource it needs, so the QPU is not reserved while the GPU
emulates or the analysis runs. Use a login node where the repository is installed
(`pip install -e '.[gpu-cu13]'` or `'.[gpu-cu12]'`, see the main README) and the site has a QRMI
integration (SPANK plugin or HRES). Run all commands from the repository root.

## The files

| File | Does |
|---|---|
| `blockade.py` | Writes the distance sweep of notebook 1 as a program file: coordinates, pulse samples, shots and $C_6$ |
| `run.py` | Turns each point into a `RydbergHamiltonian`, submits all of them with `cudaq.evolve_async` and saves the counts |
| `compare.py` | Prints P(11) per distance for each saved run and plots them together |
| `run.sbatch`, `analyze.sbatch`, `gpu_to_qpu.sh` | The Slurm jobs and the chain that links them |

`run.py` is short and uses the same CUDA-Q calls as notebook 1; read it before submitting.

## One QPU job

Inside a job assigned a Pasqal resource, `run.py` selects
`cudaq.set_target("pasqal", machine="qrmi")`. The scheduler provides the resource
and session (`QRMI_JOB_QPU_RESOURCES`, `QRMI_JOB_QPU_TYPES`); on-prem access uses
the site's QRMI authentication, with no Pasqal Cloud login. With a SPANK
installation, for example:

```bash
python 04-slurm/blockade.py --output blockade.json
sbatch --qpu=PASQAL_LOCAL 04-slurm/run.sbatch blockade.json --output blockade-qpu.json
# After the job completes:
python 04-slurm/compare.py blockade-qpu.json --plot blockade-qpu.png
```

`PASQAL_LOCAL` is an example resource name. With HRES, request
`--resources=PASQAL_LOCAL:1` instead. Add the site's license, partition and
account options. Both run the same Python code.

## The full chain

```bash
GPU_REQUEST='--gres=gpu:1' \
QPU_REQUEST='--qpu=PASQAL_LOCAL' \
bash 04-slurm/gpu_to_qpu.sh blockade.json runs/blockade-01
```

For HRES, set `QPU_REQUEST='--resources=PASQAL_LOCAL:1'`. `ANALYSIS_REQUEST` takes
options for the CPU comparison, for example
`ANALYSIS_REQUEST='--account=YOUR_ACCOUNT --partition=CPU_PARTITION --time=00:05:00'`;
it defaults to the site's ordinary batch settings. Options are whitespace-separated,
so use paths without spaces inside them.

Each step runs only after its predecessor succeeds (`afterok`), and a failed step
cancels its dependents (`--kill-on-invalid-dep=yes`). The script copies the program
into a new output directory and saves each accepted job ID immediately, so you can
cancel a partial chain if a later submission is rejected:

```bash
squeue -u "$USER"
cat runs/blockade-01/analysis.out          # comparison.png is in the same directory
xargs scancel < runs/blockade-01/job-ids.txt   # stop a submitted chain
```

The GPU and QPU runs read the same recorded program. Its `c6` sets the GPU
emulation only; the QPU uses its own. `blockade.py` defaults to the level-60 value
of notebooks 1–3. For a device at another Rydberg level, pass its value, e.g.
`--c6 5.42015853e-24` for level 70, otherwise the GPU curve puts the blockade radius
in the wrong place. Error bars are one standard
error from the shots; they do not include device noise or calibration uncertainty.
Agreement with the ideal GPU curve is an observation, not a pass/fail threshold.
At another site the backend is recorded as `qrmi`; an operator can set
`PASQAL_QRMI_LABEL` to describe the assigned device in records and plots.

## Try it: trace the resources

Read `gpu_to_qpu.sh`. Identify the GPU, QPU and CPU requests and the two `afterok`
dependencies. What happens if the GPU job fails, and how do you cancel accepted jobs
if a later submission is rejected?

<details>
<summary>Answer</summary>

The GPU job uses `GPU_REQUEST`, the QPU job `QPU_REQUEST`, and the analysis
neither. A failed GPU job cancels both dependents. `job-ids.txt` records each
accepted job as it is submitted; `xargs scancel < runs/YOUR_RUN/job-ids.txt`
cancels them.

</details>

## When something stalls

| Symptom | Check |
|---|---|
| Job stays pending | `squeue -o '%.18i %.9T %.35R'`, `sinfo`, `scontrol show licenses`. A dependent job waits for its predecessor. |
| Job disappears from `squeue` | `sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed`, then the matching `gpu.out`, `qpu.out` or `analysis.out`. |
| Output directory already exists | Pick a new `runs/` directory; each one holds one submitted chain. |
| GPU and QPU disagree strongly | Compare the recorded programs, especially `c6`, coordinates and waveforms, with the device description. |
