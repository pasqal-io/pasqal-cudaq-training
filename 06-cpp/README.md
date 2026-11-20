# 06 · Optional: the same program in C++

CUDA-Q has the same analog API in C++: `cudaq::rydberg_hamiltonian`,
`cudaq::scalar_operator`, `cudaq::schedule` and `cudaq::evolve`.
[`blockade.cpp`](blockade.cpp) is the distance sweep from notebook 1.

The pip wheel has no C++ compiler. Use the CUDA-Q container (or the
[installer](https://nvidia.github.io/cuda-quantum/latest/using/install/local_installation.html)),
from this directory:

```bash
docker run --rm -it --gpus all -v "$PWD:/work" -w /work \
  nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.17.0
# inside the container:
nvq++ --target pasqal --emulate --pasqal-emulation_rydberg_c6 8.6572302e-25 blockade.cpp -o blockade
./blockade
```

In C++ the target is chosen when you compile, not in the code:

| Python | `nvq++` |
|---|---|
| `cudaq.set_target("pasqal", emulate=True, emulation_rydberg_c6="8.6572302e-25")` | `--target pasqal --emulate --pasqal-emulation_rydberg_c6 8.6572302e-25` |
| `cudaq.set_target("pasqal", machine="qrmi")` | `--target pasqal --pasqal-machine qrmi` |
| `cudaq.set_target("pasqal", machine="EMU_FREE")` | `--target pasqal --pasqal-machine EMU_FREE` |

`8.6572302e-25` rad m⁶/s is the level-60 $C_6$ used in notebooks 1–3 (see notebook 1).
It is also what the emulator uses when you omit the flag. For another device,
pass that device's value; the QPU itself ignores the flag.

The output lists P(00), P(01 or 10) and P(11) per separation. Compare it with the
distance sweep in notebook 1: the numbers agree within sampling error.

**Try it.** Rebuild with `--pasqal-emulation_rydberg_c6 5.42015853e-24` (level 70).
Where does `11` start to appear now?
