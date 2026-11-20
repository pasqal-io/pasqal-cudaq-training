// Copyright 2026 Pasqal
// SPDX-License-Identifier: Apache-2.0
// The blockade sweep of notebook 1 in C++. Build for the GPU emulator with
//   nvq++ --target pasqal --emulate --pasqal-emulation_rydberg_c6 8.6572302e-25 blockade.cpp -o blockade
// and for the QPU assigned to a Slurm job with
//   nvq++ --target pasqal --pasqal-machine qrmi blockade.cpp -o blockade

#include <cudaq.h>
#include <cudaq/algorithms/evolve.h>
#include <cudaq/algorithms/integrator.h> // complete types needed by evolve.h
#include <cudaq/operators.h>
#include <cudaq/schedule.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const double omega = M_PI * 1e6; // Rabi frequency in rad/s
  // Collective pi pulse of a blockaded pair, rounded to the 4 ns clock: 708 ns.
  const double duration = std::round(M_PI / (std::sqrt(2) * omega) / 4e-9) * 4e-9;

  // The callbacks receive the schedule parameters by name. CUDA-Q evaluates them
  // only at the schedule times; amplitude and detuning are linear in between.
  cudaq::schedule schedule({0.0, duration}, {"t"}, {});
  auto amplitude = cudaq::scalar_operator(omega);
  auto phase = cudaq::scalar_operator(0.0);
  auto detuning = cudaq::scalar_operator(0.0);

  std::printf("distance_um  P(00)  P(01|10)  P(11)\n");
  for (int distance = 4; distance <= 14; ++distance) {
    const double half = 0.5e-6 * distance; // metres
    std::vector<std::pair<double, double>> sites = {{-half, 0.0}, {half, 0.0}};
    auto hamiltonian = cudaq::rydberg_hamiltonian(sites, amplitude, phase, detuning);
    auto result = cudaq::evolve(hamiltonian, schedule, 500);
    auto &counts = *result.sampling_result; // atom-order bitstrings, atom 0 first
    std::printf("%11d  %5.3f  %8.3f  %5.3f\n", distance, counts.probability("00"),
                counts.probability("01") + counts.probability("10"),
                counts.probability("11"));
  }
  return 0;
}
