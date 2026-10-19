# Scaling benchmark results, 2026-10-18

Measured on an AMD Ryzen Threadripper 2990WX (32 cores, 64 threads) with an NVIDIA
GeForce RTX 2070 (driver 615.71.09). Simulation time for 1,000 ms at dt = 1.0 ms; the
setup is in `../benchmark.py`. Summary and caveats: `REPRODUCING.md`, section
"Scaling benchmark".

| File | Produced by | Environment |
|------|-------------|-------------|
| `bindsnet_scaling.csv` | `python examples/benchmark/bindsnet_scaling.py` at commit `f392a65` | project environment: Python 3.11.9, PyTorch 2.14.0 (CUDA 13.0 build) |
| `benchmark.csv` | `python examples/benchmark/benchmark.py --simulators bindsnet_cpu brian2 nest annarchy_cpu brian2genn annarchy_gpu --start 1000 --stop 10000 --step 1000` at commit `f392a65` (brian2genn and annarchy_gpu skipped: not installed / no CUDA compiler) | separate environment: BRIAN2 2.9.0, NEST 3.10.0, ANNarchy 5.0, PyTorch 2.14.1 CPU build, numpy 2.3.5 |
| `annarchy_cpu_64_threads.csv` | same as `benchmark.csv`, `--simulators annarchy_cpu`, with ANNarchy set to 64 threads (a test; the committed `benchmark.py` uses ANNarchy's default of one thread) | as above |

Columns: `sim_s` is the simulation call alone; `total_s` adds building the network and
encoding the input (and, for ANNarchy, compiling it).
