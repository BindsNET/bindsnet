# Reproducing results with BindsNET

This table traces each model BindsNET describes or ships back to executable code: the
model class, the example script, an exact command, the seed, the measured output, and
the data it needs (declared in [DATA.md](DATA.md)).

> **Measured, not quoted.** Every number in the **Measured output** column was
> produced by the command next to it, at commit `f392a65`, on the machine described
> under [Measurement setup](#measurement-setup). The runs are short (one pass over
> 20,000 training images, 100 neurons), so they serve as reference points for checking
> a reproduction; Diehl & Cook (2015) report up to 95% with 6,400 neurons.

> **Which release.** Each BindsNET release is an archived snapshot. BindsNET is a
> research package that keeps gaining features and fixes, and results can also depend
> on the hardware, so results may differ between releases. A published result is
> reproduced with the release it was produced on; please name that release in the
> publication, ideally by its version DOI (see "Citing the software" in the README).

## Model → code → command map

Run the MNIST commands from `examples/mnist/` (the scripts read and download MNIST at
`../../data/MNIST`). `CUDA_VISIBLE_DEVICES=` runs the script on the CPU; leave it out
to use a GPU. Runs with the same seed on the same device repeat exactly, and CPU and
GPU runs follow different learning trajectories (see [Determinism](#determinism)).

| Claim / source | Model class | Script | Command | Seed | Measured output | Data |
|----------------|-------------|--------|---------|------|-----------------|------|
| Diehl & Cook 2015 MNIST replication (DOI `10.3389/fncom.2015.00099`) | `DiehlAndCook2015` | `examples/mnist/eth_mnist.py` | `CUDA_VISIBLE_DEVICES= python eth_mnist.py --n_neurons 100 --n_epochs 1 --time 250 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | CPU. Test accuracy: all-activity **0.75**, proportion-weighting **0.76**. 2.06 h. | MNIST |
| Batched ETH MNIST | `DiehlAndCook2015` | `examples/mnist/batch_eth_mnist.py` | `python batch_eth_mnist.py --n_neurons 100 --batch_size 32 --time 100 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | GPU (RTX 2070). Test accuracy: all-activity **0.81**, proportion-weighting **0.83**. 7.6 min. | MNIST |
| Supervised MNIST (label-clamped) | `DiehlAndCook2015` | `examples/mnist/supervised_mnist.py` | `CUDA_VISIBLE_DEVICES= python supervised_mnist.py --n_neurons 100 --time 250 --intensity 32 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | CPU. Test accuracy: all-activity **0.64**, proportion-weighting **0.64**. 2.06 h. | MNIST |
| Convolutional SNN on MNIST | (in-script conv network) | `examples/mnist/conv_mnist.py` | `CUDA_VISIBLE_DEVICES= python conv_mnist.py --time 50 --batch_size 1 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | CPU. Test accuracy: all-activity **0.12**, proportion-weighting **0.14**. It learns early (0.27 / 0.28 after 1,000 images) and then declines; see [conv_mnist over training](#conv_mnist-over-training). 1.58 h. | MNIST |
| Reservoir / liquid-state MNIST | (in-script reservoir) | `examples/mnist/reservoir.py` | `python reservoir.py --n_neurons 500 --n_epochs 100 --time 250 --seed 0` | `--seed 0` | CPU. Readout accuracy on 500 held-out test images: **53.00 %** (trained on 500 images, the script's default `--examples`). 5.8 min. | MNIST |
| Scaling benchmark (Hazan et al. 2018, DOI `10.3389/fninf.2018.00089`) | `Input` + `LIFNodes` via `Connection` | `examples/benchmark/bindsnet_scaling.py`, `examples/benchmark/benchmark.py` | `python examples/benchmark/bindsnet_scaling.py --csv bindsnet_scaling.csv` | `--seed 0` | Simulation time for 1,000 ms; see [Scaling benchmark](#scaling-benchmark). | synthetic Poisson drive (DATA.md) |
| Atari Breakout (ANN→SNN demo) | trained ANN + SNN pipeline | `examples/breakout/play_breakout_from_ANN.py` | `cd examples/breakout && python play_breakout_from_ANN.py` | `@@seed 42` (default) | GPU (RTX 2070). Reward per episode over the default 10 episodes: 5, 5, 9, 5, 5, 4, 5, 6, 4, 6 (mean **5.4**), identical in two runs. 73 min (the two runs ran at the same time). Writes `play-breakout_results.csv` in the working directory. | Atari Breakout (DATA.md) |

Times are wall-clock times for training plus testing. The four CPU runs and the GPU run
ran at the same time, each pinned to its own 12 CPU cores; a run with the machine to
itself can be faster.

## Measurement setup

- Commit `f392a65` (`git worktree` of the branch, so no uncommitted change was used).
- AMD Ryzen Threadripper 2990WX (32 cores), NVIDIA GeForce RTX 2070 (driver 615.71.09).
- Python 3.11.9, PyTorch 2.14.0 (CUDA 13.0 build), the dependency versions in
  `poetry.lock`.
- MNIST as downloaded by torchvision.

## Notes

### Determinism
- Each MNIST example accepts `--seed` (default `0`) and calls `torch.manual_seed(seed)`,
  which seeds the CPU generator and every GPU (scripts that also draw from numpy seed
  it too), so weight initialisation, Poisson encoding and data shuffling all repeat.
  This holds since 2026-10; earlier releases seeded only the GPU generator on GPU
  runs. The `eth_mnist.py` figure listed in earlier versions of this file (0.81 /
  0.82) came from such a run, and the table above replaces it with a repeatable one.
- Checked with the code of commit `8dd308c`: each of the five MNIST examples, run
  twice with the same command on the GPU, printed identical accuracies (300 training
  and 100 test images; `batch_eth_mnist.py` 640 and 320; `reservoir.py` 100 and 100).
  `eth_mnist.py` and `supervised_mnist.py` also repeat exactly on the CPU, and
  `play_breakout_from_ANN.py` gave the same reward in all 10 episodes in two GPU runs.
  The input-to-excitatory weights of `DiehlAndCook2015` after 20 samples on the GPU
  were bit-for-bit equal across two runs (`torch.equal`).
- CPU and GPU runs with the same seed give different numbers (for example
  `eth_mnist.py` with 300 training images: 0.30 on the CPU, 0.19 on the GPU):
  floating-point sums round differently on the two devices, and learning carries the
  difference forward. Report the device with a result.
- `test/repro/test_smoke_repro.py` runs a tiny network end-to-end on CPU and asserts an
  exact pre-measured output in CI. `test/repro/test_packaging_consistency.py` checks
  that every example with `--seed` seeds torch unconditionally.

### Scaling benchmark
The README and the 2018 paper describe: n Poisson inputs with rates from U(0, 100) Hz,
all-to-all to n LIF neurons, weights from N(0, 1), 1,000 ms at dt = 1.0 ms, n from 250
to 10,000. `examples/benchmark/benchmark.py` runs exactly this setup in every simulator,
with the LIF parameters of BindsNET's `LIFNodes` (rest and reset -65 mV, threshold
-52 mV, membrane time constant 100 ms, refractory 5 ms; input spikes add the weight in
mV to the membrane potential), each simulator in its own process.
`examples/benchmark/bindsnet_scaling.py` runs BindsNET alone. They replace the script
written for the 2018 paper (commit `6a1168d`, first in release 0.2.2), which used its
own input settings for each simulator and a helper package that was not released, so
the published figure (`docs/BindsNET benchmark.png`) stands as the 2018 result;
details in `CHANGELOG.md`.

Measured simulation time (`network.run` or the simulator's equivalent, excluding
network construction) for 1,000 ms, same machine as above. BindsNET from
`bindsnet_scaling.py` in the project environment; the other simulators from
`benchmark.py` in a separate environment (BRIAN2 2.9.0, NEST 3.10.0, ANNarchy 5.0,
PyTorch 2.14.1 CPU build, numpy 2.3.5), each run with the machine to itself.

| Simulator (threads) | n = 1,000 | n = 5,000 | n = 10,000 |
|---------------------|-----------|-----------|------------|
| BindsNET GPU, RTX 2070 (`bindsnet_scaling.py`) | 0.43 s | 0.51 s | 1.27 s |
| BindsNET CPU, PyTorch default 32 threads (`bindsnet_scaling.py`) | 0.32 s | 6.10 s | 17.24 s |
| BindsNET CPU, PyTorch CPU build (`benchmark.py`) | 0.29 s | 5.02 s | 18.54 s |
| BRIAN2, runtime mode, 1 thread | 0.41 s | 6.69 s | 33.51 s |
| PyNEST, 64 threads | 3.71 s | 12.54 s | 22.82 s |
| ANNarchy CPU, 1 thread (its default) | 1.33 s | 36.60 s | 163.54 s |
| ANNarchy CPU, 64 threads (a test with `benchmark.py` changed to 64 threads, not the committed code) | 5.33 s | 149.30 s | 341.38 s |

Each simulator ran with its default thread setting, except NEST with every core; none
was tuned, so the table shows these settings on this machine rather than a ranking of
the simulators. ANNarchy was also tried with all 64 threads, which was slower here and
uneven between sizes (5,000: 149 s, 6,000: 58 s).

brian2genn (needs the GeNN code generator) and ANNarchy on GPU (needs the CUDA
compiler) are included in `benchmark.py` and were not run here. This run confirms that
each simulator executes the network and reports its time; comparing spike statistics
across simulators was outside its scope. Full results:
`examples/benchmark/results_2026-10/`.

### conv_mnist over training
Same command with different training lengths (CPU, seed 0, 1,000 test images unless
noted):

| Training images | All-activity | Proportion-weighting |
|-----------------|--------------|----------------------|
| 1,000 | 0.27 | 0.28 |
| 3,000 | 0.25 | 0.28 |
| 10,000 | 0.15 | 0.17 |
| 20,000 (10,000 test images) | 0.12 | 0.14 |

Logging the convolutional layer during the first 2,750 training images located the
cause: the mean adaptive threshold `theta` kept rising, from 19.4 mV to 68.8 mV, while
recorded spikes per image went from 36.3 to 20.4 and the neurons that fired at all
from 400 to 353. In `DiehlAndCookNodes.forward` (`bindsnet/network/nodes.py`) every
neuron that crosses threshold is reset and raises `theta` before `one_spike` keeps a
single spike per time step, so neurons whose spike is not kept also raise their
threshold. Charging only the neuron that fires was measured with seeds 0, 1 and 2
(20,000 training images, 10,000 test images, all-activity accuracy). `conv_mnist`
averaged 0.17 against 0.15 with the current order, and `eth_mnist` averaged 0.74
against 0.77, lower on every seed. This release keeps the current order; the
`conv_mnist` threshold growth remains open.

### Data
All datasets and synthetic stimuli these scripts use are declared in
[DATA.md](DATA.md), including how they are downloaded and the spike encoding applied.
