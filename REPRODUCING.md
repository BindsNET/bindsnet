# Reproducing results with BindsNET

This table traces each model BindsNET describes or ships back to executable code: the
model class, the example script, an exact command, the seed, the measured output, and
the data it needs (declared in [DATA.md](DATA.md)).

> **Honesty note.** Every number in the **Measured output** column was produced by the
> command next to it, at commit `f392a65`, on the machine described under
> [Measurement setup](#measurement-setup). Nothing here is copied from a paper. The
> runs are short (one pass over 20,000 training images, not the paper's full schedule),
> so the accuracies are lower than published ones; they are reference points for
> checking a reproduction, not claims about the models' best accuracy.

> **Which release.** BindsNET is a research package: new features and fixes change
> results between releases, and so can different hardware. Results are not guaranteed
> to stay the same across releases. A published result is reproduced with the release
> it was run on, which the publication should state (see "Citing the software" in the
> README). The numbers below were measured at commit `f392a65`.

## Model → code → command map

Run the MNIST commands from `examples/mnist/` (the scripts read and download MNIST at
`../../data/MNIST`). `CUDA_VISIBLE_DEVICES=` hides the GPU, so the script runs on the
CPU; leave it out to use a GPU. CPU and GPU runs with the same seed give different
numbers (see [Determinism](#determinism)).

| Claim / source | Model class | Script | Command | Seed | Measured output | Data |
|----------------|-------------|--------|---------|------|-----------------|------|
| Diehl & Cook 2015 MNIST replication (DOI `10.3389/fncom.2015.00099`) | `DiehlAndCook2015` | `examples/mnist/eth_mnist.py` | `CUDA_VISIBLE_DEVICES= python eth_mnist.py --n_neurons 100 --n_epochs 1 --time 250 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | CPU. Test accuracy: all-activity **0.75**, proportion-weighting **0.76**. 2.06 h. | MNIST |
| Batched ETH MNIST | `DiehlAndCook2015` | `examples/mnist/batch_eth_mnist.py` | `python batch_eth_mnist.py --n_neurons 100 --batch_size 32 --time 100 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | GPU (RTX 2070). Test accuracy: all-activity **0.81**, proportion-weighting **0.83**. 7.6 min. | MNIST |
| Supervised MNIST (label-clamped) | `DiehlAndCook2015` | `examples/mnist/supervised_mnist.py` | `CUDA_VISIBLE_DEVICES= python supervised_mnist.py --n_neurons 100 --time 250 --intensity 32 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | CPU. Test accuracy: all-activity **0.64**, proportion-weighting **0.64**. 2.06 h. | MNIST |
| Convolutional SNN on MNIST | (in-script conv network) | `examples/mnist/conv_mnist.py` | `CUDA_VISIBLE_DEVICES= python conv_mnist.py --time 50 --batch_size 1 --seed 0 --n_train 20000 --n_test 10000` | `--seed 0` | CPU. Test accuracy: all-activity **0.12**, proportion-weighting **0.14**, near the 0.10 of guessing; accuracy falls as training continues (see [conv_mnist accuracy](#conv_mnist-accuracy-falls-with-training)). 1.58 h. | MNIST |
| Reservoir / liquid-state MNIST | (in-script reservoir) | `examples/mnist/reservoir.py` | `python reservoir.py --n_neurons 500 --n_epochs 100 --time 250 --seed 0` | `--seed 0` | CPU. Readout accuracy on 500 held-out test images: **53.00 %** (trained on 500 images, the script's default `--examples`). 5.8 min. | MNIST |
| Scaling benchmark (Hazan et al. 2018, DOI `10.3389/fninf.2018.00089`) | `Input` + `LIFNodes` via `Connection` | `examples/benchmark/bindsnet_scaling.py`, `examples/benchmark/benchmark.py` | `python examples/benchmark/bindsnet_scaling.py --csv bindsnet_scaling.csv` | `--seed 0` | Simulation time for 1,000 ms; see [Scaling benchmark](#scaling-benchmark). | synthetic Poisson drive (DATA.md) |
| Atari Breakout (ANN→SNN demo) | trained ANN + SNN pipeline | `examples/breakout/play_breakout_from_ANN.py` | `cd examples/breakout && python play_breakout_from_ANN.py` | `@@seed 42` (default) | GPU (RTX 2070). Reward per episode over the default 10 episodes: 5, 5, 9, 5, 5, 4, 5, 6, 4, 6 (mean **5.4**), identical in two runs. 73 min (the two runs ran at the same time). Writes `play-breakout_results.csv` in the working directory. | Atari Breakout (DATA.md) |

Times are wall-clock times for training plus testing. The four CPU runs and the GPU run
ran at the same time, each pinned to its own 12 CPU cores, so a run alone on the
machine may be faster.

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
  it too). Before 2026-10 the GPU path seeded only the GPU generator, so weight
  initialisation, Poisson encoding and data shuffling were not repeatable; the
  published `eth_mnist.py` figure in earlier versions of this file (0.81 / 0.82) came
  from such a run and cannot be repeated.
- Checked with the code of commit `8dd308c`: each of the five MNIST examples, run
  twice with the same command on the GPU, printed identical accuracies (300 training
  and 100 test images; `batch_eth_mnist.py` 640 and 320; `reservoir.py` 100 and 100).
  `eth_mnist.py` and `supervised_mnist.py` also repeat exactly on the CPU, and
  `play_breakout_from_ANN.py` gave the same reward in all 10 episodes in two GPU runs.
  The input-to-excitatory weights of `DiehlAndCook2015` after 20 samples on the GPU
  were bit-for-bit equal across two runs (`torch.equal`).
- CPU and GPU runs with the same seed do not give the same numbers: floating-point
  sums round differently on the two devices, and learning amplifies the difference.
- `test/repro/test_smoke_repro.py` runs a tiny network end-to-end on CPU and asserts an
  exact pre-measured output in CI. `test/repro/test_packaging_consistency.py` fails if
  an example with `--seed` seeds torch only inside a branch.

### Scaling benchmark
The README and the 2018 paper describe: n Poisson inputs with rates from U(0, 100) Hz,
all-to-all to n LIF neurons, weights from N(0, 1), 1,000 ms at dt = 1.0 ms, n from 250
to 10,000. The benchmark script written for the paper (`examples/benchmark/benchmark.py`,
commit `6a1168d`, 2018-11-13, first in release 0.2.2) gave the simulators different input
from that description and imported a package that is not in the repository, so it
cannot regenerate the published figure (`docs/BindsNET benchmark.png`); details in
`CHANGELOG.md`. It was rewritten in 2026-10 so every simulator gets the described
network, with the LIF parameters of BindsNET's `LIFNodes` (rest and reset -65 mV, threshold -52 mV, membrane
time constant 100 ms, refractory 5 ms; input spikes add the weight in mV to the
membrane potential). `examples/benchmark/bindsnet_scaling.py` runs BindsNET alone.

Measured simulation time (`network.run` or the simulator's equivalent, excluding
network construction) for 1,000 ms, same machine as above. BindsNET from
`bindsnet_scaling.py` in the project environment; the other simulators from
`benchmark.py` in a separate environment (BRIAN2 2.9.0, NEST 3.10.0, ANNarchy 5.0,
PyTorch 2.14.1 CPU build, numpy 2.3.5), run alone on the machine.

| Simulator (threads) | n = 1,000 | n = 5,000 | n = 10,000 |
|---------------------|-----------|-----------|------------|
| BindsNET GPU, RTX 2070 (`bindsnet_scaling.py`) | 0.43 s | 0.51 s | 1.27 s |
| BindsNET CPU, PyTorch default 32 threads (`bindsnet_scaling.py`) | 0.32 s | 6.10 s | 17.24 s |
| BindsNET CPU, PyTorch CPU build (`benchmark.py`) | 0.29 s | 5.02 s | 18.54 s |
| BRIAN2, runtime mode, 1 thread | 0.41 s | 6.69 s | 33.51 s |
| PyNEST, 64 threads | 3.71 s | 12.54 s | 22.82 s |
| ANNarchy CPU, 1 thread (its default) | 1.33 s | 36.60 s | 163.54 s |
| ANNarchy CPU, 64 threads (a test with `benchmark.py` changed to 64 threads, not the committed code) | 5.33 s | 149.30 s | 341.38 s |

Thread settings were not tuned. ANNarchy with all 64 threads was slower than with one
and uneven between sizes (5,000: 149 s, 6,000: 58 s); the reason was not investigated.
This table compares these settings on this machine only; it is not a ranking of the
simulators.

Not run: brian2genn (needs the GeNN code generator) and ANNarchy on GPU (needs the
CUDA compiler); `benchmark.py` contains both, untested. The simulators were checked to
run and report times; it was **not** checked that they produce the same spike
statistics for the same input. Full results: `examples/benchmark/results_2026-10/`.

### conv_mnist accuracy falls with training
Same command with fewer training images (CPU, seed 0, 1,000 test images unless noted):

| Training images | All-activity | Proportion-weighting |
|-----------------|--------------|----------------------|
| 1,000 | 0.27 | 0.28 |
| 3,000 | 0.25 | 0.28 |
| 10,000 | 0.15 | 0.17 |
| 20,000 (10,000 test images) | 0.12 | 0.14 |

Logging the convolutional layer during the first 2,750 training images showed the mean
adaptive threshold `theta` rising from 19.4 mV to 68.8 mV without levelling off, while
recorded spikes per image fell from 36.3 to 20.4 and neurons that fired at all fell from
400 to 353. A likely cause: in `DiehlAndCookNodes.forward`
(`bindsnet/network/nodes.py`), every neuron that crosses threshold is reset and has
`theta` raised before `one_spike` keeps a single spike per time step, so neurons whose
spike is discarded still raise their threshold. This has not been changed or tested
further.

### Data
All datasets and synthetic stimuli these scripts use are declared in
[DATA.md](DATA.md), including how they are downloaded and the spike encoding applied.
