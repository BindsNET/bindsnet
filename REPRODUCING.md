# Reproducing results with BindsNET

This table traces each model BindsNET describes or ships back to executable code: the
model class, the example script, an exact command, the seed, the expected output, and
the data it needs (declared in [DATA.md](DATA.md)).

> **Honesty note.** Commands, defaults, seeds, and model classes below are verified
> against the source. The **Expected output** cells describe *what the script reports*
> and the qualitative trend; cells marked *(not measured here)* have **not** been run to
> a final metric in producing this table — run the command to obtain the number for your
> hardware. No accuracy/timing figure is asserted that was not measured.

## Model → code → command map

| Claim / source | Model class | Script | Command (defaults shown) | Seed | Expected output | Data |
|----------------|-------------|--------|--------------------------|------|-----------------|------|
| Diehl & Cook 2015 MNIST replication (DOI `10.3389/fncom.2015.00099`) | `DiehlAndCook2015` | `examples/mnist/eth_mnist.py` | `python examples/mnist/eth_mnist.py --n_neurons 100 --n_epochs 1 --time 250 --seed 0` | `--seed 0` (`torch.manual_seed`) | Prints test accuracy at end. **Measured:** all-activity **0.81**, proportion-weighting **0.82** at seed 0 with `--n_train 20000 --n_test 10000` (GPU, torch 2.6, ~7.8 h on an RTX 2070). Accuracy rises with `--n_neurons` and with the full 60000-sample train set (Diehl & Cook report up to ~95% at 6400 neurons). | MNIST |
| Batched ETH MNIST | `DiehlAndCook2015` | `examples/mnist/batch_eth_mnist.py` | `python examples/mnist/batch_eth_mnist.py --n_neurons 100 --batch_size 32 --time 100 --seed 0` | `--seed 0` | Prints test accuracy; faster per-epoch via batching. *(not measured here)* | MNIST |
| Supervised MNIST (label-clamped) | `DiehlAndCook2015` | `examples/mnist/supervised_mnist.py` | `python examples/mnist/supervised_mnist.py --n_neurons 100 --time 250 --intensity 32 --seed 0` | `--seed 0` | Prints test accuracy. *(not measured here)* | MNIST |
| Convolutional SNN on MNIST | (in-script conv network) | `examples/mnist/conv_mnist.py` | `python examples/mnist/conv_mnist.py --time 50 --batch_size 1 --seed 0` | `--seed 0` | Prints accuracy. *(not measured here)* | MNIST |
| Reservoir / liquid-state MNIST | (in-script reservoir) | `examples/mnist/reservoir.py` | `python examples/mnist/reservoir.py --n_neurons 500 --n_epochs 100 --time 250 --seed 0` | `--seed 0` | Prints accuracy after readout training. *(not measured here)* | MNIST |
| Scaling benchmark (Hazan et al. 2018, DOI `10.3389/fninf.2018.00089`) | `Input` + `LIFNodes` via `Connection` | `examples/benchmark/benchmark.py` | **Not single-command** — see note below | n/a (timing study) | Runtime-vs-`n` curve; published figure is `docs/BindsNET benchmark.png` | synthetic Poisson drive (DATA.md) |
| Atari Breakout (ANN→SNN demo) | trained ANN + SNN pipeline | `examples/breakout/play_breakout_from_ANN.py` | `python examples/breakout/play_breakout_from_ANN.py` | set in script | Plays Breakout from the shipped `trained_shallow_ANN.pt` | Atari Breakout (DATA.md) |

## Notes

### Determinism
- Each MNIST example accepts `--seed` (default `0`) and calls `torch.manual_seed(seed)`
  and `torch.cuda.manual_seed_all(seed)`. Pass the same `--seed` to repeat a run.
- Residual nondeterminism can come from CUDA atomic operations and first-run dataset
  download ordering. For stricter determinism run on CPU and, where feasible, set
  `torch.use_deterministic_algorithms(True)`.
- An automated, seeded smoke-reproduction test
  (`test/repro/test_smoke_repro.py`) runs a tiny network end-to-end on CPU and asserts
  an exact pre-measured output, so determinism is checked continuously in CI.

### Scaling benchmark is a multi-simulator study
`examples/benchmark/benchmark.py` compares BindsNET against **BRIAN2, PyNEST, ANNarchy,
BRIAN2genn, and Nengo**, and imports those packages plus an `experiments` helper module.
It is therefore **not** a single-command reproduction: it requires those external
simulators installed and the benchmark harness. The published BindsNET result is the
figure `docs/BindsNET benchmark.png` and the parameters in the README "Benchmarking"
section (Poisson inputs U(0,100) Hz, weights N(0,1), dt = 1.0 ms, 1000 ms/run, n from
250 to 10,000). A BindsNET-only timing reproduction (no external simulators) can be
built from `Input` + `LIFNodes` + `Connection`.

### Data
All datasets and synthetic stimuli these scripts use are declared in
[DATA.md](DATA.md), including how they are downloaded and the spike encoding applied.
