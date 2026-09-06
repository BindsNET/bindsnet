# Changelog

All notable changes to BindsNET are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). For releases prior to the entries below,
see the [GitHub releases / tags](https://github.com/BindsNET/bindsnet/releases).

## [Unreleased]

### Added
- Reproducibility/transparency docs: `DATA.md` (dataset & stimulus declaration),
  `REPRODUCING.md` (model→script→command→seed map), and a
  `docs/source/models_spec.rst` neural-model specification page.
- `CITATION.cff` with the paper citation and the Zenodo software DOI.
- `CHANGELOG.md`.
- `examples/breakout/README.md` documenting the `trained_shallow_ANN.pt` provenance.

### Added
- `bindsnet.learning.DiehlAndCook` and `bindsnet.learning.MCC_learning.DiehlAndCook`:
  the post-spike-only STDP of Diehl & Cook (2015), Sect. 2.3,
  `dw = eta (x_pre - x_tar)(w_max - w)^mu` (keyword arguments `x_tar`, `mu`).
  `DiehlAndCook2015(learning_rule=..., learning_rule_kwargs=...)` selects it; the
  model's default stays `PostPre` so published results are unchanged.
- Multicompartment `Weight` features forward extra keyword arguments to their
  learning rule.
- `bindsnet/learning/README.md`: rules, source papers, equation numbers, tests and
  pitfalls (moved from the top-level README).

### Changed
- `MCC_learning.PostPre` no longer multiplies its update by the simulation step
  `dt`; like the classic `PostPre` and Morrison et al. (2008) eqs. 13-14 it is a
  per-spike increment. Identical at `dt = 1`; at other steps the effective learning
  rate is now `nu` instead of `nu * dt`.
- README Python requirement aligned to `>=3.11,<3.14`; added a reproducible-install note.
- `pyproject.toml` version bumped to 0.3.4 to match the released tag.
- Performance pass on the per-timestep hot paths (numerics unchanged; every item
  is pinned by `test/network/test_perf_equivalence.py`, and a seeded old-vs-new
  comparison of 59 networks was bit-identical except three batch>1 weight
  matrices that differ by one float32 rounding step):
  - `PostPre` / `Hebbian` on dense `Connection` and `MulticompartmentConnection`
    apply the STDP update with one fused `addmm_` instead of materialising the
    `[batch, source.n, target.n]` outer product (dense 784->1000 STDP,
    batch 16, 250 steps on CPU: 9.6 s -> 0.42 s; Diehl & Cook 784->400,
    batch 1: 1.6 s -> 0.26 s).
  - `LearningRule.update` no longer multiplies the whole weight matrix by `1.0`
    every step when no weight decay is configured.
  - `LocalConnection1D/2D/3D` learning rules scale rows directly instead of
    building an `[n, n]` identity matrix per step (64-filter local connection on
    GPU: 61 MiB -> 2.8 MiB of per-step temporaries). `MSTDP`/`MSTDPET` keep
    their post-synaptic trace as a `[batch, n, 1]` vector instead of a diagonal
    matrix.
  - `MSTDP` / `MSTDPET` cache `exp(-dt / tc)` and the default learning-rate
    tensors instead of recomputing / re-copying them to the device each step.
  - Neuron models update `v`, `refrac_count`, `theta`, `x`, ... in place with
    the same operations in the same order, avoiding a `Module.__setattr__`
    round-trip per assignment per step.
  - `rank_order` encoding is vectorised.
- Benchmark script for the above: `examples/benchmark/hot_path_bench.py`.
- Learning rules validated against their source papers, with the equations cited in
  `docs/source/models_spec.rst` and pinned by `test/network/test_learning_rule_specs.py`:
  `PostPre` / `WeightDependentPostPre` / `Hebbian` against Morrison, Diesmann &
  Gerstner (2008) eqs. (11)-(14); `MSTDP` / `MSTDPET` against Florian (2007)
  eqs. (3.9)-(3.12) and (2.7)-(2.8) (equation numbers added to
  `test_mstdp_florian.py`); `Rmax` against Vasilaki et al. (2009) eqs. (7), (8), (13).
  The `MCC_learning` `PostPre` / `Hebbian` are checked to match the classic rules.
- Docstrings corrected: `Rmax` `tc_c` limits were stated backwards (`0` is the strict
  policy-gradient rule, `inf` the naive Hebbian rule, Vasilaki et al. eq. 8); the
  `MSTDP` / `MSTDPET` `zero_lag` comments called the un-lagged variant "exact Florian",
  whereas the default one-step lag is Florian's discrete-time eq. (3.9).

### Fixed
- `network.run(clamp=...)` / `unclamp` are now applied inside `Nodes.forward` before
  the spike trace is updated, so a forced spike leaves a trace (and a suppressed one
  does not). Previously the clamp was applied after the trace update, so clamped
  spikes entered the same-step potentiation term of STDP rules but never the trace
  used by later depression terms (affected `examples/mnist/supervised_mnist.py`).
  Pinned by `TestClampEntersTraces` and the clamp-driven STDP window test.
- `network.to(device)` crashed on any `MulticompartmentConnection` (used by
  `DiehlAndCook2015`) with `_apply() takes 2 positional arguments but 3 were
  given`; `AbstractMulticompartmentConnection._apply` now accepts `recurse`.

## [0.3.4] - 2026-06-15

Archived on Zenodo — concept DOI [10.5281/zenodo.20695115](https://doi.org/10.5281/zenodo.20695115),
version DOI [10.5281/zenodo.20695116](https://doi.org/10.5281/zenodo.20695116).

### Added
- Sparse-tensor support for additional learning rules (plus a batch dimension and docs
  for `sparse=True`).
- Validation tests for the reward-modulated learning rules `MSTDP` and `MSTDPET`.
- Regression test for a preallocated `Monitor` short-run bug (PR #761).
- Read the Docs configuration for documentation builds.

### Changed
- `assign_labels` / evaluation: handle abstention for inactive samples, mark
  never-firing neurons with `-1`, and accuracy/performance improvements.
- CI: dropped Python 3.10 (project requires `>=3.11`); upgraded GitHub Actions; test on
  Python 3.11/3.12/3.13.
- Routine dependency updates via Poetry.

### Fixed
- `bernoulli_loader` now honors the `max_prob` kwarg (PR #743).
- Bug with preallocated buffers and `torch.cat`.
- `torch.save` compatibility for PyTorch 2.6.0.
- Python 3.13 support / tests.
- `eth_mnist` example.

## [0.3.3] - 2024-10-18

Baseline for this changelog. See the
[releases page](https://github.com/BindsNET/bindsnet/releases) for the history of
0.1.x–0.3.3.
