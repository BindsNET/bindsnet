# Changelog

All notable changes to BindsNET are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/). For releases prior to the entries below,
see the [GitHub releases / tags](https://github.com/BindsNET/bindsnet/releases).

## [Unreleased]

### Fixed
- API reference on Read the Docs was empty: the build never installed `bindsnet`, so
  every `automodule` failed to import (`No module named 'matplotlib'`), and
  `docs/pyproject.toml` downgraded Sphinx to 7.2.6. `.readthedocs.yaml` now installs a
  CPU build of torch and the package (Python 3.13, Ubuntu 24.04);
  `docs/pyproject.toml` removed.
- The docs build has no warnings (was 48 on Read the Docs, 53 with the package
  installed): docstring markup fixed in `topology.py`, `topology_features.py`,
  `monitors.py`, `learning.py`, `nodes.py`, `encoders.py`, `plotting.py`,
  `conversion.py`, `davis.py`, `preprocess.py`, `cue_reward.py`, `dot_simulator.py`;
  broken links in `index.rst` and the guide; `conf.py` takes the version from the
  installed package. Docstring text only; no code changed.
- API reference now includes `learning.MCC_learning`, `network.topology_features`,
  `environment.cue_reward`, `environment.dot_simulator` and
  `analysis.dotTrace_plotter`, which were missing.

### Changed
- CI: one test workflow (`python-app.yml`: job `build` on Python 3.13 plus a 3.11/3.12
  matrix, Poetry 2.4.3 with dependency caching, superseded runs cancelled);
  `pythonpackage.yml` removed (it ran on every push to every branch and its
  `black .` step reformatted instead of checking). `black.yml` checks with the black
  version from `poetry.lock` instead of the floating `psf/black@stable`.
- Dependabot also updates the Dockerfile base image.
- Added `.pre-commit-config.yaml` (black from Poetry); `CONTRIBUTING.md` already told
  contributors to install pre-commit, but there was no configuration.
- `[tool.black] target-version` is `py311`-`py313` (was `py38`); no file changes.
- README: dead link to Markram et al. (1997) replaced with its DOI; RL example named
  correctly (Breakout, not Space Invaders); OpenAI gym text replaced (Gymnasium and
  ale-py install with BindsNET); benchmark marked as from the 2018 paper; PyPI badge
  refreshes hourly.

## [0.3.4 (PyPI)] - 2026-09-16

First PyPI upload since 0.2.7. It is built from the `master` branch on this date, not
from the GitHub tag `0.3.4`, so `pip install bindsnet==0.3.4` contains everything in
this section **in addition to** the tag. The Zenodo archive
[10.5281/zenodo.20695116](https://doi.org/10.5281/zenodo.20695116) is the tag only.
Results can differ between the two: see the `MCC_learning.PostPre` entry under Changed.

### Packaging
- Published to PyPI by `.github/workflows/publish.yml` (PyPI trusted publishing; runs
  when a GitHub Release is published, or by hand).
- Removed install requirements that no module in `bindsnet/` or `examples/` imports:
  `Cython`, `scikit-build`, `foolbox`, `numba`.
- `torch` is now `>=2.14,<3` and `torchvision` `>=0.29,<1` instead of exact pins;
  `poetry.lock` still pins the tested versions (torch 2.14.0, torchvision 0.29.0).
- README: `pip install bindsnet`, PyPI badge, and absolute links and logo URL so the
  PyPI project page renders.
- `pyproject.toml` package metadata moved from `[tool.poetry]` to the standard
  `[project]` table (Poetry 2 deprecates the old one); `[tool.poetry]` now only routes
  torch/torchvision to the CUDA 13.0 wheel index. `poetry.lock` resolves to the same
  packages and versions. Build backend `poetry-core>=2.0`; the unused `setup.py` is removed.
- Poetry 2.4.3 in CI (was 2.1.2) and in `CONTRIBUTING.md` (said 1.1.8; `poetry shell`
  replaced by `poetry env activate`, which Poetry 2 uses).
- `Dockerfile` rewritten. The old one could not build: its CUDA 11.1 base image, the
  `get-poetry.py` installer and the `.python-version` file it copied no longer exist.
  The new one uses `python:3.13-slim`, Poetry 2.4.3 and `poetry.lock`. The README no
  longer links the Docker Hub image `hqkhan/bindsnet` (last updated 2019-01-28).
- Remaining links to the old `Hananel-Hazan/bindsnet` repository point to `BindsNET/bindsnet`.

### Tests
- `test_perf_equivalence.py`: the batch-1 checks of the fused `addmm_` STDP update
  required bit-for-bit equality with the un-fused formula. That holds on some CPUs
  and not on GitHub's CI runners, where 5 tests failed. They now accept float32
  rounding (the tolerance already used for batch>1) and warn with the size of any
  difference.

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
- `Network.clone()` was broken outright: it called `torch.load` without
  `weights_only=False`, so it raised `UnpicklingError` under PyTorch 2.6+, which
  changed that default to `True`. It had no test and no caller in the tree, so the
  breakage went unnoticed. Pinned by `TestNetwork.test_clone`.
- `Network.save()` called `torch.serialization.add_safe_globals([self])` with a
  network instance where PyTorch expects a class. It did nothing useful and
  corrupted PyTorch's safe-globals registry, so any later load in the same process
  failed with `'Network' object has no attribute '__qualname__'`. Removed. Pinned by
  `TestNetwork.test_clone_after_save`.
- `bindsnet.conversion.ann_to_snn` and `data_based_normalization` were broken when
  given a path instead of a `torch.nn.Module`, for the same PyTorch 2.6 reason as
  `Network.clone()`. Only the in-memory form was tested. Pinned by
  `test_conversion_from_path` and `test_data_based_normalization_from_path`.

### Security
- Documented that loading a saved network runs code. `bindsnet.network.load`,
  `bindsnet.conversion.ann_to_snn` and `bindsnet.conversion.data_based_normalization`
  read Python pickle files via `torch.load`, so a file from an untrusted source can
  execute arbitrary code on load. This is the standard behaviour of `torch.load`
  across the PyTorch ecosystem and is not a defect specific to BindsNET, but it was
  undocumented. Added warnings to each function's docstring and a "Loading saved
  networks and models" section to `SECURITY.md`.
- `bindsnet.network.load` gained a `weights_only` parameter, passed through to
  `torch.load`. It defaults to `False`, which is required to read files written by
  `Network.save` (those store the whole network object, not a tensor state dict), so
  behaviour is unchanged. `weights_only=True` refuses code execution and is usable
  only for files holding plain tensors.
- `SpokenMNIST` now reads its processed-data cache with `weights_only=True`. That
  cache holds only tensors, so refusing code execution there costs nothing. Pinned by
  `test/datasets/test_cache_serialization.py`.
- Reported by Gavin Branaa <gbranaa4@gmail.com>, who also prompted the three
  `torch.load` fixes listed under Fixed above. Thank you.

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
