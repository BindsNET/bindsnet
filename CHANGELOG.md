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

### Changed
- README Python requirement aligned to `>=3.11,<3.14`; added a reproducible-install note.
- `pyproject.toml` version bumped to 0.3.4 to match the released tag.

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
