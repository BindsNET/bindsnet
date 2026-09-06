# BindsNET project notes

## Python environment

This project runs in the conda environment named `bindsNET`.

- Interpreter: `/home/hananel/miniconda3/envs/bindsNET/bin/python`
- Activate: `conda activate bindsNET`
- The package is installed in editable mode from this directory, so edits to
  `bindsnet/` are picked up without reinstalling.
- Torch 2.14 with CUDA 13.0 is installed there. Do not use the `base` env.

Run tests with:

```shell
conda run -n bindsNET python -m pytest -q
```

Run a single file with:

```shell
conda run -n bindsNET python -m pytest -q test/network/test_learning.py
```

## Committing

Use `sc "message"` instead of `git commit` (see the user's global instructions).

## Learning rules: sources and how to validate

Each rule is pinned to its paper's equations by a from-scratch reference test.
Read these before touching any rule:

| Rule | Paper and equations | Test |
|---|---|---|
| `PostPre`, `WeightDependentPostPre`, `Hebbian` | Morrison, Diesmann & Gerstner 2008, *Biol. Cybern.* 98:459, eqs. 11-14 (traces Sect. 2.3) | `test/network/test_learning_rule_specs.py` |
| `DiehlAndCook` (classic and MCC) | Diehl & Cook 2015, *Front. Comput. Neurosci.* 9:99, Sect. 2.3: post-spike-only, dw = eta (x_pre - x_tar)(w_max - w)^mu | `test/network/test_learning_rule_specs.py` |
| `MSTDP`, `MSTDPET` | Florian 2007, *Neural Comput.* 19:1468, eqs. 3.9-3.12, 2.7-2.8 | `test/network/test_mstdp_florian.py` |
| `Rmax` | Vasilaki et al. 2009, *PLoS Comput. Biol.* 5:e1000586, eqs. 7, 8, 13 | `test/network/test_learning_rule_specs.py` |

Facts that were wrong in docstrings once and are now fixed (do not reintroduce):
- `MSTDP` default `zero_lag=False` **is** Florian's discrete eq. 3.9 (reward at a
  step multiplies the previous step's eligibility). `zero_lag=True` is the
  un-lagged variant, not "exact Florian".
- `Rmax` `tc_c = 0` is strict policy gradient; `inf` is naive Hebbian.
- `clamp` spikes enter the trace (applied in `Nodes.forward` before the trace update).

- Pair STDP rules carry no `dt` factor (the MCC `PostPre` used to; removed
  2026-09-06). `MSTDPET` keeps the paper's `dt` (Florian eq. 2.7).
- `PostPre` is not Diehl & Cook 2015's rule; the paper's rule is `DiehlAndCook`.
  `DiehlAndCook2015` keeps `PostPre` as default (published replication); opt in
  with `learning_rule=MCC_learning.DiehlAndCook`.

User-facing summary of all this: `bindsnet/learning/README.md` (keep it in sync
with `docs/source/models_spec.rst`).

## Performance changes: the rule

Any change to a per-timestep path must ship with a test that pins it to the
formula it replaced (`test/network/test_perf_equivalence.py`). Before claiming
"no change in results", run the same seeded networks on the old code (a git
worktree of the previous commit) and the new code and compare with
`torch.equal`; only batch>1 summation-order differences (about 1e-7) are
acceptable, and must be stated. Benchmark: `examples/benchmark/hot_path_bench.py`.
