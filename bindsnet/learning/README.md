# Learning rules and their sources

Every weight-changing rule in `bindsnet.learning` is validated step by step against
the equations of the paper it implements. The equations are written out with their
numbers in [`docs/source/models_spec.rst`](../../docs/source/models_spec.rst)
("Learning rules") and pinned by the tests in the last column, which drive the rule
with the spike trains the network actually produces and compare the weights on every
step against a from-scratch transcription of the paper.

| Rule (`bindsnet.learning`) | Multicompartment twin (`bindsnet.learning.MCC_learning`) | Paper | Test |
|---|---|---|---|
| `PostPre` | `PostPre` | Morrison, Diesmann & Gerstner (2008), *Biol. Cybern.* 98:459, eqs. 11-14: additive pair STDP with traces | `test/network/test_learning_rule_specs.py` |
| `WeightDependentPostPre` | - | same, soft-bounded: F+ = nu_post (w_max - w), F- = nu_pre (w - w_min) | same |
| `Hebbian` | `Hebbian` | BindsNET's own definition: both trace terms potentiate | same |
| `DiehlAndCook` | `DiehlAndCook` | Diehl & Cook (2015), *Front. Comput. Neurosci.* 9:99, Sect. 2.3: post-spike-only rule dw = eta (x_pre - x_tar)(w_max - w)^mu | same |
| `MSTDP`, `MSTDPET` | `MSTDP`, `MSTDPET` | Florian (2007), *Neural Comput.* 19:1468, eqs. 3.9-3.12 and 2.7-2.8 | `test/network/test_mstdp_florian.py` |
| `Rmax` | - | Vasilaki et al. (2009), *PLoS Comput. Biol.* 5:e1000586, eqs. 7, 8, 13 | `test/network/test_learning_rule_specs.py` |

## Things that are easy to get wrong

- **Reward timing.** The reward passed to `network.run` at a step multiplies the
  eligibility built from the *previous* step's spikes. That is Florian's discrete
  equation 3.9 and the default (`zero_lag=False`). `zero_lag=True` is the un-lagged
  variant.
- **`PostPre` is not the Diehl & Cook rule.** `PostPre` is standard pair STDP: pre
  spikes depress by the post trace, post spikes potentiate by the pre trace. The
  `DiehlAndCook2015` model uses it by default because that is what the published
  BindsNET replication used. The paper's own rule is `DiehlAndCook` (below).
- **`Rmax` `tc_c`.** `tc_c = 0` is the strict policy-gradient rule; `tc_c = inf` is
  naive Hebbian (Vasilaki et al. eq. 8).
- **Traces.** `traces_additive=False` (default) resets the trace to `trace_scale` on
  each spike (Morrison's saturating trace with A = 1); `traces_additive=True` adds
  `trace_scale` per spike (the accumulating trace, which is what Diehl & Cook use).
- **Clamped spikes count.** Spikes forced with `network.run(clamp=...)` enter the
  spike trace and the learning rules like any other spike; `unclamp` removes a
  spike before the trace sees it.
- **No `dt` factor in pair STDP.** `PostPre`, `WeightDependentPostPre`, `Hebbian`
  and `DiehlAndCook` are per-spike increments; the same spike pair changes the
  weight by the same amount at any simulation step. (`MSTDPET` does carry the
  paper's `dt` factor, eq. 2.7.)

## Using the Diehl & Cook (2015) rule

```python
from bindsnet.learning import DiehlAndCook
from bindsnet.network.topology import Connection

conn = Connection(source, target, nu=(0.0, 1e-2), update_rule=DiehlAndCook,
                  wmin=0.0, wmax=1.0, x_tar=0.4, mu=1.0)
```

`source` must record traces; use `traces_additive=True` on the source layer for the
paper's accumulating trace. Only the second learning rate (post-synaptic) is used.
`wmax` must be finite. The paper does not give numbers for `x_tar` and `mu`; the
defaults (0 and 1) are BindsNET's, so set them for your experiment.

With the `DiehlAndCook2015` model:

```python
from bindsnet.learning.MCC_learning import DiehlAndCook
from bindsnet.models import DiehlAndCook2015

net = DiehlAndCook2015(n_inpt=784, n_neurons=100, inpt_shape=(1, 28, 28),
                       learning_rule=DiehlAndCook,
                       learning_rule_kwargs={"x_tar": 0.4, "mu": 1.0})
```

This also switches the input layer to accumulating traces. Leaving `learning_rule`
unset keeps the published `PostPre` behaviour, so results in `REPRODUCING.md` are
unchanged.

For any multicompartment `Weight` feature, extra keyword arguments (`x_tar`, `mu`,
`tc_plus`, ...) are forwarded to its learning rule.
