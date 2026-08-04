Neural model specifications
===========================

This page gives the **mathematical specification** of the neuron, and learning-rule
models BindsNET implements: the difference equations actually solved each timestep and
the default parameters with units. Equations and defaults below were transcribed from
the source (``bindsnet/network/nodes.py`` and ``bindsnet/learning/learning.py``); when
in doubt, the source is authoritative. Parameter defaults are stated as of package
version 0.3.4.

Discretization
--------------

BindsNET does not integrate ODEs symbolically. Continuous-time dynamics are converted to
**difference equations** and advanced at a fixed timestep :math:`dt` (milliseconds; the
examples use :math:`dt = 1.0`). A network-wide :math:`dt` is set on the
``Network`` object. Exponential leak terms are precomputed once per :math:`dt` as a
decay factor

.. math::

   \text{decay} = \exp\!\left(-\,dt / \tau\right),

so a leaky variable :math:`y` relaxing toward a baseline :math:`y_0` updates as
:math:`y \leftarrow \text{decay}\,(y - y_0) + y_0`.

Notation: :math:`v` membrane voltage, :math:`v_\text{rest}` rest, :math:`v_\text{reset}`
post-spike reset, :math:`v_\text{thr}` threshold, :math:`s` spike (boolean),
:math:`x` spike trace, :math:`\tau` a time constant. All voltages are in millivolts and
follow the biological convention used in the code (e.g. rest :math:`-65`\ mV).

Neuron models
-------------

All neuron layers live in ``bindsnet.network.nodes``. Spikes are emitted when the
(possibly adapted) threshold is crossed; most models then apply a reset and a refractory
period ``refrac`` during which inputs are ignored. Optional spike **traces** decay with
time constant ``tc_trace`` (default 20 ms) and are used by the trace-based learning
rules.

McCulloch–Pitts (``McCullochPitts``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Stateless threshold unit; the voltage equals the input and a spike is emitted when it
reaches threshold.

.. math::

   v_t = x_t, \qquad s_t = \big[\,v_t \ge v_\text{thr}\,\big]

Defaults: ``thresh`` :math:`= 1.0`.

Integrate-and-fire (``IFNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-leaky accumulator with reset and refractory period.

.. math::

   v_t = v_{t-1} + \mathbb{1}[\text{refrac}\le 0]\,x_t, \qquad
   s_t = [\,v_t \ge v_\text{thr}\,], \qquad v_t \leftarrow v_\text{reset}\ \text{if}\ s_t

Defaults: ``thresh`` :math:`=-52`, ``reset`` :math:`=-65`, ``refrac`` :math:`=5`.

Leaky integrate-and-fire (``LIFNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Leak toward rest, integrate input, threshold-reset-refractory.

.. math::

   v_t = \text{decay}\,(v_{t-1} - v_\text{rest}) + v_\text{rest} + x_t,
   \qquad \text{decay} = \exp(-dt/\tau_\text{decay})

.. math::

   s_t = [\,v_t \ge v_\text{thr}\,], \qquad v_t \leftarrow v_\text{reset}\ \text{if}\ s_t

Defaults: ``thresh`` :math:`=-52`, ``rest`` :math:`=-65`, ``reset`` :math:`=-65`,
``refrac`` :math:`=5`, ``tc_decay`` :math:`=100`\ ms. Inputs are masked to zero while a
neuron is refractory.

``BoostedLIFNodes`` is a performance-oriented LIF variant (per source: no separate
rest/reset/lower-bound handling); use it when those features are not needed.

Current-based LIF (``CurrentLIFNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adds a decaying synaptic **current** :math:`i` between input and membrane:

.. math::

   i_t = i_\text{decay}\,i_{t-1} + x_t, \qquad
   v_t = \text{decay}\,(v_{t-1} - v_\text{rest}) + v_\text{rest}
         + \mathbb{1}[\text{refrac}\le 0]\,i_t

with :math:`i_\text{decay} = \exp(-dt/\tau_{i})`. See source for the ``tc_i_decay``
default and the remaining (LIF-shared) parameters.

Adaptive-threshold LIF (``AdaptiveLIFNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LIF with a threshold adaptation variable :math:`\theta` that increases on each spike and
decays otherwise:

.. math::

   \theta_t = \theta_\text{decay}\,\theta_{t-1} + \theta_+ \textstyle\sum s_t,
   \qquad s_t = [\,v_t \ge v_\text{thr} + \theta_t\,]

Defaults add ``theta_plus`` :math:`=0.05`, ``tc_theta_decay`` :math:`=10^{7}`\ ms (on top
of the LIF defaults). Adaptation is applied while ``learning`` is enabled.

Diehl & Cook 2015 (``DiehlAndCookNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adaptive-threshold LIF tuned for the Diehl & Cook (2015) MNIST replication, with the
additional ``one_spike`` option (default ``True``) that permits at most one spike per
layer per timestep. Same parameter defaults as ``AdaptiveLIFNodes``. Used by the
``DiehlAndCook2015`` model and ``examples/mnist/eth_mnist.py``.

Izhikevich (``IzhikevichNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two-variable model (membrane :math:`v`, recovery :math:`u`) integrated with two half
Euler steps per timestep:

.. math::

   v \leftarrow v + \tfrac{dt}{2}\,(0.04 v^2 + 5 v + 140 - u + x)\quad(\text{applied twice}),
   \qquad u \leftarrow u + dt\,a\,(b v - u)

On spike (:math:`v \ge v_\text{thr}`): :math:`v \leftarrow c`, :math:`u \leftarrow u + d`.
Excitatory/inhibitory populations are parameterized as in Izhikevich (2003): excitatory
:math:`a=0.02,\,b=0.2,\,c=-65+15r^2,\,d=8-6r^2`; inhibitory
:math:`a=0.02+0.08r,\,b=0.25-0.05r,\,c=-65,\,d=2`, with :math:`r\sim U(0,1)`.

SRM0 (``SRM0Nodes``)
~~~~~~~~~~~~~~~~~~~~~
Simplified Spike Response Model with **stochastic** ("escape noise") firing:

.. math::

   v_t = \text{decay}\,(v_{t-1}-v_\text{rest}) + v_\text{rest}
         + \mathbb{1}[\text{refrac}\le 0]\,\varepsilon_0\,x_t

.. math::

   \rho = \rho_0 \exp\!\Big(\tfrac{v_t - v_\text{thr}}{\Delta_\text{thr}}\Big),
   \qquad P(\text{spike}) = 1 - e^{-\rho\,dt},
   \qquad s_t = [\,U(0,1) < P(\text{spike})\,]

Cumulative SRM (``CSRMNodes``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cumulative Spike Response Model (Gerstner & van Hemmen 1992; Gerstner et al. 1996):
refractoriness and adaptation arise from the summed after-potentials of several previous
spikes rather than only the most recent one. See the source for the response-kernel
implementation.

Input (``Input``)
~~~~~~~~~~~~~~~~~
Passes externally provided spike tensors (e.g. from ``bindsnet.encoding``) into the
network; it has no internal membrane dynamics.

Learning rules
--------------

Learning rules live in ``bindsnet.learning``. They modify connection weights ``w`` from
pre-synaptic spikes/traces (``source``) and post-synaptic spikes/traces (``target``).
``nu`` is the (pre, post) learning-rate pair; ``reduction`` aggregates over the batch;
``weight_decay`` optionally decays weights each step.

Post-pre STDP (``PostPre``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Trace-based spike-timing-dependent plasticity (requires traces on both layers). For a
dense ``Connection`` the per-step update is

.. math::

   \Delta w = -\,\nu_\text{pre}\;(s_\text{pre} \otimes x_\text{post})
              \;+\; \nu_\text{post}\;(x_\text{pre} \otimes s_\text{post})

i.e. a pre-synaptic spike **depresses** the synapse in proportion to the post-synaptic
trace, and a post-synaptic spike **potentiates** it in proportion to the pre-synaptic
trace. Convolutional and locally-connected variants apply the same rule patch-wise.

Hebbian (``Hebbian``)
~~~~~~~~~~~~~~~~~~~~~~
Both pre- and post-synaptic events **increase** the weight (no depression term),
proportional to the opposite layer's trace.

Weight-dependent post-pre (``WeightDependentPostPre``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``PostPre`` whose potentiation/depression magnitudes are scaled by the distance of the
weight from its bounds (``wmin``/``wmax``), yielding soft saturation at the limits.

Reward-modulated STDP (``MSTDP``, ``MSTDPET``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Three-factor rules: a STDP-like eligibility signal is gated by a scalar **reward**.
``MSTDP`` modulates the immediate pre/post correlation by reward; ``MSTDPET`` adds an
**eligibility trace** that accumulates the correlation over time (time constant
``tc_e_trace``) before reward gating. Reward is supplied via the pipeline / an
``AbstractReward`` (e.g. ``MovingAvgRPE``). See source for the exact eligibility update.

Rmax (``Rmax``)
~~~~~~~~~~~~~~~
Reward-maximizing rule intended for stochastic (SRM0) neurons; see source for its
formulation.

.. note::

   Where this page summarizes a rule "see source", the equations were not reproduced here
   to avoid mis-stating constants; consult ``bindsnet/learning/learning.py`` for the
   authoritative form. If an implementation deviates from a textbook model, the code is
   the specification.
