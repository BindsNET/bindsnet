.. _guide_part_ii:

Part II: Creating and Adding Learning Rules
===========================================

.. note::

    Code enclosed in angle brackets (:code:`<example>`) refers to a placeholder value. Method arguments of the form
    :code:`arg: Type` denote type annotations.

What is considered a learning rule?
-----------------------------------

Learning rules are necessary for the automated adaption of network parameters during simulation. At present, BindsNET
supports two different categories of learning rules:

- **Two factor**: Associative learning takes place based on pre- and post-synaptic neural activity. Examples include:
    - The typical example is `Hebbian learning <https://en.wikipedia.org/wiki/Hebbian_theory>`_, which may be
      summarized as "Cells that fire together wire together." That is, co-active neurons causes their connection
      strength to increase.
    - `Spike-timing-dependent plasticity <http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity>`_
      (STDP) stipulates that the ordering of pre- and post-synaptic spikes matters. A synapse is strengthened if the
      pre-synaptic neuron fires *before* the post-synaptic neuron, and, conversely, is weakened if it fires *after* the
      post-synaptic neuron. The magnitude of these updates is a decreasing function of the time between pre- and
      post-synaptic spikes.
- **Three factor**: In addition to associating pre- and post-synaptic neural activity, a third factor is introduced which modulates plasticity on a more global level; e.g., for all synapses in the network. Examples include:
    - `(Reward, error, attention)-modulated (Hebbian learning, STDP) <https://www.sciencedirect.com/science/article/pii/S0959438817300612>`_:
      The same learning rules described above are modulated by the presence of global signals such as reward, error, or
      attention, which can be variously defined in machine learning or reinforcement learning contexts. These signals
      act to gate plasticity, turning it on or off and switching its sign and magnitude, based on the task at hand.

The above are examples of local learning rules, where the information needed to make updates are thought to be available
at the synapse. For example, pre- and post-synaptic neurons are adjacent to synapses, rendering their spiking activity
accessible, whereas chemical signals like dopamine (hypothesized to be a reward prediction error (RPE) signal) are
widely distributed across certain neuron populations; i.e., they are *globally* available. This is in contrast to
learning algorithms like back-propagation, where per-synapse error signals are derived by computing backwards from a
loss function at the network's output layer. Such error derivation is thought to be biologically implausible, especially
compared to the two- and three-factor rules mentioned above.

Creating a learning rule in BindsNET
------------------------------------


