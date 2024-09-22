.. _guide_part_ii:

Part II: Creating and Adding Learning Rules
===========================================

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

At present, learning rules are attached to specific :code:`Connection` objects. For
example, to create a connection with a STDP learning rule on the synapses:

.. code-block:: python

    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection
    from bindsnet.learning import PostPre

    # Create two populations of neurons, one to act as the "source"
    # population, and the other, the "target population".
    # Neurons involved in certain learning rules must record synaptic
    # traces, a vector of short-term memories of the last emitted spikes.
    source_layer = Input(n=100, traces=True)
    target_layer = LIFNodes(n=1000, traces=True)

    # Connect the two layers.
    connection = Connection(
        source=source_layer, target=target_layer, update_rule=PostPre, nu=(1e-4, 1e-2)
    )

The connection may be added to a :code:`Network` instance as usual. The :code:`Connection` object
takes arguments :code:`update_rule`, of type :code:`bindsnet.learning.LearningRule`, as well
as :code:`nu`, a 2-tuple specifying pre- and post-synaptic learning rates; i.e., multiplicative
factors which modulate how quickly synapse weights change.

Learning rules also accept arguments :code:`reduction`, which specifies how parameter updates are
aggregated across the batch dimension, and :code:`weight_decay`, which specifies the time constant
of the rate of decay of synapse weights to zero. By default, parameter updates are averaged across
the batch dimension, and there is no weight decay.

Other supported learning rules include :code:`Hebbian`, :code:`WeightDependentPostPre`,
:code:`MSTDP` (reward-modulated STDP), and :code:`MSTDPET` (reward-modulated STDP with
eligibility traces).

Custom learning rules can be implemented by subclassing :code:`bindsnet.learning.LearningRule`
and providing implementations for the types of :code:`AbstractConnection` objects intended to be used.
For example, the :code:`Connection` and :code:`LocalConnection` objects rely on the implementation
of a private method, :code:`_connection_update`, whereas the :code:`Conv2dConnection` object
uses the :code:`_conv2d_connection_update` version.

If using a MulticompartmentConneciton, you can add a learning rule to a specific feature. Note that only
:code:`NoOp`, :code:`PostPre`, :code:`MSTDP`, :code:`MSTDPET` are supported, and located at 
bindsnet.learning.MCC_learning. Below is an example of how to apply a PostPre learning rule to a weight function.
Note that the bias does not have a learning rule, so it will remain static.

.. code-block:: python

    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import MulticompartmentConnection
    from bindsnet.learning.MCC_learning import PostPre

    # Create two populations of neurons, one to act as the "source"
    # population, and the other, the "target population".
    # Neurons involved in certain learning rules must record synaptic
    # traces, a vector of short-term memories of the last emitted spikes.
    source_layer = Input(n=100, traces=True)
    target_layer = LIFNodes(n=1000, traces=True)

    # Create 'pipeline' of features that spikes will pass through
    weights = Weight(name='weight_feature', value=torch.rand(100, 1000),
                      learning_rule=PostPre, nu=(1e-4, 1e-2))
    bias = Bias(name='bias_feature', value=torch.rand(100, 1000))

    # Connect the two layers.
    connection = MulticompartmentConnection(
        source=source_layer, target=target_layer,
        pipeline=[weights, bias])
    )