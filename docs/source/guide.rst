BindsNET User Manual
====================


Creating a Network
------------------

The :class:`bindsnet.network.Network` object is BindsNET's main offering. To create one:

.. code-block:: python

    from bindsnet.network import Network

    network = Network()

The :py:class:`bindsnet.network.Network` object accepts optional keyword arguments :code:`dt: float`,
:code:`learning: bool`, and :code:`reward_fn: Optional[bindsnet.learning.reward.AbstractReward]`.

The :code:`dt` argument specifies the simulation time step, which determines what temporal granularity, in milliseconds,
simulations are solved at. All simulation is done with the Euler method for the sake of computational simplicity. If
inaccuracy in simulation is encountered, use a smaller :code:`dt` to resolve numerical instability.

The :code:`learning` argument acts to enable or disable updates to adaptive parameters of network components; e.g.,
synapse weights or adaptive voltage thresholds. See `Using Learning Rules`_ for more details.

The :code:`reward_fn` argument takes in class that specifies how a scalar reward signal will be computed and fed to the
network and its components. Typically, the output of this callable class will be used in certain "reward-modulated", or
"three-factor" learning rules. See `Using Learning Rules`_ for more details.


Adding Network Components
-------------------------

BindsNET supports three categories of network component: *layers* of neurons (:code:`nodes`), *connections* between them
(:py:mod:`bindsnet.network.topology`), and *monitors* for recording the evolution of state variables
(:py:mod:`bindsnet.network.monitors`).

Creating and adding layers
**************************

To create a layer (or *population*) of nodes (in this case, leaky integrate-and-fire (LIF) neurons:

.. code-block:: python

    from bindsnet.network.nodes import LIFNodes

    layer = LIFNodes(
        n=100,
        shape=(10, 10).
        ...
    )

Each :py:mod:`bindsnet.network.nodes` object has many keyword arguments, but one of either :code:`n` (the number of
nodes in the layer, or :code:`shape` (the arrangement of the layer, from which the number of nodes can be computed) is
required.

To add a layer to the network, use the :code:`add_layer` function, and give it a name (a string) to call it by:

.. code-block:: python

    network.add_layer(
        layer=layer,
        name='LIF population'
    )

Such layers are kept in the dictionary attribute :code:`network.layers`, and can be accessed by the user; e.g., by
:code:`network.layers['LIF population']`.

Other layer types include :py:class:`bindsnet.network.nodes.Input` (for user-specified input spikes),
:py:class:`bindsnet.network.nodes.RealInput` (for user-specified real-valued inputs),
:py:class:`bindsnet.network.nodes.McCullochPitts` (the McCulloch-Pitts neuron model),
:py:class:`bindsnet.network.nodes.AdaptiveLIFNodes` (LIF neurons with adaptive thresholds), and
:py:class:`bindsnet.network.nodes.IzhikevichNodes` (the Izhikevich neuron model). Any number of layers can be
added to the network.

Custom nodes objects can be implemented by sub-classing :py:class:`bindsnet.network.nodes.Nodes`, an abstract class with
common logic for neuron simulation. The functions :code:`forward(self, x: torch.Tensor)` (computes effects of input
data on neuron population; e.g., voltage changes, spike occurrences, etc.), :code:`reset_(self)` (resets neuron state
variables to default values), and :code:`_compute_decays(self)` must be implemented, as they are included as abstract
functions of :py:class:`bindsnet.network.nodes.Nodes`.

Creating and adding connections
*******************************

Connections can be added between different populations of neurons (a *projection*), or from a population back to itself
(a *recurrent* connection). To create an all-to-all connection:

.. code-block:: python

    from bindsnet.network.topology import Connection

    connection = Connection(
        source=[source population],
        target=[target population],
        ...
    )


Like nodes, each connection object has many keyword arguments, but both :code:`source` and :code:`target` are required.
These must be objects that subclass `bindsnet.network.nodes.Nodes`. Other arguments include :code:`w` and :code:`b`
(weight and bias tensors for the connection), :code:`wmin` and :code:`wmax` (minimum and maximum allowable weight
values), :code:`update_rule` (:py:class:`bindsnet.learning.LearningRule`; used for updating connection weights based on
pre- and post-synaptic neuron activity and global neuromodulatory signals), and :code:`norm` (a floating point value
to normalize weights to sum to).

To add a connection to the network, use the :code:`add_connection` function, and pass the names given to source and
target populations as :code:`source` and :code:`target` arguments:

.. code-block:: python

    network.add_connection(
        connection=connection,
        source='A'
        target='B'
    )

Such layers are kept in the dictionary attribute :code:`network.layers`, and can be accessed by the user; e.g., by
:code:`network.connections['A', 'B']`.

Custom connection objects can be implemented by sub-classing :py:class:`bindsnet.network.topology.AbstractConnection`, an
abstract class with common logic for computing synapse outputs and updates. The functions :code:`compute` (for computing
input to downstream layer as a function of spikes and connection weights), :code:`update` (for updating connection
weights based on pre-, post-synaptic activity and possibly other signals; e.g., reward prediction error),
:code:`normalize`, and :code:`reset_`.

Specifying monitors
*******************



Simulation Notes
----------------

The simulation of all network components is *synchronous* (*clock-driven*); i.e., all components are updated at each
time step. Other frameworks use event-driven simulation, where spikes can occur at arbitrary times instead of at regular
multiples of :code:`dt`.

During a simulation step, input to each layer is computed as the sum of all outputs from layers connecting to it
(weighted by synapse weights) from the *previous* simulation time step. This model allows us to decouple network
components and perform their simulation separately at the temporal granularity of chosen :code:`dt`, interacting only
between simulation steps.

This is a strict departure from the computation of *deep neural networks* (DNNs), in which an ordering of layers is
supposed, and layers' activations are computed *in sequence* from the shallowest to the deepest layer in a single time
step, with the exclusion of recurrent layers, whose computations are still ordered in time.

Using Learning Rules
--------------------
