from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn  # <<< THIS LINE IS CRUCIAL

from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection

from bindsnet.learning import NoOp # Using NoOp as FF updates are external

class TwoLayerNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a
    fully-connected ``Connection``.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        dt: float = 1.0,
        wmin: float = 0.0,
        wmax: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        norm: float = 78.4,
    ) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization
            constant.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt

        self.add_layer(Input(n=self.n_inpt, traces=True, tc_trace=20.0), name="X")
        self.add_layer(
            LIFNodes(
                n=self.n_neurons,
                traces=True,
                rest=-65.0,
                reset=-65.0,
                thresh=-52.0,
                refrac=5,
                tc_decay=100.0,
                tc_trace=20.0,
            ),
            name="Y",
        )

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(
            Connection(
                source=self.layers["X"],
                target=self.layers["Y"],
                w=w,
                update_rule=PostPre,
                nu=nu,
                reduction=reduction,
                wmin=wmin,
                wmax=wmax,
                norm=norm,
            ),
            source="X",
            target="Y",
        )


class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        inh_thresh: float = -40.0,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=inh_thresh,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")


class DiehlAndCook2015v2(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")


class IncreasingInhibitionNetwork(Network):
    # language=rst
    """
    Implements the inhibitory layer structure of the spiking neural network architecture
    from `(Hazan et al. 2018) <https://arxiv.org/abs/1807.09374>`_
    """

    def __init__(
        self,
        n_input: int,
        n_neurons: int = 100,
        start_inhib: float = 1.0,
        max_inhib: float = 100.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``IncreasingInhibitionNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.n_sqrt = int(np.sqrt(n_neurons))
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.dt = dt
        self.inpt_shape = inpt_shape

        input_layer = Input(
            n=self.n_input, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        input_output_conn = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_output_conn, source="X", target="Y")

        # add internal inhibitory connections
        w = torch.ones(self.n_neurons, self.n_neurons) - torch.diag(
            torch.ones(self.n_neurons)
        )
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j:
                    x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                    x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                    w[i, j] = np.sqrt(euclidean([x1, y1], [x2, y2]))
        w = w / w.max()
        w = (w * self.max_inhib) + self.start_inhib
        recurrent_output_conn = Connection(
            source=self.layers["Y"], target=self.layers["Y"], w=w
        )
        self.add_connection(recurrent_output_conn, source="Y", target="Y")


class LocallyConnectedNetwork(Network):




    # language=rst
    """
    Defines a two-layer network in which the input layer is "locally connected" to the
    output layer, and the output layer is recurrently inhibited connected such that
    neurons with the same input receptive field inhibit each other.
    """

    def __init__(
        self,
        n_inpt: int,
        input_shape: List[int],
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        inh: float = 25.0,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        wmin: float = 0.0,
        wmax: float = 1.0,
        norm: Optional[float] = 0.2,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``LocallyConnectedNetwork``. Uses ``DiehlAndCookNodes`` to
        avoid multiple spikes per timestep in the output layer population.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param input_shape: Two-dimensional shape of input population.
        :param kernel_size: Size of input windows. Integer or two-tuple of integers.
        :param stride: Length of horizontal, vertical stride across input space. Integer
            or two-tuple of integers.
        :param n_filters: Number of locally connected filters per input region. Integer
            or two-tuple of integers.
        :param inh: Strength of synapse weights from output layer back onto itself.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``DiehlAndCookNodes``
            synapses.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param norm: ``Input`` to ``DiehlAndCookNodes`` layer connection weights
            normalization constant.
        """
        super().__init__(dt=dt)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.n_inpt = n_inpt
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.inh = inh
        self.dt = dt
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        if kernel_size == input_shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
                int((input_shape[1] - kernel_size[1]) / stride[1]) + 1,
            )

        input_layer = Input(n=self.n_inpt, traces=True, tc_trace=20.0)

        output_layer = DiehlAndCookNodes(
            n=self.n_filters * conv_size[0] * conv_size[1],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        input_output_conn = LocalConnection(
            input_layer,
            output_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size[0]):
                        for j in range(conv_size[1]):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(
            n_filters * conv_size[0] * conv_size[1],
            n_filters * conv_size[0] * conv_size[1],
        )
        recurrent_conn = Connection(output_layer, output_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(output_layer, name="Y")
        self.add_connection(input_output_conn, source="X", target="Y")
        self.add_connection(recurrent_conn, source="Y", target="Y")


import snntorch as snn

class FFSNN(nn.Module):
    # language=rst
    """
    A simple feedforward Spiking Neural Network (SNN) using snntorch,
    designed for use with the ForwardForwardPipeline.
    It consists of a sequence of Linear layers followed by Leaky Integrate-and-Fire
    (LIF) spiking neuron layers.
    """

    def __init__(
        self,
        input_size: int, # This should be 794 (image_features + num_classes)
        hidden_sizes: List[int], # e.g., [500, 500]
        output_size: Optional[int] = None, # If the last FF layer is also the output layer for classification
        beta: Union[float, torch.Tensor] = 0.9,  # Decay rate for snn.Leaky neurons
        threshold: float = 1.0,  # Firing threshold for snn.Leaky neurons
        reset_mechanism: str = "subtract",  # "subtract", "zero", or "none"
        # Add other snn.Leaky parameters if needed, e.g., spike_grad
    ) -> None:
        # language=rst
        """
        Constructor for FFSNN.

        :param input_size: Number of input features (after encoding and label embedding).
        :param hidden_sizes: A list of integers, where each integer is the number of
                             neurons in a hidden layer.
        :param output_size: Optional. Number of neurons in the final layer if it's
                            distinct or specifically for output. If None, the last
                            size in hidden_sizes is considered the final FF layer.
        :param beta: Membrane potential decay rate for Leaky neurons.
        :param threshold: Firing threshold for Leaky neurons.
        :param reset_mechanism: Reset mechanism for Leaky neurons after a spike.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.beta = beta
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism

        self.fc_layers = nn.ModuleList()
        self.snn_layers = nn.ModuleList()
        self._ff_layer_pairs_info = []

        current_dim = self.input_size # Starts at 794
        for i, hidden_dim in enumerate(self.hidden_sizes):
            linear_layer = nn.Linear(current_dim, hidden_dim) # Layer 1: 794 -> 500
                                                              # Layer 2: 500 -> 500
            self.fc_layers.append(linear_layer)

            snn_layer = snn.Leaky(
                beta=self.beta, 
                threshold=self.threshold, 
                reset_mechanism=self.reset_mechanism,
                # output_shape=[hidden_dim] # Optional: snntorch can infer this
            )
            self.snn_layers.append(snn_layer)
            self._ff_layer_pairs_info.append((linear_layer, snn_layer))
            current_dim = hidden_dim # Update current_dim for the *next* layer's input
        
        # If there's an output_size for a final classifier (not typical for pure FF layers)
        if self.output_size is not None:
            self.fc_out = nn.Linear(current_dim, self.output_size)
            # Potentially another SNN layer if output is spiking
            # self.snn_out = snn.Leaky(...)
            # self._ff_layer_pairs_info.append((self.fc_out, self.snn_out)) # If FF applies here too

    def forward(self, x_batch_time: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # language=rst
        """
        Defines the forward pass of the SNN over time.
        This method might be used if the network is called directly with time-series data.
        However, the ForwardForwardPipeline._run_snn_batch currently iterates
        through self.network_sequence modules per time step.

        :param x_batch_time: Input tensor with shape [batch_size, time_steps, num_features].
        :return: Final layer output spikes and a list of hidden states (membrane potentials)
                 from all spiking layers.
        """
        # Initialize hidden states for all spiking layers in the sequence
        # This assumes they are snn.Leaky and support init_leaky() or similar
        # Or, more generally, that they initialize if mem is None on first call.
        
        spiking_layer_modules = [info['spiking'] for info in self._ff_layer_pairs_info]
        # mem_states = [layer.init_leaky() for layer in spiking_layer_modules] # This creates new states
        # For snntorch, typically pass None for initial state, layer handles it.
        
        # The pipeline's _run_snn_batch actually handles the time loop and state passing.
        # This forward method is more for standalone use or if the pipeline changes.
        # If you want this model to be directly callable with (B, T, F) input and manage its own time loop:

        batch_size = x_batch_time.shape[0]
        
        # Initialize states for each spiking layer for this batch
        # This is tricky because snn.Leaky.init_hidden() doesn't take batch_size.
        # State initialization is usually handled by passing None to the layer's forward method
        # for the first time step, and it initializes based on the input batch size.
        
        # Placeholder: The pipeline's _run_snn_batch is the primary runner.
        # This forward method would need a more elaborate state management if used directly.
        # For now, let's make it compatible with how _run_snn_batch works if it were to call this.
        
        # If this 'forward' is to be used, it should mirror _run_snn_batch's logic:
        # Initialize all spiking layer states to None
        spiking_layer_states = {module: None for module in self.network_sequence if isinstance(module, snn.SpikingNeuron)}
        
        # Record outputs if needed (e.g., for a final classification layer not part of FF)
        # final_spk_rec = [] # If you want to record output spikes over time

        for t in range(x_batch_time.shape[1]): # Iterate over time
            x_t = x_batch_time[:, t, :]
            layer_input = x_t
            
            current_module_idx = 0
            for module in self.network_sequence:
                if isinstance(module, snn.SpikingNeuron):
                    spk_out, new_mem = module(layer_input, spiking_layer_states.get(module))
                    spiking_layer_states[module] = new_mem
                    layer_input = spk_out
                else: # nn.Linear
                    layer_input = module(layer_input)
            # After passing through all layers for time step t, layer_input is the output of the last layer
            # final_spk_rec.append(layer_input) 
        
        # return torch.stack(final_spk_rec, dim=1), [state for state in spiking_layer_states.values()]
        return layer_input, [spiking_layer_states[info['spiking']] for info in self._ff_layer_pairs_info]


    def get_ff_layer_pairs(self) -> List[Tuple[nn.Linear, snn.SpikingNeuron]]:
        # language=rst
        """
        Returns the list of (Linear, SpikingNeuron) pairs for Forward-Forward training.
        """
        # If _ff_layer_pairs_info contains tuples, return them directly
        return self._ff_layer_pairs_info
        
        # Alternative: If you want to be explicit about the structure
        # return [(pair[0], pair[1]) for pair in self._ff_layer_pairs_info]


class FFSNN_BindsNET(Network):
    # language=rst
    """
    A feedforward Spiking Neural Network (SNN) using BindsNET components,
    structured for use with an adapted ForwardForwardPipeline.
    It consists of an Input layer followed by a sequence of Connection objects
    (representing linear weights) and LIF spiking neuron layers.
    """

    def __init__(
        self,
        n_inpt: int,
        hidden_sizes: List[int],
        output_size: Optional[int] = None,
        dt: float = 1.0,
        thresh: Union[float, List[float]] = -52.0,
        rest: Union[float, List[float]] = -65.0,
        reset: Union[float, List[float]] = -65.0,
        tc_decay: Union[float, List[float]] = 100.0,
        refrac: Union[int, List[int]] = 5,
        initial_w_mag: float = 0.3,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for FFSNN_BindsNET.

        :param n_inpt: Number of input neurons.
        :param hidden_sizes: A list of integers for neurons in each hidden LIF layer.
        :param output_size: Optional. Number of neurons in the final LIF layer.
        :param dt: Simulation time step.
        :param thresh: Firing threshold for LIF neurons. Can be a list for per-layer values.
        :param rest: Resting potential for LIF neurons. Can be a list.
        :param reset: Reset potential for LIF neurons. Can be a list.
        :param tc_decay: Membrane potential decay time constant. Can be a list.
        :param refrac: Refractory period for LIF neurons. Can be a list.
        :param initial_w_mag: Magnitude for random weight initialization.
        """
        super().__init__(dt=dt, **kwargs)

        self.n_inpt = n_inpt
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self._ff_layer_pairs_info = []  # Stores (Connection, LIFNodes_layer)

        all_node_counts = [n_inpt] + hidden_sizes
        if output_size is not None and (not hidden_sizes or output_size != hidden_sizes[-1]):
            all_node_counts.append(output_size)
        elif output_size is not None and hidden_sizes and output_size == hidden_sizes[-1] and len(hidden_sizes) > 1:
            pass # Already included
        elif output_size is not None and not hidden_sizes:
             all_node_counts = [n_inpt, output_size]

        num_spiking_layers = len(all_node_counts) - 1

        def _get_param(param_val, idx):
            return param_val[idx] if isinstance(param_val, list) else param_val

        # Input Layer
        self.input_layer_name = "InputLayer"
        input_node = Input(n=self.n_inpt, traces=True, tc_trace=20.0)
        self.add_layer(input_node, name=self.input_layer_name)
        
        prev_layer_name = self.input_layer_name
        prev_n_neurons = self.n_inpt

        for i in range(num_spiking_layers):
            current_n_neurons = all_node_counts[i+1]
            
            # Spiking Layer (LIFNodes)
            lif_layer_name = f"LIFLayer_{i}"
            lif_node = LIFNodes(
                n=current_n_neurons,
                traces=True, # Essential for monitoring spikes for goodness
                thresh=_get_param(thresh, i),
                rest=_get_param(rest, i),
                reset=_get_param(reset, i),
                tc_decay=_get_param(tc_decay, i),
                refrac=_get_param(refrac, i),
                tc_trace=20.0
            )
            self.add_layer(lif_node, name=lif_layer_name)

            # Connection to this Spiking Layer
            conn_name = f"Conn_{prev_layer_name}_to_{lif_layer_name}"
            # Weights will be updated by the external FF pipeline
            w_initial = initial_w_mag * torch.rand(prev_n_neurons, current_n_neurons)
            
            connection = Connection(
                source=self.layers[prev_layer_name],
                target=self.layers[lif_layer_name],
                w=w_initial,
                update_rule=NoOp, # FF updates are external
                # nu=(0.0, 0.0) # No learning for NoOp
            )
            # For PyTorch optimizers to update connection.w, it ideally should be an nn.Parameter.
            # The pipeline would need to handle this, e.g., by wrapping connection.w or
            # ensuring optimizers can work with tensors if grads are manually assigned.
            # Example: connection.w = nn.Parameter(connection.w.clone()) # Do this in pipeline setup
            self.add_connection(connection, source=prev_layer_name, target=lif_layer_name)
            
            self._ff_layer_pairs_info.append({
                'connection': connection,    # Weights of this connection are trained
                'spiking_layer': lif_node, # Spikes from this layer determine goodness
                'name': f'ff_bindsnet_pair_{i}'
            })
            
            prev_layer_name = lif_layer_name
            prev_n_neurons = current_n_neurons

    def get_ff_layer_pairs(self) -> List[Tuple[Connection, LIFNodes]]:
        # language=rst
        """
        Returns the list of (Connection to Spiking Layer, Spiking Neuron Layer) pairs
        intended for Forward-Forward training.
        """
        return [(pair_info['connection'], pair_info['spiking_layer']) for pair_info in self._ff_layer_pairs_info]

    # The forward pass for BindsNET is network.run(), called by the pipeline.
    # No explicit forward() method needed here for that.
