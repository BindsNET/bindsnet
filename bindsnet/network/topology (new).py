from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Sequence

import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from nodes import Nodes, CSRMNodes
import warnings

class AbstractConnection(ABC, Module):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        :param dict features: Features to modify how connection behaves.
        """
        super().__init__()


        ### General Assertions ###
        assert isinstance(source, Nodes), "Source is not a Nodes object"
        assert isinstance(target, Nodes), "Target is not a Nodes object"
        assert wmin < wmax, "wmin must be smaller than wmax"


        ### Args/Kwargs ###
        self.source = source
        self.target = target
        # self.nu = nu
        self.weight_decay = weight_decay
        self.reduction = reduction

        self.update_rule = kwargs.get("update_rule", NoOp)
        self.wmin = kwargs.get("wmin", -np.inf)
        self.wmax = kwargs.get("wmax", np.inf)
        self.norm = kwargs.get("norm", None)
        self.decay = kwargs.get("decay", None)
        # self.features = kwargs.get("features", None)


        # ### Update Rule ###
        # from ..learning import NoOp
        # if self.update_rule is None:
        #     self.update_rule = NoOp
        #
        # self.update_rule = self.update_rule(
        #     connection=self,
        #     nu=nu,
        #     reduction=reduction,
        #     weight_decay=weight_decay,
        #     **kwargs
        # )

        ### Feature Pipeline ###
        order = ['delay', 'weights', 'probability', 'mask']
        funcs = [self.delay, self.weight, self.probability, self.mask]
        self.pipeline = []

        # Initialize present features
        if 'delay' in features:
            args = args['delay']

            assert args['range'][0] < args['range'][1], "Invalid delay range: lower bound larger than upper bound"
            assert args['range'][1] > 0, "Maximum delay must be greater than 0."

            # Indexing for delays (Keep track of how many outputs are needed)
            self.delays_idx = Parameter(
                torch.arange(0, source.n * target.n, dtype=torch.long), requires_grad=False
            )

            # Variable to record delays for each output signal
            self.delay_buffer = Parameter(
                torch.zeros(source.n * target.n, args['range'][1], dtype=torch.float),
                requires_grad=False,
            )

            # Initialize time index
            self.time_idx = 0

        # Get smallest weight min and largest weight max (depends if tensor or float)
        min = self.wmin if not isinstance(self.wmin, torch.Tensor) else torch.min(self.wmin)
        max = self.wmax if not isinstance(self.wmax, torch.Tensor) else torch.max(self.wmax)
        if 'weights' in features:

            # Clamp custom weights
            if min != -np.inf or max != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        else:

            # Randomly initialize weights if none provided
            if min == -np.inf or max == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)

        if 'probability' in features:
            self.probabilities = features['probability']['probabilities']

        if 'mask' in features:
            self.mask = features['mask']['m']



    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        # language=rst
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """
        pass


    @abstractmethod
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.

        Keyword arguments:

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get("learning", True)

        if learning:
            self.update_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        pass

    #########################
    ### Pipeline Features ###
    #########################

    @abstractmethod
    def weight(self, s, params) -> torch.Tensor:
        pass

    @abstractmethod
    def probability(self, s, params) -> torch.Tensor:
        pass

    @abstractmethod
    def delay(self, s, params) -> torch.Tensor:
        pass

    @abstractmethod
    def mask(self, s, params) -> torch.Tensor:
        pass


class Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """

        ### General connection setup ###
        # Decay weights
        if self.weight_decay is not None:
            if self.weight_linear_decay:
                self.w.data = self.w.data - self.weight_decay
            else:
                self.w.data = self.w.data - (self.w.data * self.weight_decay)

        # Clip min and max values
        self.w.data = torch.clamp(self.w.data, min=self.wmin, max=self.wmax)

        # Prepare broadcast from incoming spikes to all output neurons
        # Note: |conn_spikes| = [source.n * target.n]
        conn_spikes = s.view(self.source.n, 1).repeat(1, self.target.n).flatten()

        # Run through pipeline



    def weights(self, conn_spikes, params) -> torch.Tensor:
        return s @ self.w

    def probability(self, conn_spikes, params) -> torch.Tensor:
        travel_gate = torch.bernoulli(self.probabilities)
        return s & travel_gate

    def delay(self, conn_spikes, params) -> torch.Tensor:

        # convert weights to delays, in the given delay range
        # delays = self.max_delay - (self.w.flatten() * self.max_delay).long()
        delays = self.max_delay - (w_norm.flatten() * self.max_delay).long()

        # Drop late spikes and surpress new spikes in favor of old ones
        if self.refrac_count is not None:
            if conn_spikes.device != self.refrac_count.device:
                self.refrac_count = self.refrac_count.to(conn_spikes.device)
            if self.drop_late_spikes:
                conn_spikes[delays == self.max_delay] = 0
            conn_spikes &= self.refrac_count <= 0
            self.refrac_count -= 1
            self.refrac_count[conn_spikes.bool()] = delays[conn_spikes.bool()]

        # add circular time index to delays
        delays = (delays + self.time_idx) % self.max_delay

        return s


    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """"""

        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()



if __name__ == '__main__':

    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.learning import PostPre, MSTDP

    input_l = Input(n=784)
    hidden_l = LIFNodes(n=2500)
    m = torch.ones(784,2500)

    test_conn = Connection(
    source=input_l,
    target=hidden_l,
    features={
        'mask' : m,
        'probability': {
            'probabilities': 0.5 * torch.rand(784, 2500),
            'update_rule': PostPre,
            'nu': [1e-2, 0],
            'norm': 0.25,
            'range': [0.0, 0.5],
        },
        'weights': {
            'w': 0.5 + 0.5*torch.rand(784, 2500),
        },
        'delays': {
            'delays': torch.rand(784, 2500),
            'update_rule': MSTDP,  # reward-modulated STDP
            'nu': [1e-2, 1e-3],
            'norm': 0.8,
            'range': [0.0, 1.0]
        }
    })



# if __name__ == '__main__':
#
#     from nodes import LIFNodes
#     from bindsnet.network.nodes import Input
#     from bindsnet.network import Network
#     from bindsnet.network.topology import Connection
#
#     model = Network()
#
#     input_l = Input(n=784, spike_value=1.2)
#
#     # generate a 20% inh neuron map, with both exc and inh spike values
#     v = torch.rand(2500)
#     v[v < 0.2] = -5.0
#     v[v >= 0.2] = 2.5
#     hidden_l = LIFNodes(n=2500, spike_values=v)
#
#     output_l = LIFNodes(n=10, spike_values=-1000.0)
#
#     model.add_layer(input_l, name='X')
#     model.add_layer(hidden_l, name='H')
#     model.add_layer(output_l, name='Y')
#
#     # input to hidden connection definition :
#     # generating a statistical local con mask
#     m = torch.zeros(784, 2500)
#     for xi in range(28):
#         for yi in range(28):
#             for xo in range(50):
#                 for yo in range(50):
#                     dx = xi - xo
#                     dy = yi - yo
#                     m[xi + yi * 28, xo + 50 * yo] = dx * dx + dy * dy
#
#     m = torch.sqrt(m)
#     m /= m.max()
#     m *= torch.rand(784, 2500)
#     m = m > 0.5
#
#     in_hid_con = Connection(
#         source=input_l,
#         target=hidden_l,
#         features={
#             'mask': m,
#             'probability': {
#                 'probabilities': 0.5 * torch.rand(784, 2500),
#                 'update_rule': PostPre,
#                 'nu': [1e-2, 0],
#                 'norm': 0.25,
#                 'range': [0.0, 0.5],
#             },
#             'weights': {
#                 'w': 0.5 + 0.5 * torch.rand(784, 2500),
#             },
#             'delays': {
#                 'delays': torch.rand(784, 2500),
#                 'update_rule': MSTDP,  # reward-modulated STDP
#                 'nu': [1e-2, 1e-3],
#                 'norm': 0.8,
#                 'range': [0.0, 1.0]
#             }
#         }
#     )
#
#     # hidden to output connection definition :
#     hid_out_con = Connection(
#         source=hidden_l,
#         target=output_l,
#         features={
#             'weights': {
#                 'w': torch.rand(784, 2500),
#             }
#         }
#     )
#
#     # recurrent WTA connection on output definition :
#     recurrent_con = Connection(
#         source=output_l,
#         target=output_l,
#         features={
#             'weights': {
#                 'w': torch.ones(10, 10),
#             }
#         }
#     )
#
#     model.add_connection(in_hid_con, source='X', target='H')
#     model.add_connection(hid_out_con, source='H', target='Y')
#     model.add_connection(recurrent_con, source='Y', target='Y')