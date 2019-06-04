from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union

import torch


class Nodes(ABC, torch.nn.Module):
    # language=rst
    """
    Abstract base class for groups of neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        learning: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param learning: Whether to be in learning or testing.
        """
        super().__init__()

        assert (
            n is not None or shape is not None
        ), "Must provide either no. of neurons or shape of layer"

        if n is None:
            self.n = reduce(mul, shape)  # No. of neurons product of shape.
        else:
            self.n = n  # No. of neurons provided.

        if shape is None:
            self.shape = [self.n]  # Shape is equal to the size of the layer.
        else:
            self.shape = shape  # Shape is passed in as an argument.

        assert self.n == reduce(
            mul, self.shape
        ), "No. of neurons and shape do not match"

        self.traces = traces  # Whether to record synaptic traces.
        self.traces_additive = traces_additive  # Whether to record spike traces additively.
        self.register_buffer('s', torch.zeros(self.shape,
            dtype=torch.uint8))  # Spike occurrences.

        self.sum_input = sum_input  # Whether to sum all inputs.

        if self.traces:
            self.register_buffer('x', torch.zeros(self.shape))  # Firing traces.
            self.register_buffer('tc_trace', torch.tensor(
                tc_trace
            ))  # Time constant of spike trace decay.
            if self.traces_additive:
                self.register_buffer('trace_scale',
                    torch.tensor(trace_scale))  # Scaling factor for spike trace.
            self.register_buffer('trace_decay',
                    torch.empty_like(self.tc_trace))  # Set in _compute_decays.

        if self.sum_input:
            self.register_buffer('summed', torch.zeros(self.shape))  # Summed inputs.

        self.dt = None
        self.learning = learning

    @abstractmethod
    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Abstract base class method for a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            # Decay and set spike traces.
            self.x *= self.trace_decay
            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s != 0, 1)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += x.float()

    @abstractmethod
    def reset_(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        self.s.zero_()

        if self.traces:
            self.x.zero_()  # Firing traces.

        if self.sum_input:
            self.summed.zero_()  # Summed inputs.

    @abstractmethod
    def _compute_decays(self) -> None:
        # language=rst
        """
        Abstract base class method for setting decays.
        """
        if self.traces:
            self.trace_decay = torch.exp(
                -self.dt / self.tc_trace
            )  # Spike trace decay (per timestep).

    def train(self, mode: bool=True):
        """Sets the node in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)


class AbstractInput(ABC):
    # language=rst
    """
    Abstract base class for groups of input neurons.
    """


class Input(Nodes, AbstractInput):
    # language=rst
    """
    Layer of nodes with user-specified spiking behavior.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the spikes of the population equal to the inputs.

        :param x: Inputs to the layer.
        """
        # Set spike occurrences to input values.
        self.s = x.byte()

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()


class RealInput(Nodes, AbstractInput):
    # language=rst
    """
    Layer of nodes with user-specified real-valued outputs.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the outputs of the population equal to the inputs.

        :param x: Inputs to the layer.
        """

        # Set spike occurrences to input values.
        self.s = self.dt * x

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()


class McCullochPitts(Nodes):
    # language=rst
    """
    Layer of `McCulloch-Pitts neurons
    <http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a McCulloch-Pitts layer of neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.thresh = thresh  # Spike threshold voltage.
        self.register_buffer('v', torch.zeros(self.shape,
            dtype=torch.float)) # Neuron voltages.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = x  # Voltages are equal to the inputs.
        self.s = self.v >= self.thresh  # Check for spiking neurons.

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()


class IFNodes(Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('reset', torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer('refrac', torch.tensor(refrac))  # Post-spike refractory period.
        self.register_buffer('v', self.reset * torch.ones(self.shape))  # Neuron voltages.
        self.register_buffer('refrac_count', torch.zeros(self.shape))  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Integrate input voltages.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (
            self.refrac_count - self.dt
        )

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.reset)  # Neuron voltages.
        self.refrac_count.zero_() # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()


class LIFNodes(Nodes):
    # language=rst
    """
    Layer of `leaky integrate-and-fire (LIF) neurons
    <http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02311000000000000000>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('rest', torch.tensor(rest))  # Rest voltage.
        self.register_buffer('reset', torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer('refrac', torch.tensor(refrac))  # Post-spike refractory period.
        self.register_buffer('tc_decay', torch.tensor(tc_decay))  # Time constant of neuron voltage decay.
        self.register_buffer('decay', torch.zeros(self.shape))  # Set in _compute_decays.
        self.register_buffer('v', self.rest * torch.ones(self.shape))  # Neuron voltages.
        self.register_buffer('refrac_count', torch.zeros(self.shape))  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """

        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (
            self.refrac_count - self.dt
        )

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_() # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).


class CurrentLIFNodes(Nodes):
    # language=rst
    """
    Layer of `current-based leaky integrate-and-fire (LIF) neurons
    <http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02313000000000000000>`_.
    Total synaptic input current is modeled as a decaying memory of input spikes multiplied by synaptic strengths.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        tc_i_decay: Union[float, torch.Tensor] = 2.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of synaptic input current-based LIF neurons.
        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param tc_i_decay: Time constant of synaptic input current decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('rest', torch.tensor(rest))  # Rest voltage.
        self.register_buffer('reset', torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer('refrac', torch.tensor(refrac))  # Post-spike refractory period.
        self.register_buffer('tc_decay', torch.tensor(tc_decay))  # Time constant of neuron voltage decay.
        self.register_buffer('decay', torch.empty_like(self.tc_decay))  # Set in _compute_decays.
        self.register_buffer('tc_i_decay', torch.tensor(tc_i_decay))  # Time constant of synaptic input current decay.
        self.register_buffer('i_decay', torch.empty_like(self.tc_i_decay))  # Set in _compute_decays.

        self.register_buffer('v', self.rest * torch.ones(self.shape))  # Neuron voltages.
        self.register_buffer('i', torch.zeros(self.shape))  # Synaptic input currents.
        self.register_buffer('refrac_count', torch.zeros(self.shape))  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and current.
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.i *= self.i_decay

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (
            self.refrac_count - self.dt
        )

        # Integrate inputs.
        self.i += x
        self.v += (self.refrac_count == 0).float() * self.i

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.i.zero_()  # Synaptic input currents.
        self.refrac_count.zero_()  # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.i_decay = torch.exp(
            -self.dt / self.tc_i_decay
        )  # Synaptic input current decay (per timestep).


class AdaptiveLIFNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds. A neuron's voltage threshold is increased
    by some constant each time it spikes; otherwise, it is decaying back to its default value.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        thresh: Union[float, torch.Tensor] = -52.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons with adaptive firing thresholds.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('rest', torch.tensor(rest))  # Rest voltage.
        self.register_buffer('reset', torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer('refrac', torch.tensor(refrac))  # Post-spike refractory period.
        self.register_buffer('tc_decay', torch.tensor(tc_decay))  # Time constant of neuron voltage decay.
        self.register_buffer('decay', torch.empty_like(self.tc_decay))  # Set in _compute_decays.
        self.register_buffer('theta_plus', torch.tensor(theta_plus))  # Constant threshold increase on spike.
        self.register_buffer('tc_theta_decay', torch.tensor(tc_theta_decay))  # Time constant of adaptive threshold decay.
        self.register_buffer('theta_decay', torch.empty_like(self.tc_theta_decay))  # Set in _compute_decays.

        self.register_buffer('v', self.rest * torch.ones(self.shape))  # Neuron voltages.
        self.register_buffer('theta', torch.zeros(self.shape))  # Adaptive thresholds.
        self.register_buffer('refrac_count', torch.zeros(self.shape))  # Refractory period counters.
        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (
            self.refrac_count - self.dt
        )

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float()

        # voltage clipping to lowerbound
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).


class DiehlAndCookNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        one_spike: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('rest', torch.tensor(rest))  # Rest voltage.
        self.register_buffer('reset', torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer('refrac', torch.tensor(refrac))  # Post-spike refractory period.
        self.register_buffer('tc_decay', torch.tensor(tc_decay))  # Time constant of neuron voltage decay.
        self.register_buffer('decay', torch.empty_like(self.tc_decay))  # Set in _compute_decays.
        self.register_buffer('theta_plus', torch.tensor(theta_plus))  # Constant threshold increase on spike.
        self.register_buffer('tc_theta_decay', torch.tensor(tc_theta_decay))  # Time constant of adaptive threshold decay.
        self.register_buffer('theta_decay', torch.empty_like(self.tc_theta_decay))  # Set in _compute_decays.
        self.register_buffer('v', self.rest * torch.ones(self.shape))  # Neuron voltages.
        self.register_buffer('theta', torch.zeros(self.shape))  # Adaptive thresholds.
        self.register_buffer('refrac_count', torch.zeros(self.shape))  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.
        self.one_spike = one_spike  # One spike per timestep.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (
            self.refrac_count - self.dt
        )

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float()

        # Choose only a single neuron to spike.
        if self.one_spike:
            if self.s.any():
                ind = torch.multinomial(self.s.float().view(-1), 1)
                self.s.zero_()
                self.s.view(-1)[ind] = 1

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).


class IzhikevichNodes(Nodes):
    # language=rst
    """
    Layer of Izhikevich neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1,
        thresh: Union[float, torch.Tensor] = 45.0,
        rest: Union[float, torch.Tensor] = -65.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('rest', torch.tensor(rest))  # Rest voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.lbound = lbound

        self.register_buffer('r', None)
        self.register_buffer('a', None)
        self.register_buffer('b', None)
        self.register_buffer('c', None)
        self.register_buffer('d', None)
        self.register_buffer('S', None)
        self.register_buffer('excitatory', None)

        if excitatory > 1:
            excitatory = 1
        elif excitatory < 0:
            excitatory = 0

        if excitatory == 1:
            self.r = torch.rand(n)
            self.a = 0.02 * torch.ones(n)
            self.b = 0.2 * torch.ones(n)
            self.c = -65.0 + 15 * (self.r ** 2)
            self.d = 8 - 6 * (self.r ** 2)
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()

        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * self.r
            self.c = -65.0 * torch.ones(n)
            self.d = 2 * torch.ones(n)
            self.S = -torch.rand(n, n)

            self.excitatory = torch.zeros(n).byte()

        else:
            self.excitatory = torch.zeros(n).byte()

            ex = int(n * excitatory)
            inh = n - ex

            # init
            self.r = torch.zeros(n)
            self.a = torch.zeros(n)
            self.b = torch.zeros(n)
            self.c = torch.zeros(n)
            self.d = torch.zeros(n)
            self.S = torch.zeros(n, n)

            # excitatory
            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = 0.02 * torch.ones(ex)
            self.b[:ex] = 0.2 * torch.ones(ex)
            self.c[:ex] = -65.0 + 15 * self.r[:ex] ** 2
            self.d[:ex] = 8 - 6 * self.r[:ex] ** 2
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

            # inhibitory
            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = 0.02 + 0.08 * self.r[ex:]
            self.b[ex:] = 0.25 - 0.05 * self.r[ex:]
            self.c[ex:] = -65.0 * torch.ones(inh)
            self.d[ex:] = 2 * torch.ones(inh)
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0

        self.register_buffer('v', self.rest * torch.ones(n))  # Neuron voltages.
        self.register_buffer('u', self.b * self.v)  # Neuron recovery.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Voltage and recovery reset.
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        # Add inter-columnar input.
        if self.s.any():
            x += self.S[:, self.s].sum(dim=1)

        # Apply v and u updates.
        self.v += self.dt * 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.v += self.dt * 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.u = self.b * self.v  # Neuron recovery.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()

class SRM0Nodes(Nodes):
    # language=rst
    """
    Layer of simpliefied spike response model (SRM0) neurons with stochastic threshold (escape noise). Adapted from
    `(Vasilaki et al., 2009) <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 traces_additive: bool = False, tc_trace: Union[float, torch.Tensor] = 20.0,
                 trace_scale: Union[float, torch.Tensor] = 1.0, sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = -50.0, rest: Union[float, torch.Tensor] = -70.0,
                 reset: Union[float, torch.Tensor] = -70.0, refrac: Union[int, torch.Tensor] = 5,
                 tc_decay: Union[float, torch.Tensor] = 10.0, lbound: float = None,
                 eps_0: Union[float, torch.Tensor] = 1.0, rho_0: Union[float, torch.Tensor] = 1.0,
                 d_thresh: Union[float, torch.Tensor] = 5.0, **kwargs) -> None:
        # language=rst
        """
        Instantiates a layer of SRM0 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        :param eps_0: Scaling factor for pre-synaptic spike contributions.
        :param rho_0: Stochastic intensity at threshold.
        :param d_thresh: Width of the threshold region.
        """
        super().__init__(n, shape, traces, traces_additive, tc_trace, trace_scale, sum_input)

        self.register_buffer('rest', torch.tensor(rest))  # Rest voltage.
        self.register_buffer('reset', torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer('thresh', torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer('refrac', torch.tensor(refrac))  # Post-spike refractory period.
        self.register_buffer('tc_decay', torch.tensor(tc_decay))  # Time constant of neuron voltage decay.
        self.register_buffer('decay', torch.tensor(tc_decay))  # Set in _compute_decays.
        self.register_buffer('eps_0', torch.tensor(eps_0))  # Scaling factor for pre-synaptic spike contributions.
        self.register_buffer('rho_0', torch.tensor(rho_0))  # Stochastic intensity at threshold.
        self.register_buffer('d_thresh', torch.tensor(d_thresh))  # Width of the threshold region.

        self.register_buffer('v', self.rest * torch.ones(self.shape))  # Neuron voltages.
        self.register_buffer('refrac_count', torch.zeros(self.shape))  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * self.eps_0 * x

        # Compute (instantaneous) probabilities of spiking, clamp between 0 and 1 using exponentials.
        # Also known as 'escape noise', this simulates nearby neurons.
        self.rho = self.rho_0 * torch.exp((self.v - self.thresh) / self.d_thresh)
        self.s_prob = 1.0 - torch.exp(-self.rho * self.dt)

        # Decrement refractory counters.
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)

        # Check for spiking neurons (spike when probability > some random number).
        self.s = torch.rand_like(self.s_prob) < self.s_prob

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()
        self.decay = torch.exp(-self.dt / self.tc_decay)  # Neuron voltage decay (per timestep).
