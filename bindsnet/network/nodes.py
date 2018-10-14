import torch
import numpy as np

from operator import mul
from functools import reduce
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union


class Nodes(ABC):
    # language=rst
    """
    Abstract base class for groups of neurons.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__()

        assert n is not None or shape is not None, 'Must provide either no. of neurons or shape of layer'

        if n is None:
            self.n = reduce(mul, shape)                 # No. of neurons product of shape.
        else:
            self.n = n                                  # No. of neurons provided.

        if shape is None:
            self.shape = [self.n]                       # Shape is equal to the size of the layer.
        else:
            self.shape = shape                          # Shape is passed in as an argument.

        assert self.n == reduce(mul, self.shape), 'No. of neurons and shape do not match'

        self.traces = traces                            # Whether to record synaptic traces.
        self.s = torch.zeros(self.shape).byte()         # Spike occurrences.
        self.sum_input = sum_input                      # Whether to sum all inputs.

        if self.traces:
            self.x = torch.zeros(self.shape)            # Firing traces.

            if isinstance(trace_tc, float):
                self.trace_tc = torch.tensor(trace_tc)  # Rate of decay of spike trace time constant.
            else:
                self.trace_tc = trace_tc

        if self.sum_input:
            self.summed = torch.zeros(self.shape)       # Summed inputs.

    @abstractmethod
    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Abstract base class method for a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        if self.traces:
            # Decay and set spike traces.
            self.x -= dt * self.trace_tc * self.x
            self.x.masked_fill_(self.s, 1)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += inpts.float()

    @abstractmethod
    def reset_(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        if not isinstance(self, RealInput):
            self.s = torch.zeros(self.shape).byte()  # Spike occurrences.
        else:
            self.s = torch.zeros(self.shape)  # Real-valued "spikes".

        if self.traces:
            self.x = torch.zeros(self.shape)  # Firing traces.

        if self.sum_input:
            self.summed = torch.zeros(self.shape)  # Summed inputs.


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

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None,
                 traces: bool = False, trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        On each simulation step, set the spikes of the population equal to the inputs.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Set spike occurrences to input values.
        self.s = inpts.byte()

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()


class RealInput(Nodes, AbstractInput):
    """
    Layer of nodes with user-specified real-valued outputs.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        self.s = self.s.float()

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        On each simulation step, set the outputs of the population equal to the inputs.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Set spike occurrences to input values.
        self.s = inpts

        if self.traces:
            # Decay and set spike traces.
            self.x -= dt * self.trace_tc * self.x
            self.x.masked_fill_(self.s != 0, 1)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += inpts.float()

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()


class McCullochPitts(Nodes):
    # language=rst
    """
    Layer of `McCulloch-Pitts neurons
    <http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html>`_.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = 1.0) -> None:
        # language=rst
        """
        Instantiates a McCulloch-Pitts layer of neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        self.thresh = thresh              # Spike threshold voltage.
        self.v = torch.zeros(self.shape)  # Neuron voltages.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        self.v = inpts                  # Voltages are equal to the inputs.
        self.s = self.v >= self.thresh  # Check for spiking neurons.

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()


class IFNodes(Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = -52.0, reset: Union[float, torch.Tensor] = -65.0,
                 refrac: Union[int, torch.Tensor] = 5) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        # Post-spike reset voltage.
        if isinstance(reset, float):
            self.reset = torch.tensor(reset)
        else:
            self.reset = reset

        # Spike threshold voltage.
        if isinstance(thresh, float):
            self.thresh = torch.tensor(thresh)
        else:
            self.thresh = thresh

        # Post-spike refractory period.
        if isinstance(refrac, float):
            self.refrac = torch.tensor(refrac)
        else:
            self.refrac = refrac

        self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)   # Refractory period counters.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Integrate input voltages.
        self.v += (self.refrac_count == 0).float() * inpts

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)   # Refractory period counters.


class LIFNodes(Nodes):
    # language=rst
    """
    Layer of `leaky integrate-and-fire (LIF) neurons
    <http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02311000000000000000>`_.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = -52.0, rest: Union[float, torch.Tensor] = -65.0,
                 reset: Union[float, torch.Tensor] = -65.0, refrac: Union[int, torch.Tensor] = 5,
                 decay: Union[float, torch.Tensor] = 1e-2) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param decay: Time constant of neuron voltage decay.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        # Rest voltage.
        if isinstance(rest, float):
            self.rest = torch.tensor(rest)
        else:
            self.rest = rest

        # Post-spike reset voltage.
        if isinstance(reset, float):
            self.reset = torch.tensor(reset)
        else:
            self.reset = reset

        # Spike threshold voltage.
        if isinstance(thresh, float):
            self.thresh = torch.tensor(thresh)
        else:
            self.thresh = thresh

        # Post-spike refractory period.
        if isinstance(refrac, float):
            self.refrac = torch.tensor(refrac)
        else:
            self.refrac = refrac

        # Rate of decay of neuron voltage.
        if isinstance(decay, float):
            self.decay = torch.tensor(decay)
        else:
            self.decay = decay

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Decay voltages.
        self.v -= dt * self.decay * (self.v - self.rest)

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * inpts

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class AdaptiveLIFNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds. A neuron's voltage threshold is increased
    by some constant each time it spikes; otherwise, it is decaying back to its default value.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False,
                 rest: Union[float, torch.Tensor] = -65.0, reset: Union[float, torch.Tensor] = -65.0,
                 thresh: Union[float, torch.Tensor] = -52.0, refrac: Union[int, torch.Tensor] = 5,
                 decay: Union[float, torch.Tensor] = 1e-2, theta_plus: Union[float, torch.Tensor] = 0.05,
                 theta_decay: Union[float, torch.Tensor] = 1e-7) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons with adaptive firing thresholds.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param theta_decay: Time constant of adaptive threshold decay.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        # Rest voltage.
        if isinstance(rest, float):
            self.rest = torch.tensor(rest)
        else:
            self.rest = rest

        # Post-spike reset voltage.
        if isinstance(reset, float):
            self.reset = torch.tensor(reset)
        else:
            self.reset = reset

        # Spike threshold voltage.
        if isinstance(thresh, float):
            self.thresh = torch.tensor(thresh)
        else:
            self.thresh = thresh

        # Post-spike refractory period.
        if isinstance(refrac, float):
            self.refrac = torch.tensor(refrac)
        else:
            self.refrac = refrac

        # Rate of decay of neuron voltage.
        if isinstance(decay, float):
            self.decay = torch.tensor(decay)
        else:
            self.decay = decay

        # Constant threshold increase on spike.
        if isinstance(theta_plus, float):
            self.theta_plus = torch.tensor(theta_plus)
        else:
            self.theta_plus = theta_plus

        # Rate of decay of adaptive thresholds.
        if isinstance(theta_decay, float):
            self.theta_decay = torch.tensor(theta_decay)
        else:
            self.theta_decay = theta_decay

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.theta = torch.zeros(self.shape)         # Adaptive thresholds.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Decay voltages and adaptive thresholds.
        self.v -= dt * self.decay * (self.v - self.rest)
        self.theta -= dt * self.theta_decay * self.theta

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * inpts

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = (self.v >= self.thresh + self.theta)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.theta += self.theta_plus * self.s.float()

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class DiehlAndCookNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = -52.0, rest: Union[float, torch.Tensor] = -65.0,
                 reset: Union[float, torch.Tensor] = -65.0, refrac: Union[int, torch.Tensor] = 5,
                 decay: Union[float, torch.Tensor] = 1e-2, theta_plus: Union[float, torch.Tensor] = 0.05,
                 theta_decay: Union[float, torch.Tensor] = 1e-7) -> None:
        # language=rst
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param theta_decay: Time constant of adaptive threshold decay.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        # Rest voltage.
        if isinstance(rest, float):
            self.rest = torch.tensor(rest)
        else:
            self.rest = rest

        # Post-spike reset voltage.
        if isinstance(reset, float):
            self.reset = torch.tensor(reset)
        else:
            self.reset = reset

        # Spike threshold voltage.
        if isinstance(thresh, float):
            self.thresh = torch.tensor(thresh)
        else:
            self.thresh = thresh

        # Post-spike refractory period.
        if isinstance(refrac, float):
            self.refrac = torch.tensor(refrac)
        else:
            self.refrac = refrac

        # Rate of decay of neuron voltage.
        if isinstance(decay, float):
            self.decay = torch.tensor(decay)
        else:
            self.decay = decay

        # Constant threshold increase on spike.
        if isinstance(theta_plus, float):
            self.theta_plus = torch.tensor(theta_plus)
        else:
            self.theta_plus = theta_plus

        # Rate of decay of adaptive thresholds.
        if isinstance(theta_decay, float):
            self.theta_decay = torch.tensor(theta_decay)
        else:
            self.theta_decay = theta_decay

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.theta = torch.zeros(self.shape)         # Adaptive thresholds.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Decay voltages and adaptive thresholds.
        self.v -= dt * self.decay * (self.v - self.rest)
        self.theta -= dt * self.theta_decay * self.theta

        # Integrate inputs.
        self.v += (self.refrac_count == 0).float() * inpts

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = (self.v >= self.thresh + self.theta)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.theta += self.theta_plus * self.s.float()

        # Choose only a single neuron to spike.
        if self.s.any():
            s = torch.zeros(self.n).byte()
            s[torch.multinomial(self.s.float().view(-1), 1)] = 1
            self.s = s.view(self.shape)

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class IzhikevichNodes(Nodes):
    # language=rst
    """
    Layer of Izhikevich neurons.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Iterable[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False, excitatory: float = 1,
                 thresh: Union[float, torch.Tensor] = 45.0, rest: Union[float, torch.Tensor] = -65.0) -> None:
        # language=rst
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        self.rest = rest       # Rest voltage.
        self.thresh = thresh   # Spike threshold voltage.

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
            self.excitatory = torch.ones(n).byte()
        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * self.r
            self.c = -65.0 * torch.ones(n)
            self.d = 2 * torch.ones(n)
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

            # excitatory
            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = 0.02 * torch.ones(ex)
            self.b[:ex] = 0.2 * torch.ones(ex)
            self.c[:ex] = -65.0 + 15 * (self.r[:ex] ** 2)
            self.d[:ex] = 8 - 6 * (self.r[:ex] ** 2)
            self.excitatory[:ex] = 1

            # inhibitory
            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = 0.02 + 0.08 * self.r[ex:]
            self.b[ex:] = 0.25 - 0.05 * self.r[ex:]
            self.c[ex:] = -65.0 * torch.ones(inh)
            self.d[ex:] = 2 * torch.ones(inh)
            self.excitatory[ex:] = 0

        self.v = self.rest * torch.ones(n)  # Neuron voltages.
        self.u = self.b * self.v            # Neuron recovery.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Apply v and u updates.
        self.v += dt * 0.5 * (0.04 * (self.v ** 2) + 5 * self.v + 140 - self.u + inpts)
        self.v += dt * 0.5 * (0.04 * (self.v ** 2) + 5 * self.v + 140 - self.u + inpts)
        self.u += self.a * (self.b * self.v - self.u)

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.u = self.b * self.v                     # Neuron recovery.


