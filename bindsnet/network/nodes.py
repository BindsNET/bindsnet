import torch

from operator  import mul
from functools import reduce
from abc       import ABC, abstractmethod


class Nodes(ABC):
    '''
    Abstract base class for groups of neurons.
    '''
    def __init__(self, n=None, shape=None, traces=False, trace_tc=5e-2):
        super().__init__()
        
        assert not n is None or not shape is None, \
            'Must provide either no. of neurons or shape of nodes'
        
        if n is None:
            self.n = reduce(mul, shape)          # No. of neurons product of shape.
        else:
            self.n = n                           # No. of neurons provided.
        
        if shape is None:
            self.shape = [self.n]                # Shape is equal to the size of the layer.
        else:
            self.shape = shape                   # Shape is passed in as an argument.
        
        assert self.n == reduce(mul, self.shape), \
            'No. of neurons and shape do not match'
            
        self.traces = traces                     # Whether to record synpatic traces.
        self.s = torch.zeros(self.shape).byte()  # Spike occurences.
        
        if self.traces:
            self.x = torch.zeros(self.shape)     # Firing traces.
            self.trace_tc = trace_tc             # Rate of decay of spike trace time constant.

    @abstractmethod
    def step(self, inpts, dt):
        '''
        Abstract base class method for a single simulation step.
        
        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        if self.traces:
            # Decay and set spike traces.
            self.x -= dt * self.trace_tc * self.x
            self.x.masked_fill_(self.s, 1)
    
    @abstractmethod
    def _reset(self):
        '''
        Abstract base class method for resetting state variables.
        '''
        self.s = torch.zeros(self.shape).byte()  # Spike occurences.

        if self.traces:
            self.x = torch.zeros(self.shape)     # Firing traces.


class Input(Nodes):
    '''
    Layer of nodes with user-specified spiking behavior.
    '''
    def __init__(self, n=None, shape=None, traces=False, trace_tc=5e-2):
        '''
        Instantiates a layer of input neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record decaying spike traces.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
        '''
        super().__init__(n, shape, traces, trace_tc)

    def step(self, inpts, dt):
        '''
        On each simulation step, set the spikes of the population equal to the inputs.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Set spike occurrences to input values.
        self.s = inpts.byte()
        
        super().step(inpts, dt)
    
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()


class McCullochPitts(Nodes):
    '''
    Layer of `McCulloch-Pitts neurons <http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html>`_.
    '''
    def __init__(self, n=None, shape=None, traces=False, thresh=1.0, trace_tc=5e-2):
        '''
        Instantiates a McCulloch-Pitts layer of neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
        '''
        super().__init__(n, shape, traces, trace_tc)

        self.thresh = thresh             # Spike threshold voltage.
        self.v = torch.zeros(self.shape) # Neuron voltages.

    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        self.v = inpts                  # Voltages are equal to the inputs.
        self.s = self.v >= self.thresh  # Check for spiking neurons.

        super().step(inpts, dt)
    
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()


class IFNodes(Nodes):
    '''
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    '''
    def __init__(self, n=None, shape=None, traces=False, thresh=-52.0, reset=-65.0, refrac=5, trace_tc=5e-2):
        '''
        Instantiates a layer of IF neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
            | :code:`reset` (:code:`float`): Post-spike reset voltage.
            | :code:`refrac` (:code:`int`): Refractory (non-firing) period of the neuron.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
        '''
        
        super().__init__(n, shape, traces, trace_tc)

        self.reset = reset    # Post-spike reset voltage.
        self.thresh = thresh  # Spike threshold voltage.
        self.refrac = refrac  # Post-spike refractory period.

        self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)   # Refractory period counters.

    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt
    
        # Check for spiking neurons.
        self.s = (self.v >= self.thresh) & (self.refrac_count == 0)

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Integrate input and decay voltages.
        self.v += inpts

        super().step(inpts, dt)

    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class LIFNodes(Nodes):
    '''
    Layer of leaky integrate-and-fire (LIF) neurons.
    '''
    def __init__(self, n=None, shape=None, traces=False, thresh=-52.0, rest=-65.0,
                 reset=-65.0, refrac=5, decay=1e-2, trace_tc=5e-2):
        '''
        Instantiates a layer of LIF neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
            | :code:`rest` (:code:`float`): Resting membrane voltage.
            | :code:`reset` (:code:`float`): Post-spike reset voltage.
            | :code:`refrac` (:code:`int`): Refractory (non-firing) period of the neuron.
            | :code:`decay` (:code:`float`): Time constant of neuron voltage decay.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
        '''
        super().__init__(n, shape, traces, trace_tc)

        self.rest = rest       # Rest voltage.
        self.reset = reset     # Post-spike reset voltage.
        self.thresh = thresh   # Spike threshold voltage.
        self.refrac = refrac   # Post-spike refractory period.
        self.decay = decay # Rate of decay of neuron voltage.

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Decay voltages.
        self.v -= dt * self.decay * (self.v - self.rest)
        
        # Decrement refrac counters.
        self.refrac_count[self.refrac_count != 0] -= dt
        
        # Check for spiking neurons.
        self.s = (self.v >= self.thresh) & (self.refrac_count == 0)

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        
        # Integrate inputs.
        self.v += inpts

        super().step(inpts, dt)
        
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class CurrentLIFNodes(Nodes):
    '''
    Layer of current-based leaky integrate-and-fire (LIF) neurons.
    '''
    def __init__(self, n=None, shape=None, traces=False, thresh=-52.0, rest=-65.0,
                 reset=-65.0, refrac=5, decay=1e-2, i_decay=2e-2, trace_tc=5e-2):
        '''
        Instantiates a layer of synaptic input current-based LIF neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
            | :code:`rest` (:code:`float`): Resting membrane voltage.
            | :code:`reset` (:code:`float`): Post-spike reset voltage.
            | :code:`refrac` (:code:`int`): Refractory (non-firing) period of the neuron.
            | :code:`decay` (:code:`float`): Time constant of neuron voltage decay.
            | :code:`i_decay` (:code:`float`): Time constant of synaptic input current decay.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
        '''
        super().__init__(n, shape, traces, trace_tc)

        self.rest = rest       # Rest voltage.
        self.reset = reset     # Post-spike reset voltage.
        self.thresh = thresh   # Spike threshold voltage.
        self.refrac = refrac   # Post-spike refractory period.
        self.decay = decay # Rate of decay of neuron voltage.
        self.i_decay = i_decay # Rate of decay of synaptic input current.

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.i = torch.zeros(self.shape)             # Synaptic input currents.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Decay voltages and current.
        self.v -= dt * self.decay * (self.v - self.rest)
        self.i -= dt * self.i_decay * self.i
        
        # Decrement refrac counters.
        self.refrac_count[self.refrac_count != 0] -= dt
        
        # Check for spiking neurons.
        self.s = (self.v >= self.thresh) & (self.refrac_count == 0)

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        
        # Integrate inputs.
        self.i += inpts
        self.v += self.i

        super().step(inpts, dt)
        
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.i = torch.zeros(self.shape)             # Synaptic input currents.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class AdaptiveLIFNodes(Nodes):
    '''
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds.
    '''
    def __init__(self, n=None, shape=None, traces=False, rest=-65.0, reset=-65.0, thresh=-52.0,
                 refrac=5, decay=1e-2, trace_tc=5e-2, theta_plus=0.05, theta_decay=1e-7):
        '''
        Instantiates a layer of LIF neurons with adaptive firing thresholds.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`rest` (:code:`float`): Resting membrane voltage.
            | :code:`reset` (:code:`float`): Post-spike reset voltage.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
            | :code:`refrac` (:code:`int`): Refractory (non-firing) period of the neuron.
            | :code:`decay` (:code:`float`): Time constant of neuron voltage decay.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
            | :code:`theta_plus` (:code:`float`): Voltage increase of threshold after spiking.
            | :code:`theta_decay` (:code:`float`): Time constant of adaptive threshold decay.
        '''
        super().__init__(n, shape, traces, trace_tc)

        self.rest = rest                # Rest voltage.
        self.reset = reset              # Post-spike reset voltage.
        self.thresh = thresh            # Spike threshold voltage.
        self.refrac = refrac            # Post-spike refractory period.
        self.decay = decay          # Rate of decay of neuron voltage.
        self.theta_plus = theta_plus    # Constant threshold increase on spike.
        self.theta_decay = theta_decay  # Rate of decay of adaptive thresholds.

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.theta = torch.zeros(self.shape)         # Adaptive thresholds.

        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.
        
    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Decay voltages and adaptive thresholds.
        self.v -= dt * self.decay * (self.v - self.rest)
        self.theta -= dt * self.theta_decay * self.theta
        
        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = (self.v >= self.thresh + self.theta) & (self.refrac_count == 0)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.theta += self.theta_plus * self.s.float()
        
        # Integrate inputs.
        self.v += inpts

        super().step(inpts, dt)
        
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class DiehlAndCookNodes(Nodes):
    '''
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015 replication).
    '''
    def __init__(self, n=None, shape=None, traces=False, rest=-65.0, reset=-65.0, thresh=-52.0,
                 refrac=5, decay=1e-2, trace_tc=5e-2, theta_plus=0.05, theta_decay=1e-7):
        '''
        Instantiates a layer of Diehl & Cook 2015 neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`rest` (:code:`float`): Resting membrane voltage.
            | :code:`reset` (:code:`float`): Post-spike reset voltage.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
            | :code:`refrac` (:code:`int`): Refractory (non-firing) period of the neuron.
            | :code:`decay` (:code:`float`): Time constant of neuron voltage decay.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
            | :code:`theta_plus` (:code:`float`): Voltage increase of threshold after spiking.
            | :code:`theta_decay` (:code:`float`): Time constant of adaptive threshold decay.
        '''
        super().__init__(n, shape, traces, trace_tc)

        self.rest = rest                # Rest voltage.
        self.reset = reset              # Post-spike reset voltage.
        self.thresh = thresh            # Spike threshold voltage.
        self.refrac = refrac            # Post-spike refractory period.
        self.decay = decay          # Rate of decay of neuron voltage.
        self.theta_plus = theta_plus    # Constant threshold increase on spike.
        self.theta_decay = theta_decay  # Rate of decay of adaptive thresholds.

        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.theta = torch.zeros(self.shape)         # Adaptive thresholds.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Decay voltages and adaptive thresholds.
        self.v -= dt * self.decay * (self.v - self.rest)
        self.theta -= dt * self.theta_decay * self.theta
        
        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = (self.v >= self.thresh + self.theta) & (self.refrac_count == 0)

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        self.theta += self.theta_plus * self.s.float()
        
        # Choose only a single neuron to spike.
        if torch.sum(self.s) > 0:
            s = torch.zeros(self.s.size())
            s = s.view(-1)
            s[torch.multinomial(self.s.float().view(-1), 1)] = 1
            self.s = s.view(self.s.size()).byte()
        
        # Integrate inputs.
        self.v += inpts

        super().step(inpts, dt)
        
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.


class IzhikevichNodes(Nodes):
    '''
    Layer of Izhikevich neurons.
    '''
    def __init__(self, n=None, shape=None, traces=False, excitatory=True, rest=-65.0, reset=-65.0,
                    thresh=-52.0, refrac=5, decay=1e-2, trace_tc=5e-2):
        '''
        Instantiates a layer of Izhikevich neurons.
        
        Inputs:
        
            | :code:`n` (:code:`int`): The number of neurons in the layer.
            | :code:`shape` (:code:`iterable[int]`): The dimensionality of the layer.
            | :code:`traces` (:code:`bool`): Whether to record spike traces.
            | :code:`excitatory` (:code:`bool`): Whether layer is excitatory.
            | :code:`rest` (:code:`float`): Resting membrane voltage.
            | :code:`reset` (:code:`float`): Post-spike reset voltage.
            | :code:`thresh` (:code:`float`): Spike threshold voltage.
            | :code:`refrac` (:code:`int`): refrac (non-firing) period of the neuron.
            | :code:`decay` (:code:`float`): Time constant of neuron voltage decay.
            | :code:`trace_tc` (:code:`float`): Time constant of spike trace decay.
        '''
        super().__init__(n, shape, traces, trace_tc)

        self.rest = rest       # Rest voltage.
        self.reset = reset     # Post-spike reset voltage.
        self.thresh = thresh   # Spike threshold voltage.
        self.refrac = refrac   # Post-spike refractory period.
        self.decay = decay # Rate of decay of neuron voltage.
        
        if excitatory:
            self.r = torch.rand(n)
            self.a = 0.02 * torch.ones(n)
            self.b = 0.2 * torch.ones(n)
            self.c = -65.0 + 15 * (self.r ** 2)
            self.d = 8 - 6 * (self.r ** 2)
        else:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * torch.ones(n)
            self.c = -65.0 * (self.re ** 2)
            self.d = 2 * torch.ones(n)
        
        self.v = self.rest * torch.ones(n)  # Neuron voltages.
        self.u = self.b * self.v            # Neuron recovery.
        self.refrac_count = torch.zeros(n)  # Refractory period counters.

    def step(self, inpts, dt):
        '''
        Runs a single simulation step.

        Inputs:
        
            | :code:`inpts` (:code:`torch.Tensor`): Inputs to the layer.
            | :code:`dt` (:code:`float`): Simulation time step.
        '''
        # Decrement refrac counters.
        self.refrac_count[self.refrac_count != 0] -= dt
        
        # Check for spiking neurons.
        self.s = (self.v >= self.thresh) & (self.refrac_count == 0)
        
        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        
        # Apply v and u updates.
        self.v += dt * (0.04 * (self.v ** 2) + 5 * self.v + 140 - self.u + inpts)
        self.u += self.a * (self.b * self.v - self.u)
        
        super().step(inpts, dt)
        
    def _reset(self):
        '''
        Resets relevant state variables.
        '''
        super()._reset()
        self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
        self.u = self.b * self.v                     # Neuron recovery.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.
