import torch

from abc import ABC, abstractmethod


class Nodes(ABC):
	'''
	Abstract base class for groups of neurons.
	'''
	def __init__(self):
		'''
		Abstract constructor for the Nodes class.
		'''
		super().__init__()

	@abstractmethod
	def step(self, inpts, dt):
		'''
		Abstract base class method for a single simulation step.
		
		Inputs:
			inpts (dict): Dictionary mapping of Input instances to Tensors inputs.
			dt (float): Simulation timestep.
		'''
		pass

	def get_spikes(self):
		'''
		Returns Tensor of instantaneous spike occurrences.
		
		Returns:
			(torch.Tensor or torch.cuda.Tensor): Spike occurrences.
		'''
		return self.s

	def get_voltages(self):
		'''
		Returns Tensor of instantaneous neuron voltages.
		
		Returns:
			(torch.Tensor or torch.cuda.Tensor): Neuron voltages.
		'''
		return self.v

	def get_traces(self):
		'''
		Returns Tensor of instantaneous spike traces.
		
		Returns:
			(torch.Tensor or torch.cuda.Tensor): Spike traces.
		'''
		if self.traces:
			return self.x


class Input(Nodes):
	'''
	Layer of nodes with user-specified spiking behavior.
	'''
	def __init__(self, n, shape=None, traces=False, trace_tc=5e-2):
		'''
		Instantiates a layer of input neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			shape (iterable[int]): The dimensionality of the layer.
			traces (bool): Whether to record decaying spike traces.
			trace_tc (float): Time constant of spike trace decay.
		'''
		super().__init__()

		self.n = n  # No. of neurons.
		if shape is None:
			self.shape = [self.n]  # Shape is equal to the size of the layer.
		else:
			self.shape = shape  # Shape is passed in as an argument.
			
		self.traces = traces  # Whether to record synpatic traces.
		self.s = torch.zeros(self.shape).byte()  # Spike occurences.
		
		if self.traces:
			self.x = torch.zeros(self.shape)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

	def step(self, inpts, dt):
		'''
		On each simulation step, set the spikes of the
		population equal to the inputs.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Matrix
				of inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		# Set spike occurrences to input values.
		self.s = inpts.byte()

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1

	def _reset(self):
		'''
		Resets relevant state variables.
		'''
		self.s[self.s != 0] = 0  # Spike occurences.

		if self.traces:
			self.x = torch.zeros(self.shape)  # Firing traces.


class McCullochPitts(Nodes):
	'''
	Layer of McCulloch-Pitts neurons.
	'''
	def __init__(self, n, shape=None, traces=False, threshold=1.0, trace_tc=5e-2):
		'''
		Instantiates a McCulloch-Pitts layer of neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			shape (iterable[int]): The dimensionality of the layer.
			traces (bool): Whether to record decaying spike traces.
			threshold (float): Value at which to record a spike.
		'''
		super().__init__()

		self.n = n  # No. of neurons.
		if shape is None:
			self.shape = [self.n]  # Shape is equal to the size of the layer.
		else:
			self.shape = shape  # Shape is passed in as an argument.
		
		self.traces = traces  # Whether to record synpatic traces.
		self.threshold = threshold  # Spike threshold voltage.
		self.v = torch.zeros(self.shape)  # Neuron voltages.
		self.s = torch.zeros(self.shape).byte()  # Spike occurences.

		if self.traces:
			self.x = torch.zeros(self.shape)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

	def step(self, inpts, dt):
		'''
		Runs a single simulation step.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Vector
				of inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		self.v = inpts  # Voltages are equal to the inputs.
		self.s = self.v >= self.threshold  # Check for spiking neurons.

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1

	def _reset(self):
		'''
		Resets relevant state variables.
		'''
		self.s[self.s != 0] = 0  # Spike occurences.

		if self.traces:
			self.x = torch.zeros(self.shape)  # Firing traces.


class IFNodes(Nodes):
	'''
	Layer of integrate-and-fire (IF) neurons.
	'''
	def __init__(self, n, shape=None, traces=False, threshold=-52.0, reset=-65.0, refractory=5, trace_tc=5e-2):
		'''
		Instantiates a layer of IF neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			shape (iterable[int]): The dimensionality of the layer.
			traces (bool): Whether to record decaying spike traces.
			threshold (float): Value at which to record a spike.
			reset (float): Value to which neurons are set to following a spike.
			refractory (int): The number of timesteps following
				a spike during which a neuron cannot spike again.
			trace_tc (float): Time constant of spike trace decay.
		'''
		
		super().__init__()

		self.n = n  # No. of neurons.
		if shape is None:
			self.shape = [self.n]  # Shape is equal to the size of the layer.
		else:
			self.shape = shape  # Shape is passed in as an argument.
		
		self.traces = traces  # Whether to record synpatic traces.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.

		self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
		self.s = torch.zeros(self.shape).byte()  # Spike occurences.

		if traces:
			self.x = torch.zeros(self.shape)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

	def step(self, inpts, dt):
		'''
		Runs a single simulation step.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Vector
				of inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		# Decrement refractory counters.
		self.refrac_count[self.refrac_count > 0] -= dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold) * (self.refrac_count == 0)
		self.refrac_count[self.s] = self.refractory
		self.v[self.s] = self.reset

		# Integrate input and decay voltages.
		self.v += inpts

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1.0

	def _reset(self):
		'''
		Resets relevant state variables.
		'''
		self.s[self.s != 0] = 0  # Spike occurences.
		self.v[self.v != self.rest] = self.rest  # Neuron voltages.
		self.refrac_count[self.refrac_count != 0] = 0  # Refractory period counters.
		
		if self.traces:
			self.x = torch.zeros(self.shape)  # Firing traces.


class LIFNodes(Nodes):
	'''
	Layer of leaky integrate-and-fire (LIF) neurons.
	'''
	def __init__(self, n, shape=None, traces=False, threshold=-52.0, rest=-65.0, reset=-65.0,
									refractory=5, voltage_decay=1e-2, trace_tc=5e-2):
		'''
		Instantiates a layer of LIF neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			shape (iterable[int]): The dimensionality of the layer.
			traces (bool): Whether to record decaying spike traces.
			threshold (float): Value at which to record a spike.
			rest (float): Value to which neuron voltages decay.
			reset (float): Value to which neurons are set to following a spike.
			refractory (int): The number of timesteps following
				a spike during which a neuron cannot spike again.
			voltage_decay (float): Time constant of neuron voltage decay.
			trace_tc (float): Time constant of spike trace decay.
		'''
		super().__init__()

		self.n = n  # No. of neurons.
		if shape is None:
			self.shape = [self.n]  # Shape is equal to the size of the layer.
		else:
			self.shape = shape  # Shape is passed in as an argument.
		
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.

		self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
		self.s = torch.zeros(self.shape).byte()  # Spike occurences.

		if traces:
			self.x = torch.zeros(self.shape)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

	def step(self, inpts, dt):
		'''
		Runs a single simulation step.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Vector
				of inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		# Decay voltages.
		self.v -= dt * self.voltage_decay * (self.v - self.rest)
		
		# Decrement refractory counters.
		self.refrac_count -= dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold) * (self.refrac_count <= 0)
		self.refrac_count[self.s] = self.refractory
		self.v[self.s] = self.reset
		
		# Integrate inputs.
		self.v += inpts

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1.0
		
	def _reset(self):
		'''
		Resets relevant state variables.
		'''
		self.s[self.s != 0] = 0  # Spike occurences.
		self.v[self.v != self.rest] = self.rest  # Neuron voltages.
		self.refrac_count[self.refrac_count != 0] = 0  # Refractory period counters.

		if self.traces:
			self.x[self.x != 0] = 0  # Firing traces.


class AdaptiveLIFNodes(Nodes):
	'''
	Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds.
	'''
	def __init__(self, n, shape=None, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, refractory=5,
							voltage_decay=1e-2, trace_tc=5e-2, theta_plus=0.05, theta_decay=1e-7):
		'''
		Instantiates a layer of LIF neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			shape (iterable[int]): The dimensionality of the layer.
			traces (bool): Whether to record decaying spike traces.
			threshold (float): Value at which to record a spike.
			rest (float): Value to which neuron voltages decay.
			reset (float): Value to which neurons are set to following a spike.
			refractory (int): The number of timesteps following
				a spike during which a neuron cannot spike again.
			voltage_decay (float): Time constant of neuron voltage decay.
			trace_tc (float): Time constant of spike trace decay.
			theta_plus (float): Constant value to increase threshold upon neuron firing.
			theta_decay (float): Time constant of adaptive threshold decay.
		'''
		super().__init__()

		self.n = n  # No. of neurons.
		if shape is None:
			self.shape = [self.n]  # Shape is equal to the size of the layer.
		else:
			self.shape = shape  # Shape is passed in as an argument.
		
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.
		self.theta_plus = theta_plus  # Constant threshold increase on spike.
		self.theta_decay = theta_decay  # Rate of decay of adaptive thresholds.

		self.v = self.rest * torch.ones(self.shape)  # Neuron voltages.
		self.s = torch.zeros(self.shape)  # Spike occurences.
		self.theta = torch.zeros(self.shape)  # Adaptive thresholds.

		if traces:
			self.x = torch.zeros(self.shape)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

	def step(self, inpts, dt):
		'''
		Runs a single simulation step.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Vector
				of inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		# Decay voltages and adaptive thresholds.
		self.v -= dt * self.voltage_decay * (self.v - self.rest)
		self.theta -= dt * self.theta_decay * self.theta
		
		# Decrement refractory counters.
		self.refrac_count -= dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold + self.theta) * (self.refrac_count <= 0)
		self.refrac_count[self.s] = self.refractory
		self.v[self.s] = self.reset
		self.theta += self.theta_plus * self.s.float()
		
		# Choose only a single neuron to spike (ETH replication).
		if torch.sum(self.s) > 0:
			s = torch.zeros(self.s.size())
			s[torch.multinomial(self.s.float(), 1)] = 1
			self.s = s.byte()

		# Integrate inputs.
		self.v += inpts

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1.0

	def _reset(self):
		'''
		Resets relevant state variables.
		'''
		self.s[self.s != 0] = 0  # Spike occurences.
		self.v[self.v != self.rest] = self.rest  # Neuron voltages.
		self.refrac_count[self.refrac_count != 0] = 0  # Refractory period counters.

		if self.traces:
			self.x[self.x != 0] = 0  # Firing traces.


class IzhikevichNodes(Nodes):
	'''
	Layer of Izhikevich neurons.
	'''
	def __init__(self, n, traces=False, excitatory=True, rest=-65.0, reset=-65.0,
					threshold=-52.0, refractory=5, voltage_decay=1e-2, trace_tc=5e-2):
		'''
		Instantiates a layer of Izhikevich neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			traces (bool): Whether to record decaying spike traces.
			excitatory (bool): Whether to use excitatory or inhibitory neurons.
			threshold (float): Value at which to record a spike.
			rest (float): Value to which neuron voltages decay.
			reset (float): Value to which neurons are set to following a spike.
			refractory (int): The number of timesteps following
				a spike during which a neuron cannot spike again.
			voltage_decay (float): Time constant of neuron voltage decay.
			trace_tc (float): Time constant of spike trace decay.
		'''
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.
		
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
		self.u = self.b * self.v;
		self.s = torch.zeros(n)  # Spike occurences.

		if traces:
			self.x = torch.zeros(n)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros(n)  # Refractory period counters.

	def step(self, inpts, dt):
		'''
		Runs a single simulation step.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Vector
				of inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		# Decrement refractory counters.
		self.refrac_count[self.refrac_count != 0] -= dt
		
		# Check for spiking neurons.
		self.s = (self.v >= self.threshold) * (self.refrac_count == 0)
		self.refrac_count[self.s] = self.refractory
		self.v[self.s] = self.reset
		
		# Apply v and u updates.
		self.v += dt * (0.04 * (self.v ** 2) + 5 * self.v + 140 - self.u + inpts)
		self.u += self.a * (self.b * self.v - self.u)
		
		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1.0

	def _reset(self):
		'''
		Resets relevant state variables.
		'''
		self.s = torch.zeros(self.n)  # Spike occurences.
		self.v = self.rest * torch.ones(self.n)  # Neuron voltages.
		self.u = self.b * self.v;
		self.refrac_count = torch.zeros(self.n)  # Refractory period counters.

		if self.traces:
			self.x = torch.zeros(self.n)  # Firing traces.