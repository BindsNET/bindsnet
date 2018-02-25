import torch

from abc import ABC, abstractmethod

class Nodes(ABC):
	'''
	Abstract base class for groups of neurons.
	'''
	def __init__(self):
		super().__init__()

	@abstractmethod
	def step(self, inpts, dt):
		pass

	def get_spikes(self):
		return self.s

	def get_voltages(self):
		return self.v

	def get_traces(self):
		return self.x


class Input(Nodes):
	'''
	Layer of nodes with user-specified spiking behavior.
	'''
	def __init__(self, n, traces=False, trace_tc=5e-2):
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.
		
		if self.traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

	def step(self, inpts, dt):
		'''
		On each simulation step, set the spikes of the
		population equal to the inputs.
		'''
		# Set spike occurrences to input values.
		self.s = inpts

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1

	def reset(self):
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.

		if self.traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.


class McCullochPitts(Nodes):
	'''
	McCulloch-Pitts neuron.
	'''
	def __init__(self, n, traces=False, threshold=1.0, trace_tc=5e-2):
		'''
		Instantiates a McCulloch-Pitts layer of neurons.
		
		Inputs:
			n (int): The number of neurons in the layer.
			traces (bool): Whether to record decaying spike traces.
			threshold (float): Value at which to record a spike.
		'''
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.threshold = threshold  # Spike threshold voltage.
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.

		if self.traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

	def step(self, inpts, dt):
		'''
		Runs a single simulation step.

		Inputs:
			inpts (torch.FloatTensor or torch.cuda.FloatTensor): Vector of
				inputs to the layer, with size equal to self.n.
			dt (float): Simulation time step.
		'''
		self.s = inpts >= self.threshold  # Check for spiking neurons.

		if self.traces:
			# Decay and set spike traces.
			self.x -= dt * self.trace_tc * self.x
			self.x[self.s] = 1

	def reset(self):
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.

		if self.traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.


class LIFNodes(Nodes):
	'''
	Group of leaky integrate-and-fire neurons.
	'''
	def __init__(self, n, traces=False, rest=-65.0, reset=-65.0, threshold=-52.0, 
								refractory=5, voltage_decay=1e-2, trace_tc=5e-2):
		
		super().__init__()

		self.n = n  # No. of neurons.
		self.traces = traces  # Whether to record synpatic traces.
		self.rest = rest  # Rest voltage.
		self.reset = reset  # Post-spike reset voltage.
		self.threshold = threshold  # Spike threshold voltage.
		self.refractory = refractory  # Post-spike refractory period.
		self.voltage_decay = voltage_decay  # Rate of decay of neuron voltage.

		self.v = self.rest * torch.ones(n)  # Neuron voltages.
		self.s = torch.zeros(n)  # Spike occurences.

		if traces:
			self.x = torch.zeros(n)  # Firing traces.
			self.trace_tc = trace_tc  # Rate of decay of spike trace time constant.

		self.refrac_count = torch.zeros(n)  # Refractory period counters.

	def step(self, inpts, dt):
		# Decay voltages.
		self.v -= dt * self.voltage_decay * (self.v - self.rest)

		if self.traces:
			# Decay spike traces.
			self.x -= dt * self.trace_tc * self.x

		# Decrement refractory counters.
		self.refrac_count[self.refrac_count != 0] -= dt

		# Check for spiking neurons.
		self.s = (self.v >= self.threshold) * (self.refrac_count == 0)
		self.refrac_count[self.s] = self.refractory
		self.v[self.s] = self.reset

		# Integrate input and decay voltages.
		self.v += inpts

		if self.traces:
			# Setting synaptic traces.
			self.x[self.s] = 1.0

	def reset(self):
		self.s = torch.zeros_like(torch.Tensor(n))  # Spike occurences.
		self.v = self.rest * torch.ones(n)  # Neuron voltages.
		self.refrac_count = torch.zeros(n)  # Refractory period counters.

		if self.traces:
			self.x = torch.zeros_like(torch.Tensor(n))  # Firing traces.
