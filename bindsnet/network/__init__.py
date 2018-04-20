import torch
import pickle as p

from bindsnet.network.nodes import Input


def load_network(fname):
	'''
	Loads serialized network object from disk.
	
	Inputs:
		fname (str): Path to serialized network object on disk.
	'''
	try:
		with open(fname, 'rb') as f:
			return p.load(open(fname, 'rb'))
	except FileNotFoundError:
		print('Network not found on disk.')


class Network:
	'''
	Combines neuron nodes and connections into a spiking neural network.
	'''
	def __init__(self, dt=1.0):
		'''
		Initializes network object.
		
		Inputs:
			dt (float): Simulation timestep. All simulation
				time constants are relative to this value.
		'''
		self.dt = dt
		self.layers = {}
		self.connections = {}
		self.monitors = {}

	def add_layer(self, layer, name):
		'''
		Adds a layer of nodes to the network.
		
		Inputs:
			layer (bindsnet.nodes.Nodes): A subclass of the Nodes object.
			name (str): Logical name of layer.
		'''
		self.layers[name] = layer

	def add_connection(self, connection, source, target):
		'''
		Adds a connection between layers of nodes to the network.
		
		Inputs:
			connections (bindsnet.connection.Connection): An instance of class Connection.
			source (str): Logical name of the connection's source layer.
			target (str): Logical name of the connection's target layer.
		'''
		self.connections[(source, target)] = connection

	def add_monitor(self, monitor, name):
		'''
		Adds a monitor on a network object to the network.
		
		Inputs:
			monitor (bindsnet.Monitor): An instance of class Monitor.
			name (str): Logical name of monitor object.
		'''
		self.monitors[name] = monitor

	def save(self, fname):
		'''
		Serializes the network object to disk.
		
		Inputs:
			fname (str): Path to store serialized network object on disk. 
		'''
		p.dump(self, open(fname, 'wb'))

	def get_inputs(self):
		'''
		Fetches outputs from network layers to use as inputs to connected layers.
		
		Returns:
			(dict[torch.Tensor or torch.cuda.Tensor]): Inputs
				to all layers for the current iteration.
		'''
		inpts = {}
		
		# Loop over network connections.
		for key in self.connections:
			# Fetch source and target populations.
			source = self.connections[key].source
			target = self.connections[key].target
			
			if not key[1] in inpts:
				inpts[key[1]] = torch.zeros_like(torch.Tensor(target.n))

			# Add to input: source's spikes multiplied by connection weights.
			inpt = source.s.float().view(-1) @ self.connections[key].w.view(source.n, target.n)
			inpts[key[1]] += inpt.view(*target.shape)
			
		return inpts

	def run(self, inpts, time, **kwargs):
		'''
		Simulation network for given inputs and time.
		
		Inputs:
			inpts (dict): Dictionary including Tensors of shape [time, n_input]
				for n_input per nodes.Input instance. This may be empty if there
				are no user-specified input spikes.
			time (int): Simulation time.
		'''
		timesteps = int(time / self.dt)  # effective no. of timesteps

		# Get input to all layers.
		inpts.update(self.get_inputs())
		
		# Simulate network activity for `time` timesteps.
		for timestep in range(timesteps):
			# Update each layer of nodes.
			for key in self.layers:
				if type(self.layers[key]) is Input:
					self.layers[key].step(inpts[key][timestep, :], self.dt)
				else:
					self.layers[key].step(inpts[key], self.dt)

			# Run synapse updates.
			for synapse in self.connections:
				if str(synapse) in kwargs:
					self.connections[synapse].update(kwargs[str(synapse)])
				else:
					self.connections[synapse].update({})

			# Get input to all layers.
			inpts.update(self.get_inputs())

			# Record state variables of interest.
			for monitor in self.monitors:
				self.monitors[monitor].record()

	def _reset(self):
		'''
		Reset state variables of objects in network.
		'''
		for layer in self.layers:
			self.layers[layer]._reset()

		for connection in self.connections:
			self.connections[connection]._reset()

		for monitor in self.monitors:
			self.monitors[monitor]._reset()
