import torch
import os, sys
import numpy as np
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
			inpts[key[1]] += source.s.float() @ self.connections[key].w

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
				self.connections[synapse].update(kwargs)

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


class Monitor:
	'''
	Records state variables of interest.
	'''
	def __init__(self, obj, state_vars, time=None):
		'''
		Constructs a Monitor object.
		
		Inputs:
			obj (Object): Object to record state variables from during network simulation.
			state_vars (list): List of strings indicating names of state variables to record.
			time (int): If not None, pre-allocate memory for state variable recording.
		'''
		self.obj = obj
		self.state_vars = state_vars
		self.time = time
		
		if self.time is not None:
			self.i = 0
		
		# If no simulation time is specified,
		# specify 0-dimensional recordings.
		if self.time is None:
			self.recording = {var : torch.Tensor() for var in self.state_vars}
		
		# If simulation time is specified, pre-
		# allocate recordings in memory for speed.
		else:
			self.recording = {var : torch.zeros(*self.obj.__dict__[var].size(), self.time) for var in self.state_vars}

	def get(self, var):
		'''
		Return recording to user.
		
		Inputs:
			var (str): State variable recording to return.
		
		Returns:
			(torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_1, ..., n_k, time],
				where [n_1, ..., n_k] refers to the shape of the recorded state variable.
		'''
		return self.recording[var]

	def record(self):
		'''
		Appends the current value of the recorded state variables to the recording.
		'''
		if self.time is None:
			for var in self.state_vars:
				data = self.obj.__dict__[var].view(-1, 1).float()
				self.recording[var] = torch.cat([self.recording[var], data], 1)
		else:
			for var in self.state_vars:
				data = self.obj.__dict__[var].unsqueeze(-1)
				
				# 1D data.
				if len(data.size()) - 1 == 1:
					self.recording[var][:, self.i] = data
				# 2D data.
				elif len(data.size()) - 1 == 2:
					self.recording[var][:, :, self.i] = data
		
			self.i += 1

	def _reset(self):
		'''
		Resets recordings to empty torch.Tensors.
		'''
		# If no simulation time is specified,
		# specify 0-dimensional recordings.
		if self.time is None:
			self.recording = {var : torch.Tensor() for var in self.state_vars}
		
		# If simulation time is specified, pre-
		# allocate recordings in memory for speed.
		else:
			self.recording = {var : torch.zeros(*self.obj.__dict__[var].size(), self.time) for var in self.state_vars}
			self.i = 0


class NetworkMonitor:
	'''
	Record state variables of all layers and connections.
	'''
	def __init__(self, network, layers=None, connections=None, state_vars=['v', 's', 'w'], time=None):
		'''
		Constructs a NetworkMonitor object.
		
		Inputs:
			network (bindsnet.network.Network): Network to record state variables from.
			state_vars (list): List of strings indicating names of state variables to record.
			time (int): If not None, pre-allocate memory for state variable recording.
		'''
		self.network = network
		self.state_vars = state_vars
		self.time = time
		
		if self.time is not None:
			self.i = 0
		
		if layers is None:
			self.layers = list(self.network.layers.keys())
		else:
			self.layers = layers
			
		if connections is None:
			self.connections = list(self.network.connections.keys())
		else:
			self.connections = connections
		
		# Initialize empty recording.
		self.recording = {k : {} for k in self.layers + self.connections}
		
		# If no simulation time is specified,
		# specify 0-dimensional recordings.
		if self.time is None:
			for v in self.state_vars:
				for l in self.layers:
					if v in self.network.layers[l].__dict__:
						self.recording[l][v] = torch.Tensor()

				for c in self.connections:
					if v in self.network.connections[c].__dict__:
						self.recording[c][v] = torch.Tensor()
		
		# If simulation time is specified, pre-
		# allocate recordings in memory for speed.
		else:
			for v in self.state_vars:
				for l in self.layers:
					if v in self.network.layers[l].__dict__:
						self.recording[l][v] = torch.zeros(*self.network.layers[l].__dict__[v].size(), self.time)

				for c in self.connections:
					if v in self.network.connections[c].__dict__:
						self.recording[c][v] = torch.zeros(*self.network.connections[c].__dict__[v].size(), self.time)
		
	def get(self):
		'''
		Return entire recording to user.
		
		Returns:
			(dict[torch.Tensor or torch.cuda.Tensor]): Dictionary of
				all layers' and connections' recorded state variables.
		'''
		return self.recording

	def record(self):
		'''
		Appends the current value of the recorded state variables to the recording.
		'''
		if self.time is None:
			for var in self.state_vars:
				for layer in self.layers:
					if var in self.network.layers[layer].__dict__:
						data = self.network.layers[layer].__dict__[var].unsqueeze(-1).float()
						self.recording[layer][var] = torch.cat([self.recording[layer][var], data], -1)

				for connection in self.connections:
					if var in self.network.connections[connection].__dict__:
						data = self.network.connections[connection].__dict__[var].unsqueeze(-1)
						self.recording[connection][var] = torch.cat([self.recording[connection][var], data], -1)
		
		else:
			for var in self.state_vars:
				for layer in self.layers:
					if var in self.network.layers[layer].__dict__:
						data = self.network.layers[layer].__dict__[var].unsqueeze(-1).float()
						self.recording[layer][var][:, self.i] = data

				for connection in self.connections:
					if var in self.network.connections[connection].__dict__:
						data = self.network.connections[connection].__dict__[var].unsqueeze(-1)
						self.recording[connection][var][:, :, self.i] = data
			
			self.i += 1
	
	def save(self, path, fmt='npz'):
		'''
		Write the recording dictionary out to file.
		
		Inputs:
			path (str): The directory to which to write the monitor's recording.
			fmt (str): Type of file to write to disk. One of "pickle" or "npz".
		'''
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		
		if fmt == 'npz':
			# Build a list of arrays to write to disk.
			arrays = {}
			for obj in self.recording:
				if type(obj) == tuple:
					arrays.update({'_'.join(['-'.join(obj), var]) : self.recording[obj][var] for var in self.recording[obj]})
				elif type(obj) == str:
					arrays.update({'_'.join([obj, var]) : self.recording[obj][var] for var in self.recording[obj]})
				
			np.savez_compressed(path, **arrays)
			
		elif fmt == 'pickle':
			with open(path, 'wb') as f:
				p.dump(self.recording, f, protocol=4)
		
	def _reset(self):
		'''
		Resets recordings to empty Tensors.
		'''
		# Reset to empty recordings
		self.i = 0
		self.recording = {k : {} for k in self.layers + self.connections}
		
		# If no simulation time is specified,
		# specify 0-dimensional recordings.
		if self.time is None:
			for v in self.state_vars:
				for l in self.layers:
					if v in self.network.layers[l].__dict__:
						self.recording[l][v] = torch.Tensor()

				for c in self.connections:
					if v in self.network.connections[c].__dict__:
						self.recording[c][v] = torch.Tensor()
		
		# If simulation time is specified, pre-
		# allocate recordings in memory for speed.
		else:
			for v in self.state_vars:
				for l in self.layers:
					if v in self.network.layers[l].__dict__:
						self.recording[l][v] = torch.zeros(self.network.layers[l].n, self.time)

				for c in self.connections:
					if v in self.network.connections[c].__dict__:
						self.recording[c][v] = torch.zeros(*self.network.connections[c].w.size(), self.time)
