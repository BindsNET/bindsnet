import torch
import os, sys
import numpy as np
import pickle as p

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))

from nodes import Input


def load_network(fname):
	'''
	Loads serialized network object from disk.
	
	Inputs:
		fname (str): Path to serialized network object on disk.
	'''
	try:
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

	def run(self, inpts, time):
		'''
		Simulation network for given inputs and time.
		
		Inputs:
			inpts (dict): Dictionary including Tensors of shape [time, n_input]
				for n_input per nodes.Input instance. This may be empty if there
				are no user-specified input spikes.
			time (int): Simulation time.
		
		Returns:
			(torch.Tensor or torch.cuda.Tensor): Recording of spikes over simulation episode.
		'''
		timesteps = int(time / self.dt)  # effective no. of timesteps

		# Record spikes from each population over the iteration.
		spikes = {}
		for key in self.layers:
			spikes[key] = torch.zeros(self.layers[key].n, timesteps)

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

				# Record spikes.
				spikes[key][:, timestep] = self.layers[key].s

			# Run synapse updates.
			for synapse in self.connections:
				self.connections[synapse].update()

			# Get input to all layers.
			inpts.update(self.get_inputs())

			# Record state variables of interest.
			for monitor in self.monitors:
				self.monitors[monitor].record()

		return spikes

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
	def __init__(self, obj, state_vars):
		'''
		Constructs a Monitor object.
		
		Inputs:
			obj (Object): Object to record state variables from during network simulation.
			state_vars (list): List of strings indicating names of state variables to record.
		'''
		self.obj = obj
		self.state_vars = state_vars
		
		# Initialize empty recording.
		self.recording = {var : torch.Tensor() for var in self.state_vars}

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
		Appends the instantaneous value of the recorded state variables to the recording.
		'''
		for var in self.state_vars:
			data = self.obj.__dict__[var].view(-1, 1)
			self.recording[var] = torch.cat([self.recording[var], data], 1)

	def _reset(self):
		'''
		Resets recordings to empty Tensors.
		'''
		self.recording = {var : torch.Tensor() for var in self.state_vars}