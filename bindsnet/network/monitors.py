import os
import torch
import numpy as np
import pickle as p


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
					self.recording[var][:, self.i % self.time] = data
				# 2D data.
				elif len(data.size()) - 1 == 2:
					self.recording[var][:, :, self.i % self.time] = data
		
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
						self.recording[layer][var][:, self.i % self.time] = data

				for connection in self.connections:
					if var in self.network.connections[connection].__dict__:
						data = self.network.connections[connection].__dict__[var].unsqueeze(-1)
						self.recording[connection][var][:, :, self.i % self.time] = data
			
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
