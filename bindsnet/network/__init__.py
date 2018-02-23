import torch
import os, sys
import numpy as np
import pickle as p

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))

from nodes import Input


def load_network(fname):
	try:
		return p.load(open(fname, 'rb'))
	except FileNotFoundError:
		print('Network not found on disk.')


class Network:
	'''
	Combines neuron nodes and connections into a spiking neural network.
	'''
	def __init__(self, dt=1):
		self.dt = dt
		self.layers = {}
		self.connections = {}
		self.monitors = {}

	def add_layer(self, layer, name):
		self.layers[name] = layer

	def add_connection(self, connection, source, target):
		self.connections[(source, target)] = connection

	def add_monitor(self, monitor, name):
		self.monitors[name] = monitor

	def train(self, inputs, targets):
		pass

	def test(self, inputs):
		pass

	def evaluate(self, targets, predictions):
		pass

	def save(self, fname):
		p.dump(self, open(fname, 'wb'))

	def get_inputs(self):
		inpts = {}
		for key in self.connections:
			weights = self.connections[key].w

			source = self.connections[key].source
			target = self.connections[key].target

			if not key[1] in inpts:
				inpts[key[1]] = torch.zeros_like(torch.Tensor(target.n))

			inpts[key[1]] += source.s.float() @ weights

		return inpts

	def run(self, inpts, time):
		'''
		Run network for a single iteration.
		'''
		timesteps = int(time / self.dt)

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
					self.layers[key].step(inpts[key][:, timestep], self.dt)
				else:
					self.layers[key].step(inpts[key], self.dt)

				# Record spikes.
				spikes[key][:, timestep] = self.layers[key].s

			if self.train:
				# Update synapse weights.
				for synapse in self.connections:
					self.connections[synapse].update()

			# Get input to all layers.
			inpts.update(self.get_inputs())

			for monitor in self.monitors:
				self.monitors[monitor].record()

		return spikes

	def reset(self):
		'''
		Reset state variables of objects in network.
		'''
		for layer in self.layers:
			self.layers[layer].reset()

		for connection in self.connections:
			self.connections[connection].reset()

		for monitor in self.monitors:
			self.monitors[monitor].reset()

class Monitor:
	'''
	Records state variables of interest.
	'''
	def __init__(self, obj, state_vars):
		self.obj = obj
		self.state_vars = state_vars
		self.recording = {var : torch.Tensor() for var in self.state_vars}

	def get(self, var):
		return self.recording[var]

	def record(self):
		for var in self.state_vars:
			data = self.obj.__dict__[var].view(-1, 1)
			self.recording[var] = torch.cat([self.recording[var], data], 1)

	def reset(self):
		self.recording = {var : torch.Tensor() for var in self.state_vars}