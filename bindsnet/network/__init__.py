import torch
import os, sys
import numpy as np
import pickle as p

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))


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

	def add_connections(self, connections, source, target):
		self.connections[(source, target)] = connections

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
		# Record spikes from each population over the iteration.
		spikes = {}
		for key in self.nodes:
			spikes[key] = torch.zeros(int(time / self.dt), self.nodes[key].n)

		for monitor in self.monitors:
			self.monitors[monitor].reset()

		# Get inputs to all layers from their parent layers.
		inpts.update(self.get_inputs())
		
		# Simulate neuron and synapse activity for `time` timesteps.
		for timestep in range(int(time / self.dt)):
			# Update each layer of nodes in turn.
			for key in self.layers:
				self.layers[key].step(inpts[key], self.dt)

				# Record spikes from this population at this timestep.
				spikes[key][timestep, :] = self.layers[key].s

			# Update synapse weights if we're in training mode.
			if self.train:
				for synapse in self.connections:
					if type(self.connections[synapse]) == connections.STDPconnections:
						self.connections[synapse].update()

			# Get inputs to all layers from their parent layers.
			inpts.update(self.get_inputs())

			for monitor in self.monitors:
				self.monitors[monitor].record()

		# Normalize synapse weights if we're in training mode.
		if self.train:
			for synapse in self.connections:
				if type(self.connections[synapse]) == connections.STDPconnections:
					self.connections[synapse].normalize()

		return spikes

	def reset(self, attrs):
		'''
		Reset state variables.
		'''
		for layer in self.layers:
			for attr in attrs:
				if hasattr(self.layers[layer], attr):
					self.layers[layer].reset(attr)