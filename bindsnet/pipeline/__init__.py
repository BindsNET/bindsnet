import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from .feedback           import *
from ..analysis.plotting import *
from ..encoding          import bernoulli

plt.ion()

class Pipeline:
	'''
	Allows for the abstraction of the interaction between spiking neural network, environment (or dataset), and encoding of inputs into spike trains.
	'''
	def __init__(self, network, environment, encoding=bernoulli, feedback=no_feedback, **kwargs):
		'''
		Initializes the pipeline.
		
		Inputs:
		
			| :code:`network` (:code:`bindsnet.Network`): Arbitrary network object.
			| :code:`environment` (:code:`bindsnet.Environment`): Arbitrary environment.
			| :code:`encoding` (:code:`function`): Function to encode observations into spike trains.
			| :code:`feedback` (:code:`function`): Function to convert network outputs into environment inputs.
			| :code:`kwargs`:
			
				| :code:`plot` (:code:`bool`): Plot monitor variables.
				| :code:`render` (:code:`bool`): Show the environment.
				| :code:`plot_interval` (:code:`int`): Interval to update plots.
				| :code:`time` (:code:`int`): Time input is presented for to the network.
				| :code:`history` (:code:`int`): Number of observations to keep track of.
				| :code:`delta` (:code:`int`): Step size to save observations in history. 
				| :code:`output` (:code:`str`): String name of the layer from which to take output from.
		'''
		self.network = network
		self.env = environment
		self.encoding = encoding
		self.feedback = feedback
		
		self.iteration = 0
		self.ims, self.axes = None, None
		
		# Setting kwargs.
		if 'time' in kwargs:
			self.time = kwargs['time']
		else:
			self.time = 1
		
		if 'render' in kwargs:
			self.render = kwargs['render']
		else:
			self.render = False
		
		if 'history' in kwargs and 'delta' in kwargs:
			self.delta = kwargs['delta']
			self.history_index = 0
			self.history = {i : torch.Tensor() for i in range(0, kwargs['history']*self.delta, self.delta)}
		else:
			self.history_index = 0
			self.history = {}
			self.delta = 1
		
		if 'plot' in kwargs:
			self.plot = kwargs['plot']
		else:
			self.plot = False
		
		if 'plot_interval' in kwargs:
			self.plot_interval = kwargs['plot_interval']
		else:
			self.plot_interval = 100
		
		if 'output' in kwargs:
			self.output = kwargs['output']
		else:
			self.output = None
			
		if self.plot:
			self.layers_to_plot = [layer for layer in self.network.layers]
			self.spike_record = {layer : torch.ByteTensor() for layer in self.layers_to_plot}
			self.set_spike_data()
			self.plot_data()

		self.first = True

	def set_spike_data(self):
		for layer in self.layers_to_plot:
			self.spike_record[layer] = self.network.monitors['%s_spikes' % layer].get('s')

	def get_voltage_data(self):
		voltage_record = {layer : voltages[layer].get('v') for layer in voltages}
		return voltage_record

	def step(self):
		'''
		Run an iteration of the pipeline.
		'''
		# Render game.
		if self.render:
			self.env.render()
			
		# Choose action based on readout neuron spiking
		action = self.feedback(self, output=self.output)
		
		# Run a step of the environment.
		self.obs, self.reward, self.done, info = self.env.step(action)
		
		# Store frame of history and encode the inputs
		if len(self.history) > 0:
			# Recording initial observations
			if self.iteration < len(self.history) * self.delta:
				# Store observation based on delta value
				if self.iteration % self.delta == 0:
					self.history[self.history_index] = self.obs
			else:
				# Take difference between stored frames and current frame
				temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)
								
				# Store observation based on delta value.
				if self.iteration % self.delta == 0:
					self.history[self.history_index] = self.obs
					
				self.obs = temp
		
		# Encode the observation using given encoder function
		if 'max_prob' in self.env.__dict__:
			self.encoded = self.encoding(self.obs, time=self.time, max_prob=self.env.max_prob)
		else:
			self.encoded = self.encoding(self.obs, time=self.time)
			
		# Run the network on the spike train-encoded inputs.
		self.network.run(inpts={'X' : self.encoded}, time=self.time)
		
		# Update counter
		if len(self.history) > 0:
			if self.iteration % self.delta == 0:
				if self.history_index != max(self.history.keys()):
					self.history_index += self.delta
				# Wrap around the history
				else:
					self.history_index %= max(self.history.keys())
						
		# Plot relevant data
		if self.plot and (self.iteration % self.plot_interval == 0):
			self.plot_data()
			
			if len(self.history) > 0 and not self.iteration < len(self.history) * self.delta:  
				self.plot_obs(self.obs)
			
		self.iteration += 1

	def plot_obs(self, obs):
		if self.first:
			self.fig = plt.figure()
			axes = self.fig.add_subplot(111)
			self.im = axes.imshow(obs.numpy().reshape(78, 84), cmap='gray')
			self.first = False
		else:
			self.im.set_data(obs.numpy().reshape(78, 84))
			
	def plot_data(self):
		'''
		Plot desired variables.
		'''
		# Set data
		self.set_spike_data()
		
		# Initialize plots
		if self.ims == None and self.axes == None:
			self.ims, self.axes = plot_spikes(self.spike_record)
		else: 
			# Update the plots dynamically
			self.ims, self.axes = plot_spikes(self.spike_record, ims=self.ims, axes=self.axes)
		
		plt.pause(1e-8)

	def normalize(self, source, target, norm):
		'''
		Normalize a connection in the pipeline's :code:`Network`.
		
		Inputs:
		
			:code:`source` (:code:`str`): Name of the pre-connection population.
			:code:`source` (:code:`str`): Name of the post-connection population.
			:code:`norm` (:code:`float`): Normalization constant of the connection weights.
		'''
		self.network.connections[(source, target)].normalize(norm)
	
	def _reset(self):
		'''
		Reset the pipeline.
		'''
		self.env.reset()
		self.network._reset()
		self.iteration = 0
		self.history = self.history = {i: torch.Tensor() for i in self.history}
