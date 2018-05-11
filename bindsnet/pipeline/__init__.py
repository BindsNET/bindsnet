import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from .feedback           import *
from ..analysis.plotting import *
from time                import time
from ..encoding          import bernoulli

plt.ion()

class Pipeline:
	'''
	Abstracts the interaction between network, environment (or dataset), input encoding, and environment feedback.
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
			
				| :code:`time` (:code:`int`): Time input is presented for to the network.
				| :code:`history` (:code:`int`): Number of observations to keep track of.
				| :code:`delta` (:code:`int`): Step size to save observations in history. 
				| :code:`output` (:code:`str`): String name of the layer from which to take output from.
				| :code:`save_dir` (:code:`str`): Directory to save network object to.
				| :code:`render_interval` (:code:`bool`): Interval tp show the environment.
				| :code:`plot_interval` (:code:`int`): Interval to update plots.
				| :code:`print_interval` (:code:`int`): Interval to print text output.
				| :code:`save_interval` (:code:`int`): How often to save the network to disk.
		'''
		self.network = network
		self.env = environment
		self.encoding = encoding
		self.feedback = feedback
		
		self.iteration = 0
		self.ims_s, self.axes_s = None, None
		self.ims_v, self.axes_v = None, None
		
		# Setting kwargs.
		self.time = kwargs.get('time', 1)
		self.output = kwargs.get('output', None)
		self.save_dir = kwargs.get('save_dir', 'network.p')
		self.plot_interval = kwargs.get('plot_interval', None)
		self.save_interval = kwargs.get('save_interval', None)
		self.print_interval = kwargs.get('print_interval', None)
		self.render_interval = kwargs.get('render_interval', None)
		
		if 'history' in kwargs and 'delta' in kwargs:
			self.delta = kwargs['delta']
			self.history_index = 0
			self.history = {i : torch.Tensor() for i in range(0, kwargs['history'] * self.delta, self.delta)}
		else:
			self.history_index = 0
			self.history = {}
			self.delta = 1
		
		if self.plot_interval is not None:
			self.spike_record = {layer : torch.ByteTensor() for layer in self.network.layers}
			self.set_spike_data()
			self.plot_data()

		self.first = True
		self.clock = time()

	def set_spike_data(self):
		'''
		Get the spike data from all layers in the pipeline's network.
		'''
		self.spike_record = {layer : self.network.monitors['%s_spikes' % layer].get('s') for layer in self.network.layers}

	def set_voltage_data(self):
		'''
		Get the voltage data from all applicable layers in the pipeline's network.
		'''
		self.voltage_record = {}
		for layer in self.network.layers:
			if 'v' in self.network.layers[layer].__dict__:
				self.voltage_record[layer] = self.network.monitors['%s_voltages' % layer].get('v')

	def step(self):
		'''
		Run an iteration of the pipeline.
		'''
		if self.print_interval is not None and self.iteration % self.print_interval == 0:
			print('Iteration: %d (Time: %.4f)' % (self.iteration, time() - self.clock))
			self.clock = time()
		
		if self.save_interval is not None and self.iteration % self.save_interval == 0:
			self.network.save(self.save_dir)
		
		# Render game.
		if self.render_interval is not None and self.iteration % self.render_interval == 0:
			self.env.render()
			
		# Choose action based on output neuron spiking.
		action = self.feedback(self, output=self.output)
		
		# Run a step of the environment.
		self.obs, self.reward, self.done, info = self.env.step(action)

		# Store frame of history and encode the inputs.
		if len(self.history) > 0:
			self.update_history()
			self.update_index()
			
		# Encode the observation using given encoding function.
		self.encoded = self.encoding(self.obs, time=self.time, max_prob=self.env.max_prob)
		
		# Run the network on the spike train-encoded inputs.
		self.network.run(inpts={'X' : self.encoded}, time=self.time, reward=self.reward)
		
		# Plot relevant data.
		if self.plot_interval is not None and (self.iteration % self.plot_interval == 0):
			self.plot_data()
			
			if len(self.history) > 0 and not self.iteration < len(self.history) * self.delta:  
				self.plot_obs()
			
		self.iteration += 1

	def plot_obs(self):
		'''
		Plot the processed observation after difference against history
		'''
		if self.first:
			self.fig = plt.figure()
			axes = self.fig.add_subplot(111)
			self.im = axes.imshow(self.obs.numpy().reshape(self.env.obs_shape), cmap='gray')
			self.first = False
		else:
			self.im.set_data(self.obs.numpy().reshape(self.env.obs_shape))
			
	def plot_data(self):
		'''
		Plot desired variables.
		'''
		# Set latest data
		self.set_spike_data()
		self.set_voltage_data()
		
		# Initialize plots
		if self.ims_s is None and self.axes_s is None and self.ims_v is None and self.axes_v is None:
			self.ims_s, self.axes_s = plot_spikes(self.spike_record)
			self.ims_v, self.axes_v = plot_voltages(self.voltage_record)
		else: 
			# Update the plots dynamically
			self.ims_s, self.axes_s = plot_spikes(self.spike_record, ims=self.ims_s, axes=self.axes_s)
			self.ims_v, self.axes_v = plot_voltages(self.voltage_record, ims=self.ims_v, axes=self.axes_v)
		
		plt.pause(1e-8)

	def update_history(self):
		'''
		Updates the observations inside history by performing subtraction from 
		most recent observation and the sum of previous observations.
		
		If there are not enough observations to take a difference from, simply 
		store the observation without any subtraction.
		'''
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
				
	def update_index(self):
		'''
		Updates the index to keep track of history.
		
		For example: history = 4, delta = 3 will produce self.history = {0, 3, 6, 9}
						  and self.history_index will be updated according to self.delta
						  and will wrap around the history dictionary.
		'''
		if self.iteration % self.delta == 0:
			if self.history_index != max(self.history.keys()):
				self.history_index += self.delta
			# Wrap around the history
			else:
				self.history_index %= max(self.history.keys())	
					
	def normalize(self, source, target, norm):
		'''
		Normalize a connection in the pipeline's :code:`Network`.
		
		Inputs:
		
			| :code:`source` (:code:`str`): Name of the pre-connection population.
			| :code:`source` (:code:`str`): Name of the post-connection population.
			| :code:`norm` (:code:`float`): Normalization constant of the connection weights.
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
