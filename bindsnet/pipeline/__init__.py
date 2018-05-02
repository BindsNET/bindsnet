import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from ..encoding import *

class Pipeline:
	'''
	| Allows for the abstraction of the interaction between spiking neural network,
	| environment (or dataset), and encoding of inputs into spike trains.
	'''
	def __init__(self, network, environment, encoding=bernoulli, **kwargs):
		'''
		Initializes the pipeline,
		
		Inputs:
			| :code:`network` (:code:`bindsnet.Network`): Arbitrary network object.
			| :code:`environment` (:code:`bindsnet.Environment`): Arbitrary environment (e.g MNIST, Space Invaders)
			| :code:`encoding` (:code:`function`): Function to encode observation into spike trains
			| :code:`kwargs`:
				| :code:`plot` (:code:`bool`): Plot monitor variables.
				| :code:`render` (:code:`bool`): Show the environment.
				| :code:`time` (:code:`int`): Time input is presented for to the network.
				| :code:`history` (:code:`int`): Number of observations to keep track of.
		'''
		self.network = network
		self.env = environment
		self.encoding = encoding
		
		self.time = kwargs['time']
		self.render = kwargs['render']
		self.history = {i : torch.Tensor() for i in range(kwargs['history'])}
		self.plot = kwargs['plot']
		
		# Default plotting behavior.
		if self.plot:
			self.axs, self.ims = self.plot()
		
		self.iteration = 0
	
	def step(self):
		'''
		Step through an iteration of pipeline.
		'''
		if self.iteration % 100 == 0:
			print('Iteration %d' % self.iteration)
		
		if self.render:
			self.env.render()
			
		# Choose action based on readout neuron spiking.
		if self.network.layers['R'].s.sum() == 0 or self.iteration == 0:
			action = np.random.choice(range(6))
		else:
			action = torch.multinomial((self.network.layers['R'].s.float() / self.network.layers['R'].s.sum().float()).view(-1), 1)[0] + 1
		
		# If an instance of OpenAI gym environment
		self.obs, self.reward, self.done, info = self.env.step(action)
		
		# Store frame of history and encode the inputs.
		if len(self.history) > 0:
			if self.iteration < len(self.history):  # Recording initial observations
				# Add current observation to the history buffer.
				self.history[self.iteration] = self.env.obs
				self.encoded = next(self.encoding(self.env.obs, max_prob=self.env.max_prob)).unsqueeze(0)
			else:
				# Subtract off overlapping data from the history buffer.
				new_obs = torch.clamp(self.env.obs - sum(self.history.values()), 0, 1)		
				self.history[self.iteration % len(self.history)] = self.env.obs
				
				# Encode the new observation.
				self.encoded = next(self.encoding(new_obs, max_prob=self.env.max_prob)).unsqueeze(0)
		
		# Encode the observation without any history.
		else:
			self.encoded = next(self.encoding(self.obs, max_prob=self.env.max_prob)).unsqueeze(0)
		
		# Run the network on the spike train encoded inputs.
		self.network.run(inpts={'X': self.encoded}, time=self.time)
		
		if self.plot:
			self.plot()
		
		self.iteration += 1

	def plot(self):
		'''
		Plot desired variables.
		'''
		if self.ims == None and self.axs == None:
			# Initialize plots
			pass
		else: # Update the plots dynamically
			pass
		
	def normalize(self, src, target, norm):
		self.network.connections[(src, target)].normalize(norm)
		
	def reset(self):
		'''
		Reset the entire pipeline.
		'''
		self.env.reset()
		self.network._reset()
		self.iteration = 0
		self.history = {i: torch.Tensor() for i in len(self.history)}

