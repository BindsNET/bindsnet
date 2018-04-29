import sys
import torch
import numpy as np
import matplotlib.pyplot as plt


class Pipeline:
	
	def __init__(self, Network, Environment, plot=False, **kwargs):
		'''
		Initializes the pipeline
		
		Inputs:
			Network (bindsnet.Network): Arbitrary network (e.g ETH)
			Environment (bindsnet.Environment): Arbitrary environment (e.g MNIST, Space Invaders)
		'''
		self.network = Network
		self.env = Environment
		
		# Create figures based on desired plots inside kwargs
		self.axs, self.ims = self.plot()
		
		# Kwargs being assigned to the pipeline
		self.time = kwargs['time']
		self.render = kwargs['render']
		self.history = kwargs['history']
		
		self.iteration = 0		
	def step(self):
		'''
		Step through an iteration 
		'''
		
		# Choose action based on readout neuron spiking.
		if self.network.layers['R'].s == {} or self.network.layers['R'].s.sum() == 0:
			action = np.random.choice(range(6))
		else:
			action = torch.multinomial((self.network.layers['R'].s / self.network.layers['R'].s.sum()).view(-1), 1)[0] + 1
#			action = max( (self.network.layers['R'].s / self.network.layers['R'].s.sum()).view(-1) ) + 1
		
		# If an instance of OpenAI gym environment
		self.obs, self.reward, self.done, self.info = self.env.step(action)
		
		# Based on the environment we could want to render
		if self.render:
			self.env.render()
	
		# Run the network
		self.network.run(inpts={'X': self.obs}, time=self.time)
		
		# Plot any relevant information
		if self.plot:
			self.plot()
		
		self.iteration += 1

		
	def plot(self):
		'''
		Plot monitor variables or desired variables?
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
		Reset the entire pipeline
		'''
		self.env.reset()
		self.network._reset()
	
	
	
	
	
	
	
	
	
		