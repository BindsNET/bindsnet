import sys
import matplotlib.pyplot as plt


class Pipeline:
	
	
	def __init__(self, Network, Environment, **kwargs):
		'''
		'''
		self.network = Network
		self.env = Environment
		
		# Create figures based on desired plots inside kwargs
		self.axs, self.ims = self.plot()
		
		
	def step(self, time, render=False, plot=False):
		'''
		
		'''
		
		# Get input from the environment
		inpts = self.env.get()
		
		# Run the network
		self.network.run(inpts=inpts, time=time)
		
		# If an instance of OpenAI gym environment
		#obs, reward, done, info = env.step(action)
		
		# Update the plots
		if plot:
			self.plot()
		
		# Based on the environment we could want to render
		if render:
			self.env.render()
	
	def plot(self, inpt):
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
	
	
	
	
	
	
	
	
	
	
		