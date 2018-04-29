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
		self.time = kwargs['time']
		self.render = kwargs['render']
		self.analysis = kwargs['analysis']
		
		
	def step(self):
		'''
		Step through an iteration 
		'''
		
		# Get input from the environment
		inpts = self.env.get()
		
		# Run the network
		self.network.run(inpts=inpts, time=self.time)
		
		# If an instance of OpenAI gym environment
		#obs, reward, done, info = env.step(action)
		
		# Update the plots
		if self.analysis:
			self.plot()
		
		# Based on the environment we could want to render
		if self.render:
			self.env.render()
	
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
	
	
	
	
	
	
	
	
	
		