import torch

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
			self.recording[var] = torch.cat([self.recording[var], self.obj.__dict__[var]])

	def reset(self):
		self.recording = {var : torch.Tensor() for var in self.state_vars}