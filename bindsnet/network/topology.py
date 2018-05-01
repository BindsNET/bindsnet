import torch
import numpy as np

from ..learning import *


class Connection:
	'''
	Specifies synapses between one or two populations of neurons.
	'''
	def __init__(self, source, target, w=None, update_rule=None, nu=1e-2,
							nu_pre=1e-4, nu_post=1e-2, wmin=0.0, wmax=1.0):
		'''
		Instantiates a :code:`Connection` object.

		Inputs:
			| :code:`source` (:code:`nodes`.Nodes): A layer of nodes from which the connection originates.
			| :code:`target` (:code:`nodes`.Nodes): A layer of nodes to which the connection connects.
			| :code:`w` (:code:`torch`.FloatTensor or torch.cuda.FloatTensor): Effective strengths of synapses.
			| :code:`update_rule` (:code:`function`): Modifies connection parameters according to some rule.
			| :code:`nu` (:code:`float`): Learning rate for both pre- and post-synaptic events.
			| :code:`nu_pre` (:code:`float`): Learning rate for pre-synaptic events.
			| :code:`nu_post` (:code:`float`): Learning rate for post-synpatic events.
			| :code:`wmin` (:code:`float`): The minimum value on the connection weights.
			| :code:`wmax` (:code:`float`): The maximum value on the connection weights.
		'''
		self.source = source
		self.target = target
		self.nu = nu
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmin = wmin
		self.wmax = wmax

		if update_rule is None:
			self.update_rule = no_update
		else:
			if update_rule is m_stdp or update_rule is m_stdp_et:
				self.e_trace = 0
				self.tc_e_trace = 0.04
				self.p_plus = 0
				self.tc_plus = 0.05
				self.p_minus = 0
				self.tc_minus = 0.05
				
			self.update_rule = update_rule
		
		if w is None:
			self.w = self.wmin + (self.wmax - self.wmin) * torch.rand(*source.shape, *target.shape)
		else:
			self.w = w
			self.w = torch.clamp(self.w, self.wmin, self.wmax)

	def get_weights(self):
		'''
		Retrieve weight matrix of the connection.
		
		Returns:
			| (:code:`torch.Tensor`): Weight matrix of the connection.
		'''
		return self.w

	def set_weights(self, w):
		'''
		Set weight matrix of the connection.
		
		Inputs:
			| :code:`w` (:code:`torch.Tensor`): Weight matrix to set to connection.
		'''
		self.w = w

	def update(self, kwargs):
		'''
		Compute connection's update rule.
		'''
		self.update_rule(self, **kwargs)
	
	def normalize(self, norm=78.0):
		'''
		Normalize weights along the first axis according
		to some desired summed weight per target neuron.
		
		Inputs:
			| :code:`norm` (:code:`float`): Desired sum of weights per target neuron.
		'''
		self.w = self.w.view(self.source.n, self.target.n)
		self.w *= norm / self.w.sum(0).view(1, -1)
		self.w = self.w.view(*self.source.shape, *self.target.shape)
		
	def _reset(self):
		'''
		Contains resetting logic for the connection.
		'''
		pass


class SparseConnection:
	'''
	Specifies sparse synapses between one or two populations of neurons.
	'''
	def __init__(self, source, target, w=None, sparsity=0.9, update_rule=None, nu=1e-2,
										nu_pre=1e-4, nu_post=1e-2, wmin=0.0, wmax=1.0):
		'''
		Instantiates a :code:`Connection` object with sparse weights.

		Inputs:
			| :code:`source` (:code:`nodes`.Nodes): A layer of nodes from which the connection originates.
			| :code:`target` (:code:`nodes`.Nodes): A layer of nodes to which the connection connects.
			| :code:`w` (:code:`torch`.FloatTensor or torch.cuda.FloatTensor): Effective strengths of synapses.
			| :code:`sparsity` (:code:`float`): Fraction of sparse connections to use.
			| :code:`update_rule` (:code:`function`): Modifies connection parameters according to some rule.
			| :code:`nu` (:code:`float`): Learning rate for both pre- and post-synaptic events.
			| :code:`nu_pre` (:code:`float`): Learning rate for pre-synaptic events.
			| :code:`nu_post` (:code:`float`): Learning rate for post-synpatic events.
			| :code:`wmin` (:code:`float`): The minimum value on the connection weights.
			| :code:`wmax` (:code:`float`): The maximum value on the connection weights.
		'''
		self.source = source
		self.target = target
		self.sparsity = sparsity
		self.nu = nu
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		self.wmin = wmin
		self.wmax = wmax

		if update_rule is None:
			self.update_rule = no_update
		else:
			if update_rule is m_stdp or update_rule is m_stdp_et:
				self.e_trace = 0
				self.tc_e_trace = 0.04
				self.p_plus = 0
				self.tc_plus = 0.05
				self.p_minus = 0
				self.tc_minus = 0.05
				
			self.update_rule = update_rule
		
		if w is None:
			i = torch.bernoulli(1 - self.sparsity * torch.ones(*source.shape, *target.shape))
			v = self.wmin + (self.wmax - self.wmin) * torch.rand(*source.shape, *target.shape)[i.byte()]
			self.w = torch.sparse.FloatTensor(i.nonzero().t(), v)
		else:
			self.w = w
			self.w[self.w < self.wmin] = self.wmin
			self.w[self.w > self.wmax] = self.wmax		