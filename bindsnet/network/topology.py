import torch
import numpy as np

from ..learning import *
from ..network.nodes import Nodes


class Connection:
	'''
	Specifies synapses between one or two populations of neurons.
	'''
	def __init__(self, source, target, nu=1e-2, nu_pre=1e-4, nu_post=1e-2, **kwargs):
		'''
		Instantiates a :code:`Connection` object.

		Inputs:
		
			| :code:`source` (:code:`nodes`.Nodes): A layer of nodes from which the connection originates.
			| :code:`target` (:code:`nodes`.Nodes): A layer of nodes to which the connection connects.
			| :code:`nu` (:code:`float`): Learning rate for both pre- and post-synaptic events.
			| :code:`nu_pre` (:code:`float`): Learning rate for pre-synaptic events.
			| :code:`nu_post` (:code:`float`): Learning rate for post-synpatic events.
			
			Kwargs:
			
				| :code:`update_rule` (:code:`function`): Modifies connection parameters according to some rule.
				| :code:`w` (:code:`torch.Tensor`): Effective strengths of synapses.
				| :code:`wmin` (:code:`float`): The minimum value on the connection weights.
				| :code:`wmax` (:code:`float`): The maximum value on the connection weights.
				| :code:`norm` (:code:`float`): Total weight per target neuron normalization.
		'''
		self.source = source
		self.target = target
		self.nu = nu
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		
		assert isinstance(source, Nodes), 'Source is not a Nodes object'
		assert isinstance(target, Nodes), 'Target is not a Nodes object'
		
		self.update_rule = kwargs.get('update_rule', no_update)
		self.w = kwargs.get('w', torch.rand(*source.shape, *target.shape))
		self.wmin = kwargs.get('wmin', float('-inf'))
		self.wmax = kwargs.get('wmax', float('inf'))
		self.norm = kwargs.get('norm', None)

		if self.update_rule is m_stdp or self.update_rule is m_stdp_et:
			self.e_trace = 0
			self.tc_e_trace = 0.04
			self.p_plus = 0
			self.tc_plus = 0.05
			self.p_minus = 0
			self.tc_minus = 0.05

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
	
	def normalize(self):
		'''
		Normalize weights along the first axis according to total weight per target neuron.
		'''
		if self.norm is not None:
			self.w = self.w.view(self.source.n, self.target.n)
			self.w *= self.norm / self.w.sum(0).view(1, -1)
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
	def __init__(self, source, target, nu=1e-2, nu_pre=1e-4, nu_post=1e-2, **kwargs):
		'''
		Instantiates a :code:`Connection` object with sparse weights.

		Inputs:
		
			| :code:`source` (:code:`nodes`.Nodes): A layer of nodes from which the connection originates.
			| :code:`target` (:code:`nodes`.Nodes): A layer of nodes to which the connection connects.
			| :code:`nu` (:code:`float`): Learning rate for both pre- and post-synaptic events.
			| :code:`nu_pre` (:code:`float`): Learning rate for pre-synaptic events.
			| :code:`nu_post` (:code:`float`): Learning rate for post-synpatic events.
			
			Kwargs:
			
				| :code:`w` (:code:`torch.Tensor`): Effective strengths of synapses.
				| :code:`sparsity` (:code:`float`): Fraction of sparse connections to use.
				| :code:`update_rule` (:code:`function`): Modifies connection parameters according to some rule.
				| :code:`wmin` (:code:`float`): The minimum value on the connection weights.
				| :code:`wmax` (:code:`float`): The maximum value on the connection weights.
				| :code:`norm` (:code:`float`): Total weight per target neuron normalization.
		'''
		self.source = source
		self.target = target
		self.nu = nu
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		
		assert isinstance(source, Nodes), 'Source is not a Nodes object'
		assert isinstance(target, Nodes), 'Target is not a Nodes object'
		
		self.w = kwargs.get('w', None)
		self.sparsity = kwargs.get('sparsity', None)
		self.update_rule = kwargs.get('update_rule', no_update)
		self.wmin = kwargs.get('wmin', float('-inf'))
		self.wmax = kwargs.get('wmax', float('inf'))
		self.norm = kwargs.get('norm', None)
		
		assert (self.w is not None and self.sparsity is None or
				self.w is None and self.sparsity is not None), \
				'Only one of "weights" or "sparsity" must be specified'
		
		if self.update_rule is m_stdp or self.update_rule is m_stdp_et:
			self.e_trace = 0
			self.tc_e_trace = 0.04
			self.p_plus = 0
			self.tc_plus = 0.05
			self.p_minus = 0
			self.tc_minus = 0.05
		
		if self.w is None and self.sparsity is not None:
			i = torch.bernoulli(1 - self.sparsity * torch.ones(*source.shape, *target.shape))
			v = self.wmin + (self.wmax - self.wmin) * torch.rand(*source.shape, *target.shape)[i.byte()]
			self.w = torch.sparse.FloatTensor(i.nonzero().t(), v)
		elif self.w is not None:
			assert self.w.is_sparse, 'Weight matrix is not sparse (see torch.sparse module)'
			
			self.w[self.w < self.wmin] = self.wmin
			self.w[self.w > self.wmax] = self.wmax		
