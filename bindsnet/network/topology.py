import torch
import numpy as np
import torch.nn.functional as F

from ..learning             import *
from ..network.nodes        import Nodes
from torch.nn.modules.utils import _pair

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
		
		self.update_rule = kwargs.get('update_rule', None)
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
	
	def compute(self, s):
		'''
		Compute pre-activations given spikes using layer weights.
		
		Inputs:
		
			| :code:`s` (:code:`torch.Tensor`): Incoming spikes.
		'''
		s = s.float().view(-1)
		w = self.w.view(self.source.n, self.target.n)
		a = s @ w
		return a.view(*self.target.shape)

	def update(self, **kwargs):
		'''
		Compute connection's update rule.
		'''
		reward = kwargs.get('reward', None)
		
		if self.update_rule is not None:
			self.update_rule(self, reward=reward)
	
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


class Conv2dConnection:
	'''
	Specifies convolutional synapses between one or two populations of neurons.
	'''
	def __init__(self, source, target, kernel_size, stride=1, padding=0,
				 dilation=1, nu=1e-2, nu_pre=1e-4, nu_post=1e-2, **kwargs):
		'''
		Instantiates a :code:`Conv2dConnection` object.

		Inputs:
		
			| :code:`source` (:code:`nodes`.Nodes): A layer of nodes from which the connection originates.
			| :code:`target` (:code:`nodes`.Nodes): A layer of nodes to which the connection connects.
			| :code:`kernel_size` (:code:tuple(`int`)): Size of convolutional kernels; single number or tuple (height, width).
			| :code:`stride` (:code:tuple(`int`)): Stride for convolution; a single number or tuple (height, width).
			| :code:`padding` (:code:tuple(`int`)): Padding for convolution; a single number or tuple (height, width).
			| :code:`dilation` (:code:tuple(`int`)): Dilation for convolution; a single number or tuple (height, width).
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
		self.kernel_size = _pair(kernel_size)
		self.stride = _pair(stride)
		self.padding = _pair(padding)
		self.dilation = _pair(dilation)
		self.nu = nu
		self.nu_pre = nu_pre
		self.nu_post = nu_post
		
		assert isinstance(source, Nodes), 'Source is not a Nodes object'
		assert isinstance(target, Nodes), 'Target is not a Nodes object'
		assert len(source.shape) == 4, 'Source dimensionality must be (minibatch, in_channels, input_height, input_width)'
		assert len(target.shape) == 4, 'Target dimensionality must be (minibatch, out_channels, \
										 (input_height - filter_height + 2 * padding_height) / stride_height + 1), \
										 (input_width - filter_width + 2 * padding_width) / stride_width + 1'
		
		self.in_channels = source.shape[1]
		self.out_channels = target.shape[1]
		
		self.update_rule = kwargs.get('update_rule', None)
		self.w = kwargs.get('w', torch.rand(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
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
	
	def compute(self, s):
		'''
		Compute convolutional pre-activations given spikes using layer weights.
		
		Inputs:
		
			| :code:`s` (:code:`torch.Tensor`): Incoming spikes.
		'''
		return F.conv2d(s.float(), self.w, stride=self.stride, padding=self.padding, dilation=self.dilation)

	def update(self, **kwargs):
		'''
		Compute connection's update rule.
		'''
		if self.update_rule is not None:
			reward = kwargs.get('reward', None)
			self.update_rule(self, reward=reward)
	
	def normalize(self):
		'''
		Normalize weights along the first axis according to total weight per target neuron.
		'''
		if self.norm is not None:
			shape = self.w.size()
			self.w = self.w.view(self.w.size(0), self.w.size(2) * self.w.size(3))
			self.w *= self.norm / self.w.sum(0).view(1, -1)
			self.w = self.w.view(*shape)
		
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
		self.update_rule = kwargs.get('update_rule', None)
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
