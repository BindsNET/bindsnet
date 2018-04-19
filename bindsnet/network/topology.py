import torch

from ..learning import no_update


class Connection:
	'''
	Specifies constant synapses between two populations of neurons.
	'''
	def __init__(self, source, target, w=None, update_rule=None, nu=1e-2,
							nu_pre=1e-4, nu_post=1e-2, wmin=0.0, wmax=1.0):
		'''
		Instantiates a Connections object, used to connect two layers of nodes.

		Inputs:
			source (nodes.Nodes): A layer of nodes from which the connection originates.
			target (nodes.Nodes): A layer of nodes to which the connection connects.
			w (torch.FloatTensor or torch.cuda.FloatTensor): Effective strengths of synaptics.
			update_rule (function): Modifies connection parameters according to some rule.
			nu (float): Learning rate for both pre- and post-synaptic events.
			nu_pre (float): Learning rate for pre-synaptic events.
			nu_post (float): Learning rate for post-synpatic events.
			wmin (float): The minimum value on the connection weights.
			wmax (float): The maximum value on the connection weights.
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
			self.update_rule = update_rule

		if w is None:
			self.w = torch.rand(*source.shape, *target.shape)
		else:
			self.w = w
		
		self.w = torch.clamp(self.w, self.wmin, self.wmax)

	def get_weights(self):
		'''
		Retrieve weight matrix of the connection.
		
		Returns:
			(torch.Tensor or torch.cuda.Tensor): 
				Weight matrix of the connection.
		'''
		return self.w

	def set_weights(self, w):
		'''
		Set weight matrix of the connection.
		
		Inputs:
			w (torch.Tensor or torch.cuda.Tensor):
				Weight matrix to set to connection.
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
			norm (float): Desired sum of weights.
		'''
		self.w = self.w.view(self.source.n, self.target.n)
		self.w *= norm / self.w.sum(0).view(1, -1)
		self.w = self.w.view(*self.source.shape, *self.target.shape)
		
	def _reset(self):
		'''
		Contains resetting logic for the connection.
		'''
		pass