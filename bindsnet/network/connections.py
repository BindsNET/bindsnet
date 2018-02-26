import torch


def post_pre(conn, nu_pre=1e-4, nu_post=1e-2):
	'''
	Simple STDP rule involving both pre- and post-synaptic spiking activity.
	
	Inputs:
		nu_pre (float): Learning rate for pre-synaptic events.
		nu_post (float): Learning rate for post-synpatic events.
	'''
	# Post-synaptic.
	conn.w += nu_post * (conn.source.x.view(conn.source.n,
			1) * conn.target.s.float().view(1, conn.target.n))
	# Pre-synaptic.
	conn.w -= nu_pre * (conn.source.s.float().view(conn.source.n,
							1) * conn.target.x.view(1, conn.target.n))

	# Bound weights.
	conn.w = torch.clamp(conn.w, conn.min, conn.wmax)


class Connection:
	'''
	Specifies constant synapses between two populations of neurons.
	'''
	def __init__(self, source, target, update_rule=None, w=None, wmin=0.0, wmax=1.0):
		'''
		Instantiates a Connections object, used to connect two layers of nodes.

		Inputs:
			source (nodes.Nodes): A layer of nodes from which the connection originates.
			target (nodes.Nodes): A layer of nodes to which the connection connects.
			update_rule (function): Modifies connection parameters according to some rule.
			w (torch.FloatTensor or torch.cuda.FloatTensor): Effective strengths of synaptics.
			wmin (float): The minimum value on the connection weights.
			wmax (float): The maximum value on the connection weights.
		'''
		self.source = source
		self.target = target
		self.wmin = wmin
		self.wmax = wmax

		if update_rule is None:
			self.update_rule = lambda : None
		else:
			self.update_rule = update_rule

		if w is None:
			self.w = torch.rand(source.n, target.n)
		else:
			self.w = w

		torch.clamp(self.w, self.wmin, self.wmax)

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

	def update(self):
		'''
		Run connection's given update rule.
		'''
		self.update_rule()
	
	def normalize(self, norm=78.0):
		'''
		Normalize weights along the first axis
		according to some desired cumulative value.
		
		Inputs:
			norm (float): Desired sum of weights.
		'''
		conn.w *= norm / conn.w.sum(0).view(1, -1)
		
	def reset(self):
		'''
		Contains resetting logic for the connection.
		'''
		pass