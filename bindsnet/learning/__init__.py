import torch

def no_update(conn, **kwargs):
	'''
	No updates; weights are static.
	'''
	pass

def post_pre(conn, **kwargs):
	'''
	Simple STDP rule involving both pre- and post-synaptic spiking activity.
	'''
	# Post-synaptic.
	conn.w += conn.nu_post * conn.source.x.unsqueeze(-1) * conn.target.s.float().unsqueeze(0)
	# Pre-synaptic.
	conn.w -= conn.nu_pre * conn.source.s.float().unsqueeze(-1) * conn.target.x.unsqueeze(0)

	# Bound weights.
	conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)

def hebbian(conn, **kwargs):
	'''
	Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
	'''
	# Post-synaptic.
	conn.w += conn.nu_post * conn.source.x.unsqueeze(-1) * conn.target.s.float().unsqueeze(0)
	# Pre-synaptic.
	conn.w += conn.nu_pre * conn.source.s.float().unsqueeze(-1) * conn.target.x.unsqueeze(0)

	# Bound weights.
	conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)

def m_stdp(conn, **kwargs):
	'''
	Reward-modulated STDP. Adapted from
	https://florian.io/papers/2007_Florian_Modulated_STDP.pdf.
	'''
	# Get arguments to MSTDPET function.
	reward = kwargs['reward']
	a_plus = kwargs['a_plus']
	a_minus = kwargs['a_minus']
	
	# Get P^+ and P^- values (function of firing traces).
	p_plus = a_plus * conn.source.x.unsqueeze(-1)
	p_minus = a_minus * conn.target.x.unsqueeze(0)
	
	# Get pre- and post-synaptic spiking neurons.
	pre_fire = conn.source.s.float().unsqueeze(-1)
	post_fire = conn.target.s.float().unsqueeze(0)
	
	# Calculate point eligibility value.
	eligibility = p_plus * post_fire + pre_fire * p_minus
	
	# Compute weight update.
	conn.w += conn.nu * reward * eligibility
	
	# Bound weights.
	conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
	
def m_stdp_et(conn, **kwargs):
	'''
	Reward-modulated STDP with eligibility trace. Adapted from
	https://florian.io/papers/2007_Florian_Modulated_STDP.pdf.
	'''
	# Get arguments to MSTDPET function.
	reward = kwargs['reward']
	a_plus = kwargs['a_plus']
	a_minus = kwargs['a_minus']
	
	# Get P^+ and P^- values (function of firing traces).
	conn.p_plus = -(conn.tc_plus * conn.p_plus) + a_plus * conn.source.x.unsqueeze(-1)
	conn.p_minus = -(conn.tc_minus * conn.p_minus) + a_minus * conn.target.x.unsqueeze(0)
	
	# Get pre- and post-synaptic spiking neurons.
	pre_fire = conn.source.s.float().unsqueeze(-1)
	post_fire = conn.target.s.float().unsqueeze(0)
	
	# Calculate value of eligibility trace.
	conn.e_trace += -(conn.tc_e_trace * conn.e_trace) + (conn.p_plus * post_fire + pre_fire * conn.p_minus)
	
	# Compute weight update.
	conn.w += conn.nu * reward * conn.e_trace
	
	# Bound weights.
	conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
