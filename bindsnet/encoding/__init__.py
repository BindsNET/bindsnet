import torch
import numpy as np


def bernoulli(datum, time=None, max_prob=1.0):
	'''
	Generates Bernoulli-distributed spike trains based on input intensity. Inputs must be non-negative. Spikes correspond to successful Bernoulli trials, with success probability equal to (normalized in [0, 1]) input value.

	Inputs:
		| :code:`datum` (:code:`torch.Tensor`): Tensor of shape :code:`[n_1, ..., n_k]`.
		| :code:`time` (:code:`int`): Length of Bernoulli spike train per input variable.
		| :code:`max_prob` (:code:`float`): Maximum probability of spike per Bernoulli trial.
	
	Returns:
		| (:code:`torch.Tensor`): Tensor of shape :code:`[time, n_1, ..., n_k]` of Bernoulli-distributed spikes.
	'''
	datum = np.copy(datum)
	shape, size = datum.shape, datum.size
	datum = datum.ravel()

	# Normalize inputs and rescale (spike probability
	# proportional to normalized intensity).
	if datum.max() > 1.0:
		datum /= datum.max()
	datum *= max_prob

	# Make spike data from Bernoulli sampling.
	if time is None:
		s = np.random.binomial(1, datum, [size])
		s = s.reshape([*shape])
	else:
		s = np.random.binomial(1, datum, [time, size])
		s = s.reshape([time, *shape])
	
	return torch.Tensor(s).byte()

def bernoulli_loader(data, time=None, max_prob=1.0):
	'''
	Lazily invokes :code:`bindsnet.encoding.bernoulli` to iteratively encode a sequence of data.

	Inputs:
		| :code:`data` (:code:`torch.Tensor`): Tensor of shape :code:`[n_samples, n_1, ..., n_k]`.
		| :code:`time` (:code:`int`): Length of Bernoulli spike train per input variable.
		| :code:`max_prob` (:code:`float`): Maximum probability of spike per Bernoulli trial.
	
	Yields:
		| (:code:`torch.Tensor`): Tensor of shape :code:`[time, n_1, ..., n_k]` of Bernoulli-distributed spikes.
	'''
	for i in range(data.size(0)):
		yield bernoulli(data[i], time, max_prob)  # Encode datum as Bernoulli spike trains.

def poisson(datum, time):
	'''
	Generates Poisson-distributed spike trains based on input intensity. Inputs must be non-negative.

	Inputs:
		| :code:`datum` (:code:`torch.Tensor`): Tensor of shape :code:`[n_1, ..., n_k]`.
		| :code:`time` (:code:`int`): Length of Bernoulli spike train per input variable.
		| :code:`max_prob` (:code:`float`): Maximum probability of spike per Bernoulli trial.
	
	Returns:
		| (:code:`torch.Tensor`): Tensor of shape :code:`[time, n_1, ..., n_k]` of Poisson-distributed spikes.
	'''
	datum = np.copy(datum)
	shape, size = datum.shape, datum.size
	datum = datum.ravel()

	# Invert inputs (input intensity inversely
	# proportional to spike inter-arrival time).
	datum[datum != 0] = 1 / datum[datum != 0] * 1000

	# Make spike data from Poisson sampling.
	s_times = np.random.poisson(datum, [time, size])
	s_times = np.cumsum(s_times, axis=0)
	s_times[s_times >= time] = 0

	# Create spike trains from spike times.
	s = np.zeros([time, size])
	for idx in range(time):
		s[s_times[idx], np.arange(size)] = 1

	s[0, :] = 0
	s = s.reshape([time, *shape])
	
	return torch.Tensor(s).byte()
   
def poisson_loader(data, time):
	'''
	Lazily invokes :code:`bindsnet.encoding.poisson` to iteratively encode a sequence of data.
	
	Inputs:
		| :code:`data` (:code:`torch.Tensor`): Tensor of shape :code:`[n_samples, n_1, ..., n_k]`
		| :code:`time` (:code:`int`): Length of Poisson spike train per input variable.

	Yields:
		| (:code:`torch.Tensor`): Tensor of shape :code:`[time, n_1, ..., n_k]` of Poisson-distributed spikes.
	'''
	for i in range(data.size(0)):
		yield poisson(data[i], time)  # Encode datum as Poisson spike trains.
