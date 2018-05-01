import torch
import numpy as np


def get_poisson(data, time):
	'''
	Generates Poisson spike trains based on input intensity. Inputs must be
	non-negative. Spike inter-arrival times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.
	
	Inputs:
		| :code:`data` (:code:`torch.Tensor`): Tensor of shape :code:`[n_samples, n_1, ..., n_k]`,
			with arbitrary data dimensionality :code:`[n_1, ..., n_k]`.
		| :code:`time` (:code:`int`): Length of Poisson spike train per input variable.

	Yields:
		| (:code:`torch.Tensor`): Tensor with shape :code:`[time, n_1, ..., n_k]` of
			Poisson-distributed spikes with inter-arrival times determines by the data.
	'''
	n_samples = data.size(0)  # Number of samples
	data = np.copy(data)
	
	for i in range(n_samples):
		# Get i-th datum.
		datum = data[i]
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
		
		# Yield Poisson-distributed spike trains.
		yield torch.Tensor(s).byte()


def get_bernoulli(data, time=None, max_prob=1.0):
	'''
	Generates Bernoulli-distributed spike trains based on input intensity. Inputs must
	be non-negative. Spikes correspond to successful Bernoulli trials, with success
	probability equal to (normalized in [0, 1]) input value.

	Inputs:
		| :code:`data` (:code:`torch.Tensor`): Tensor of shape :code:`[n_samples, n_1, ..., n_k]`,
			with arbitrary sample dimensionality :code:`[n_1, ..., n_k]`.
		| :code:`time` (:code:`int`): Length of Bernoulli spike train per input variable.
		| :code:`max_prob` (:code:`float`): Maximum probability of spike per Bernoulli trial.
	'''
	n_samples = data.size(0)  # Number of samples
	data = np.copy(data)

	for i in range(n_samples):
		# Get i-th datum.
		datum = data[i]
		shape, size = datum.shape, datum.size
		datum = datum.ravel()

		# Normalize inputs and rescale (spike probability
		# proportional to normalized intensity).
		datum /= datum.max()
		datum *= max_prob

		# Make spike data from Bernoulli sampling.
		if time is None:
			s = np.random.binomial(1, datum, [size])
			s = s.reshape([*shape])
		else:
			s = np.random.binomial(1, datum, [time, size])
			s = s.reshape([time, *shape])

		# Yield Bernoulli-distributed spike trains.
		yield torch.Tensor(s).byte()
   