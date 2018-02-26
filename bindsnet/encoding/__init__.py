import torch
import numpy as np


def get_poisson(data, time):
	'''
	Generates Poisson spike trains based on input intensity. Inputs must be
	non-negative. Spike inter-arrival times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.
    
    Inputs:
        data (torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_samples, n_1,
            ..., n_k], with arbitrarily dimensionality [n_1, ..., n_k].
        time (int): Length of Poisson spike train per input variable.
    
    Yields:
        (torch.Tensor or torch.cuda.Tensor): Tensors with shape [time, n_1, ..., n_k], with
            Poisson-distributed spikes parameterized by the data values.
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
		datum[datum != 0] = 1 / datum[datum != 0]

		# Make the spike data.
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
		yield torch.Tensor(s)