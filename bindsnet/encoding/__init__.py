import torch
import numpy as np


def get_poisson_mixture(data, time, window):
	'''
	Generates mixture models of Poisson spike trains based on input intensity. 
	Each timeframe describes a Poisson spike train, which is aggregated to the actual 
	spike train from that timestep onwards
	Inputs must be non-negative. Spike inter-arrival times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.
    
    Inputs:
        data List of (torch.Tensor or torch.cuda.Tensor): Tensors of shape [n_samples, n_1,
            ..., n_k], with arbitrary sample dimensionality [n_1, ..., n_k].
        time (int): Length of Poisson spike train per input variable.
    
    Yields:
        (torch.Tensor or torch.cuda.Tensor): Tensors with shape [time, n_1, ..., n_k], with
            Poisson-distributed spikes parameterized by the data values.
	'''
	for audio in data:
		# For poisson, add minimum element to all
		audio_shifted = audio - torch.min(audio) # Linear shifting for now.. Can try exponential/polynomial
		s = get_poisson_mixture_for_example(audio_shifted, time, window)
		yield torch.Tensor(s).byte()

def get_poisson_mixture_for_example(data, time, window):
	'''
	Generates mixture models of Poisson spike trains based on input intensity. 
	Each timeframe describes a Poisson spike train, which is aggregated to the actual 
	spike train from that timestep onwards
	Inputs must be non-negative. Spike inter-arrival times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.
    
    Inputs:
        data (torch.Tensor or torch.cuda.Tensor): Tensors of shape [T, n_1,
            ..., n_k], with arbitrary sample dimensionality [n_1, ..., n_k]
            T = #frames in utterance
        time (int): Length of Poisson spike train per input variable.
    
    Returns:
        (torch.Tensor or torch.cuda.Tensor): Tensors with shape [time, n_1, ..., n_k], with
            Poisson-distributed spikes parameterized by the data values.
	'''
	if data.shape[0]>time:
		print("Warning: more frames than timesteps. Extra frames will be skipped. Frames = ", data.shape[0]," Timesteps = ", time)

	spikes = np.zeros([time, data.shape[1]]) # (time,40) for spokenMNIST with 40 dim log filter banks
	for i,frame in enumerate(data):
		# TODO is this needed?
		frame = np.copy(frame)
		if i>time-window:
			break
		# s = get_poisson_for_frame(frame, time-i)
		s = get_poisson_for_frame(frame, window) # For every frame, generate a small window of Poisson distributions
		spikes[i:i+window] += s # add them (like deconv) beginning from their temporal location
	spikes[spikes>1] = 1

	return spikes
		

def get_poisson_for_frame(datum, time):
	# Get i-th datum.
	shape, size = datum.shape, datum.size
	datum = datum.ravel()

	# Invert inputs (input intensity inversely
	# proportional to spike inter-arrival time).
	datum[datum != 0] = 1 / datum[datum != 0]

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

	return s
	# # Yield Poisson-distributed spike trains.
	# yield torch.Tensor(s).byte()


def get_poisson(data, time):
	'''
	Generates Poisson spike trains based on input intensity. Inputs must be
	non-negative. Spike inter-arrival times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.
    
    Inputs:
        data (torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_samples, n_1,
            ..., n_k], with arbitrary sample dimensionality [n_1, ..., n_k].
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


def get_tfs(data, time):
	'''
	Generates spike trains based on the Time to First Spike scheme. First Spike times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.

    Inputs:
        data (torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_samples, n_1,
            ..., n_k], with arbitrary sample dimensionality [n_1, ..., n_k].
        time (int): Length of Poisson spike train per input variable.

    Yields:
        (torch.Tensor or torch.cuda.Tensor): Tensors with shape [time, n_1, ..., n_k], with
            Poisson-distributed spikes parameterized by the data values.
	'''
	# n_samples = data.size(0)  # Number of samples
	# data = np.copy(data)
    #
	# for i in range(n_samples):
	# 	# Get i-th datum.
	# 	datum = data[i]
	# 	shape, size = datum.shape, datum.size
	# 	datum = datum.ravel()
    #
	# 	# Invert inputs (input intensity inversely
	# 	# proportional to spike inter-arrival time).
	# 	datum[datum != 0] = 1 / datum[datum != 0]
    #
	# 	# Make spike data from Poisson sampling.
	# 	s_times = np.random.poisson(datum, [time, size])
	# 	s_times = np.cumsum(s_times, axis=0)
	# 	s_times[s_times >= time] = 0
    #
	# 	# Create spike trains from spike times.
	# 	s = np.zeros([time, size])
	# 	for idx in range(time):
	# 		s[s_times[idx], np.arange(size)] = 1
    #
	# 	s[0, :] = 0
	# 	s = s.reshape([time, *shape])
    #
	# 	# Yield Poisson-distributed spike trains.
	# 	yield torch.Tensor(s).byte()
    #

def get_bernoulli_mixture(data, time, window=1):
	for audio in data:
		# For poisson, add minimum element to all
		audio_shifted = audio - torch.min(audio)  # Linear shifting for now.. Can try exponential/polynomial
		s = get_bernoulli_for_example(audio_shifted, time, window=window)
		yield torch.Tensor(s).byte()


def get_bernoulli_for_example(data, time, window=1):
	if data.shape[0] > time:
		print("Warning: more frames than timesteps. Extra frames will be skipped. Frames = ", data.shape[0],
			  " Timesteps = ", time)

	spikes = np.zeros([time, data.shape[1]])  # (time,40) for spokenMNIST with 40 dim log filter banks
	for i, frame in enumerate(data):
		# TODO is this needed?
		frame = np.copy(frame)
		if i > time-window:
			break
		# s = get_poisson_for_frame(frame, time-i)
		s = get_poisson_for_frame(frame, window)  # For every frame, generate a small window of Poisson distributions
		spikes[i:i+window] += s  # add them (like deconv) beginning from their temporal location
	spikes[spikes > 1] = 1 # unnecessary for Bernoulli trials with 0/1

	return spikes

def get_bernoulli_for_frame(datum, time, max_prob=1.0):
	'''
	Generates Bernoulli-distributed spike trains based on input intensity. Inputs must
	be non-negative. Spikes correspond to successful Bernoulli trials, with success
	probability equal to (normalized in [0, 1]) input value.

	Inputs:
		data (torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_samples,
			n_1, ..., n_k], with arbitrary sample dimensionality [n_1, ..., n_k].
		time (int): Length of Bernoulli spike train per input variable.
		max_prob (float): Maximum probability of spike per Bernoulli trial.
	'''
	shape, size = datum.shape, datum.size
	datum = datum.ravel()

	# Normalize inputs and rescale (spike probability
	# proportional to normalized intensity).
	datum /= datum.max()
	datum *= max_prob

	# Make spike data from Bernoulli sampling.
	s = np.random.binomial(1, datum, [time, size])
	s = s.reshape([time, *shape])

	# Yield Bernoulli-distributed spike trains.
	return s


def get_bernoulli(data, time, max_prob=1.0):
	'''
	Generates Bernoulli-distributed spike trains based on input intensity. Inputs must
	be non-negative. Spikes correspond to successful Bernoulli trials, with success
	probability equal to (normalized in [0, 1]) input value.

	Inputs:
		data (torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_samples,
			n_1, ..., n_k], with arbitrary sample dimensionality [n_1, ..., n_k].
		time (int): Length of Bernoulli spike train per input variable.
		max_prob (float): Maximum probability of spike per Bernoulli trial.
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
		s = np.random.binomial(1, datum, [time, size])
		s = s.reshape([time, *shape])

		# Yield Bernoulli-distributed spike trains.
		yield torch.Tensor(s).byte()
   