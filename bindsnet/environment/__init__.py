import os
import sys
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from ..datasets.preprocess import *
from ..encoding            import *
from ..datasets            import *


class Games(ABC):
	'''
	Abstract base class for OpenAI gym environments.
	'''
	def __init__(self):
		'''
		Abstract constructor for the Games class.
		'''
		super().__init__()

	@abstractmethod
	def preprocess(self):
		'''
		Pre-processing steps for every observation.
		'''
		pass

	def close(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`close()` function.
		'''
		self.env.close()


class DatasetEnvironment(ABC):
	'''
	Abstract base class for dataset environment wrappers.
	'''
	def __init__(self):
		'''
		Abstract constructor for the DatasetEnvironment class.
		'''
		super().__init__()
	
	@abstractmethod
	def preprocess(self):
		'''
		Pre-processing steps for every observation.
		'''
		pass

	@abstractmethod
	def close(self):
		'''
		Dummy function mimicking OpenAI Gym :code:`close()` function.
		'''
		pass


class MNISTEnv(DatasetEnvironment):
	'''
	A wrapper around the :code:`MNIST` dataset object to pass to the :code:`Pipeline` object.
	'''
	def __init__(self, train=True, time=350, intensity=0.25, data_path=os.path.join('..', '..', 'data', 'MNIST')):
		'''
		Initializes the environment wrapper around the MNIST dataset.
		
		Inputs:
		
			| :code:`train` (:code:`bool`): Whether to use train or test dataset.
			| :code:`time` (:code:`time`): Length of spike train per example.
			| :code:`intensity` (:code:`intensity`): Raw data is multiplied by this value.
		'''
		super(MNIST).__init__()
		
		self.train = train
		self.time = time
		self.intensity = intensity
		
		if train:
			self.data, self.labels = MNIST(data_path).get_train()
			self.label_loader = iter(self.labels)
		else:
			self.data, self.labels = MNIST(data_path).get_test()
			self.label_loader = iter(self.labels)
		
		self.env = iter(self.data)
	
	def step(self, a=None):
		'''
		Dummy function for OpenAI Gym environment's :code:`step()` function.

		Returns:

			| :code:`obs` (:code:`torch.Tensor`): Observation from the environment (spike train-encoded MNIST digit).
			| :code:`reward` (:code:`float`): Fixed to :code:`0`.
			| :code:`done` (:code:`bool`): Fixed to :code:`False`.
			| :code:`info` (:code:`dict`): Contains label of MNIST digit.
		'''
		try:
			# Attempt to fetch the next observation.
			self.obs = next(self.env)
		except StopIteration:
			# If out of samples, reload data and label generators.
			self.env = iter(data)
			self.label_loader = iter(self.labels)
			self.obs = next(self.env)
		
		# Preprocess observation.
		self.preprocess()
		
		# Info dictionary contains label of MNIST digit.
		info = {'label' : next(self.label_loader)}
		
		return self.obs, 0, False, info
	
	def reset(self):
		'''
		Dummy function for OpenAI Gym environment's :code:`reset()` function.
		'''
		# Reload data and label generators.
		self.env = iter(self.data)
		self.label_loader = iter(self.labels)
	
	def render(self):
		'''
		Dummy function for OpenAI Gym environment's :code:`render()` function.
		'''
		pass

	def close(self):
		'''
		Dummy function for OpenAI Gym environment's :code:`close()` function.
		'''
		pass
	
	def preprocess(self):
		'''
		Preprocessing step for a state specific to Space Invaders.

		Inputs:

			| (:code:`numpy.array`): Observation from the environment.

		Returns:

			| (:code:`torch.Tensor`): Pre-processed observation.
		'''
		self.obs = self.obs.view(784)
		self.obs *= self.intensity


class SpaceInvaders(Games):
	'''
	A wrapper around the :code:`SpaceInvaders-v0` OpenAI gym environment.
	'''
	def __init__(self, max_prob=0.25, diffs=True):
		'''
		Initializes the OpenAI Gym Space Invaders environment wrapper.

		Inputs:

			| :code:`max_prob` (:code:`float`): Specifies the maximum Bernoulli spiking probability.
			| :code:`diffs` (:code:`bool`): Whether to record previous frame and take differences.
		'''
		super().__init__()

		self.max_prob = max_prob
		self.env = gym.make('SpaceInvaders-v0')
		self.diffs = diffs
		self.action_space = self.env.action_space

	def step(self, a):
		'''
		Wrapper around the OpenAI Gym environment :code:`step()` function.

		Inputs:

			| :code:`a` (:code:`int`): Action to take in Space Invaders environment.

		Returns:

			| :code:`obs` (:code:`torch.Tensor`): Observation from the environment.
			| :code:`reward` (:code:`float`): Reward signal from the environment.
			| :code:`done` (:code:`bool`): Indicates whether the simulation has finished.
			| :code:`info` (:code:`dict`): Current information about the environment.
		'''
		# No action selected corresponds to no-op.
		if a is None:
			a = 0
		
		# Call gym's environment step function.
		self.obs, self.reward, done, info = self.env.step(a)
		self.preprocess()

		# Return converted observations and other information.
		return self.obs, self.reward, done, info

	def reset(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`reset()` function.

		Returns:

			| :code:`obs` (:code:`torch.Tensor`): Observation from the environment.
		'''
		# Call gym's environment reset function.
		self.obs = self.env.reset()
		self.preprocess()
		
		return(self.obs)

	def render(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`render()` function.
		'''
		self.env.render()

	def close(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`close()` function.
		'''
		self.env.close()

	def preprocess(self):
		'''
		Preprocessing step for an observation from the Space Invaders environment.
		'''
		self.obs = subsample(gray_scale(self.obs), 84, 110)
		self.obs = self.obs[26:104, :]
		self.obs = binary_image(self.obs)
		self.obs = np.reshape(self.obs, (78, 84, 1))
		self.obs = torch.from_numpy(self.obs).view(1, -1).float()
		

class CartPole(Games):
	'''
	A wrapper around the :code:`CartPole-v0` OpenAI gym environment.
	'''
	def __init__(self, max_prob=0.5):
		'''
		Initializes the OpenAI Gym Space Invaders environment wrapper.

		Inputs:

			| :code:`max_prob` (:code:`float`): Specifies the maximum Bernoulli trial spiking probability.
		'''
		super().__init__()

		self.max_prob = max_prob
		self.env = gym.make('CartPole-v0')

	def step(self, a):
		'''
		Wrapper around the OpenAI Gym environment :code:`step()` function.

		Inputs:

			| :code:`a` (:code:`int`): Action to take in Space Invaders environment.

		Returns:

			| (:code:`torch.Tensor`): Observation from the environment.
			| (:code:`float`): Reward signal from the environment.
			| (:code:`bool`): Indicates whether the simulation has finished.
			| (:code:`dict`): Current information about the environment.
		'''
		# Call gym's environment step function.
		obs, reward, done, info = self.env.step(a)

		# Encoding into positive values.
		obs = np.array([obs[0] + 2.4,
						-min(obs[1], 0),
						max(obs[1], 0),
						obs[2] + 41.8,
						-min(obs[3], 0),
						max(obs[3], 0)])

		# Convert to torch.Tensor, and then to Bernoulli-distributed spikes.
		obs = torch.from_numpy(obs).view(1, -1).float()

		# Return converted observations and other information.
		return obs, reward, done, info

	def reset(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`reset()` function.

		Returns:

			| (:code:`torch.Tensor`): Observation from the environment.
		'''
		# Call gym's environment reset function.
		obs = self.env.reset()

		# Encoding into positive values.
		obs = np.array([obs[0] + 2.4,
						-min(obs[1], 0),
						max(obs[1], 0),
						obs[2] + 41.8,
						-min(obs[3], 0),
						max(obs[3], 0)])

		# Convert to torch.Tensor, and
		# convert to Bernoulli-distributed spikes.
		obs = torch.from_numpy(obs).view(1, -1).float()

		# Return converted observations.
		return obs

	def preprocess(self):
		pass

	def render(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`render()` function.
		'''
		self.env.render()

	def close(self):
		'''
		Wrapper around the OpenAI Gym environment :code:`close()` function.
		'''
		self.env.close()
