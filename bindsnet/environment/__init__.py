import os
import sys
import gym
import torch
import numpy as np

from skimage.measure import block_reduce

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))

from encoding import get_bernoulli


class SpaceInvaders:
	'''
	A wrapper around the SpaceInvaders-v0 OpenAI gym environment.
	'''
	def __init__(self, max_prob=0.25):
		self.max_prob = 0.25
		self.env = gym.make('SpaceInvaders-v0')

	def step(self, a):
		obs, reward, done, info = self.env.step(a)
		obs = block_reduce(obs, block_size=(3, 3, 3), func=np.mean)
		obs = torch.from_numpy(obs).view(1, -1).float()
		obs = get_bernoulli(obs, max_prob=self.max_prob)
		
		return next(obs).view(1, -1), reward, done, info

	def reset(self):
		obs = self.env.reset()
		obs = block_reduce(obs, block_size=(3, 3, 3), func=np.mean)
		obs = torch.from_numpy(obs).view(1, -1).float()
		obs = get_bernoulli(obs, max_prob=self.max_prob)

		return next(obs)

	def render(self):
		return self.env.render()

	def close(self):
		self.env.close()