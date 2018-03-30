import os
import sys
import gym
import torch

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))

from encoding import get_bernoulli


class SpaceInvaders:
	'''
	A wrapper around the SpaceInvaders-v0 OpenAI gym environment.
	'''
	def __init__(self):
		self.env = gym.make('SpaceInvaders-v0')

	def step(self, a):
		obs, reward, done, info = self.env.step(a)
		obs = torch.from_numpy(obs).view(1, -1).float()
		obs = get_bernoulli(obs)

		return next(obs), reward, done, info

	def reset(self):
		obs = self.env.reset()
		obs = torch.from_numpy(obs).view(1, -1).float()
		obs = get_bernoulli(obs)

		return next(obs)

	def render(self):
		return self.env.render()

	def close(self):
		self.env.close()