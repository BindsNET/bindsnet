from bindsnet.encoding import *

import torch

class TestEncodings:
	'''
	Tests all stable encoding functions and generators.
	'''
	def test_bernoulli(self):
		print(); print('*** test_bernoulli ***'); print()
		
		for n in [1, 10, 100, 1000]:  # number of nodes in layer
			for t in [1, 10, 100, 1000]:  # number of timesteps
				for m in [0.01, 0.1, 1.0]:  # maximum spiking probability
					datum = torch.empty(n).uniform_(0, m)
					spikes = bernoulli(datum, time=t, max_prob=m)

					assert spikes.size() == torch.Size((t, n))

					print('No. nodes: %d, no. timesteps: %d, max. firing prob.: %.2f, activity ratio: %.4f' % \
						  (n, t, m, (datum.sum() * t) / spikes.float().sum()))
		
	def test_bernoulli_loader(self):
		print(); print('*** test_bernoulli_loader ***'); print()
		
		for s in [1, 10, 100]:  # number of data samples
			for n in [1, 10, 100]:  # number of nodes in layer
				for m in [0.01, 0.1, 1.0]:  # maximum spiking probability
					for t in [1, 10, 100]:  # number of timesteps
						data = torch.empty(s, n).uniform_(0, 1)
						spike_loader = bernoulli_loader(data, time=t, max_prob=m)

						for i, spikes in enumerate(spike_loader):
							assert spikes.size() == torch.Size((t, n))

							print('No. samples: %d, no. nodes: %d, no. timesteps: %d, max. firing prob.: %.2f, activity ratio: %.4f' % \
							  (s, n, t, m, (data[i].sum() * t) / spikes.float().sum()))

	
	def test_poisson(self):
		print(); print('*** test_poisson ***'); print()
		
		for n in [1, 10, 100, 1000]:  # number of nodes in layer
			for t in [1, 10, 100, 1000]:  # number of timesteps
				datum = torch.empty(n).uniform_(0, 1)
				spikes = poisson(datum, time=t)
				
				assert spikes.size() == torch.Size((t, n))
		
	def test_poisson_loader(self):
		print(); print('*** test_poisson_loader ***'); print()
		
		for s in [1, 10, 100]:  # number of data samples
			for n in [1, 10, 100]:  # number of nodes in layer
				for t in [1, 10, 100]:  # number of timesteps
					data = torch.empty(s, n).uniform_(0, 1)
					spike_loader = poisson_loader(data, time=t)
					
					for spikes in spike_loader:
						assert spikes.size() == torch.Size((t, n))
					