import sys
import torch
import numpy as np

from plotting import plot_input
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.gridspec as gridspec


def spike_trains_main(data=None, n_ex=None, top_k=None, indices=None):
#	fig = plt.figure(figsize=(10, 8))
#	outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
#	
#	inner = gridspec.GridSpecFromSubplotSpec(top_k, 1,
#                    subplot_spec=outer[1], wspace=0.1, hspace=0.1)
	
#	ax = plt.Subplot(fig, outer[0])
	

	for j in range(top_k):
#		ax = plt.Subplot(fig, inner[j])
		plot_spike_trains_examples(data, n_ex=n_ex)
	
	
	
def weights_for_example(weights=None, subsample=10):
	pass

def plot_spike_trains_for_example(spikes=None, n_ex=None, top_k=None, indices=None):
	'''
	spikes(torch.tensor (N_neurons, time)): Spiking train data for a population of 
																neurons for one example
	n_ex(int): Allows user to pick which example to plot spikes for
	top_k(int): Plot k neurons that spiked the most for n_ex example
	indices(list(int): Plot specific neurons' spiking activity instead of top_k
								Meant to replace top_k. 
								
	If both top_k and indices are left as default values (None), plot will include
	all neurons
	
	'''
	
	
	if top_k is None:
		assert (indices is not None)
	else:
		assert (indices is None)
	
	
	# Figure out top_k locations
	#fig, axes = plt.subplot(top_k, 1)
	plt.figure()
	top_k_loc = np.argsort(np.sum(spikes[n_ex,:,:], axis=1), axis=0)[::-1]
	
	spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, top_k_loc[0:top_k], :]]
	plt.eventplot(spike_per_neuron)
	plt.show()
	
def main():
	data = np.random.binomial(1, 0.25, size=[5, 20, 100])
	plot_spike_trains_examples(data, 1, 5)
if __name__ == '__main__':
	main()
		
	
