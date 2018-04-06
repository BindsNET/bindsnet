import sys
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.animation as animation
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
	
ani = 0
def create_movie(weights):
	fig = plt.figure()
	im = plt.imshow(weights[:, :, 0], cmap='hot_r', animated=True, vmin=0, vmax=1)
	plt.axis('off'); plt.colorbar(im)
		
	def update(j):
		im.set_data(weights[:, :, j])
		return [im]
	
	global ani
	ani = animation.FuncAnimation(fig, update, frames=weights.shape[-1], interval=1000, blit=True)
	plt.show()
		
	
def plot_weights_movie(weights):
	n_examples = weights.shape[0]
	ws = []
	for i in range(n_examples):
		ws.append(get_weights_for_example(weights[i]))
	ws = np.concatenate(ws, axis=2)
	create_movie(ws)
	
	
def get_weights_for_example(w, sample_every=1):
	return w[:, :, range(0, w.shape[2], sample_every)]

def plot_spike_trains_for_example(spikes=None, n_ex=None, top_k=None, indices=None):
	'''
	spikes(torch.tensor (N_neurons, time)): Spiking train data for a population of 
																neurons for one example
	n_ex(int): Allows user to pick which example to plot spikes for
	top_k(int): Plot k neurons that spiked the most for n_ex example
	indices(list(int): Plot specific neurons' spiking activity instead of top_k
								Meant to replace top_k. 
								
	If both top_k and indices are left as default values (None), plot will include
	all neurons.
	
	'''

	# Check that either both are None or only one of the parameters is
	if top_k is None and indices is None:
		top_k = spikes.shape[0]
	elif top_k is None:
		assert (indices is not None)
	elif indices is None:
		assert (top_k is not None)
	
	
	# Figure out top_k locations
	#fig, axes = plt.subplot(top_k, 1)
	plt.figure()
	top_k_loc = np.argsort(np.sum(spikes[n_ex,:,:], axis=1), axis=0)[::-1]
	
	spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, top_k_loc[0:top_k], :]]
	plt.eventplot(spike_per_neuron)
	plt.show()
	

def main():
	#data = np.random.binomial(1, 0.25, size=[5, 20, 100])
	#plot_spike_trains_examples(data, 1, 5)
	np.random.seed(0)
	weights = np.random.random(size=(5, 4, 4, 20))
	plot_weights_movie(weights)
	
	
if __name__ == '__main__':
	main()
		
	
