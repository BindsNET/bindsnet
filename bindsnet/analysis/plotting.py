import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.ion()

def plot_input(image, inpt, ims=None, figsize=(12, 6)):
	if not ims:
		fig, axes = plt.subplots(1, 2, figsize=figsize)
		ims = axes[0].imshow(image, cmap='binary'), axes[1].imshow(inpt, cmap='binary')
		
		axes[0].set_title('Current image')
		axes[1].set_title('Poisson spiking representation')
		axes[1].set_xlabel('Simulation time'); axes[1].set_ylabel('Neuron index')
		
		for ax in axes:
			ax.set_xticks(()); ax.set_yticks(())
		
		fig.tight_layout()
	else:
		ims[0].set_data(image)
		ims[1].set_data(inpt)

	return ims


def plot_spikes(data, ims=None, axes=None, time=None, figsize=(12, 7)):
	'''
	Plot spikes for any group of neurons.

	Inputs:
		data (dict): Contains spiking data for groups of neurons of interest.
		ims (list of matplotlib.figure.Figure): Figures
			to plot on. Otherwise, new figures are created.
		time (tuple): Plot spiking activity of neurons between the given range
			of time. Default is the entire simulation time. For example, time = 
			(40, 80) will plot spiking activity of neurons from 40 ms to 80 ms.
		figsize (tuple): Figure size.
	'''
	n_subplots = len(data.keys())
    
	# Confirm only 2 values for time were given
	if time is not None: 
		assert(len(time) == 2)
		assert(time[0] < time[1])

	else: # Set it for entire duration
		for key in data.keys():
			time = (0, data[key].shape[1])
			break
	
	if not ims:
		fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
		ims = []
		
		if n_subplots == 1: # Plotting only one image
			for key in data.keys():
				ims.append(axes.imshow(data[key][:, time[0]:time[1]], cmap='binary'))
				plt.title('%s spikes from t = %1.2f ms to %1.2f ms' % (key, time[0], time[1]))
				plt.xlabel('Time (ms)'); plt.ylabel('Neuron index')

		else: # Plot each layer at a time
			for i, datum in enumerate(data.items()):
				ims.append(axes[i].imshow(datum[1][:, time[0]:time[1]], cmap='binary'))
				axes[i].set_title('%s spikes from t = %1.2f ms to %1.2f ms' % (datum[0], time[0], time[1]))

		plt.setp(axes, xticks=[], yticks=[], xlabel='Simulation time', ylabel='Neuron index')
		
		for ax in axes:
			ax.set_aspect('auto')
		
		plt.tight_layout()
           
	else: # Plotting figure given
		assert(len(ims) == n_subplots)
		for i, datum in enumerate(data.items()):
			if time is None:
				ims[i].set_data(datum[1])
				axes[i].set_title('%s spikes from t = %1.2f ms to %1.2f ms' % (datum[0], time[0], time[1]))
			else: # Plot for given time
				ims[i].set_data(datum[1][time[0]:time[1]])
				axes[i].set_title('%s spikes from t = %1.2f ms to %1.2f ms' % (datum[0], time[0], time[1]))
	
	return ims, axes
        

def plot_weights(weights, wmax=1, im=None, figsize=(6, 6)):
	if not im:
		fig, ax = plt.subplots(figsize=figsize)
		
		im = ax.imshow(weights, cmap='hot_r', vmin=0, vmax=wmax)
		div = make_axes_locatable(ax)
		cax = div.append_axes("right", size="5%", pad=0.05)
		
		ax.set_xticks(()); ax.set_yticks(())
		
		plt.colorbar(im, cax=cax)
		fig.tight_layout()
	else:
		im.set_data(weights)

	return im


def plot_assignments(assignments, im=None, figsize=(6, 6)):
	if not im:
		fig, ax = plt.subplots(figsize=figsize)

		color = plt.get_cmap('RdBu', 11)
		im = ax.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
		div = make_axes_locatable(ax)
		cax = div.append_axes("right", size="5%", pad=0.05)

		plt.colorbar(im, cax=cax, ticks=np.arange(-1, 10))
		fig.tight_layout()
	else:
		im.set_data(assignments)

	return im


def plot_performance(performances, ax=None, figsize=(6, 6)):
	if not ax:
		_, ax = plt.subplots(figsize=figsize)
	else:
		ax.clear()

	for scheme in performances:
		ax.plot(range(len(performances[scheme])), [100 * p for p in performances[scheme]], label=scheme)

	ax.set_ylim([0, 100])
	ax.set_title('Estimated classification accuracy')
	ax.set_xlabel('No. of examples')
	ax.set_ylabel('Accuracy')
	ax.legend()

	return ax


def plot_voltages(exc, inh, axes=None, figsize=(8, 8)):
	if axes is None:
		_, axes = plt.subplots(2, 1, figsize=figsize)
		axes[0].set_title('Excitatory voltages')
		axes[1].set_title('Inhibitory voltages')
	
	axes[0].clear(); axes[1].clear()
	axes[0].plot(exc), axes[1].plot(inh)

	return axes