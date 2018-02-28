import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.ion()

def plot_input(image, inpt, ims=None, figsize=(10, 5)):
	'''
	Plots a two-dimensional image and its corresponding spike-train representation.
	
	Inputs:
		image (torch.Tensor or torch.cuda.Tensor): A two-dimensional
			array of floating point values depicting an input image.
		inpt (torch.Tensor or torch.cuda.Tensor): A two-dimensional array of
			floating point values depicting an image's spike-train encoding.
		ims (list(matplotlib.image.AxesImage)): Used for re-drawing the input plots.
		figsize (tuple(int)): Horizontal, vertical figure size in inches.
	
	Returns:
		(list(matplotlib.image.AxesImage)): Used for re-drawing the input plots.
	'''
	if not ims:
		fig, axes = plt.subplots(1, 2, figsize=figsize)
		ims = axes[0].imshow(image, cmap='binary'), axes[1].imshow(inpt, cmap='binary')
		
		axes[0].set_title('Current image')
		axes[1].set_title('Poisson spiking representation')
		axes[1].set_xlabel('Simulation time'); axes[1].set_ylabel('Neuron index')
		axes[1].set_aspect('auto')
		
		for ax in axes:
			ax.set_xticks(()); ax.set_yticks(())
		
		fig.tight_layout()
	else:
		ims[0].set_data(image)
		ims[1].set_data(inpt)

	return ims


def plot_spikes(spikes, ims=None, axes=None, time=None, figsize=(12, 7)):
	'''
	Plot spikes for any group of neurons.

	Inputs:
		spikes (dict(torch.Tensor or torch.cuda.Tensor)): Contains
			spiking data for groups of neurons of interest.
		ims (list(matplotlib.image.AxesImage)): Used for re-drawing the spike plots.
		axes (list(matplotlib.axes.Axes)): Used for re-drawing the spike plots.
		time (tuple(int)): Plot spiking activity of neurons between the given range
			of time. Default is the entire simulation time. For example, time = 
			(40, 80) will plot spiking activity of neurons from 40 ms to 80 ms.
		figsize (tuple(int)): Horizontal, vertical figure size in inches.
	
	Returns:
		(list(matplotlib.image.AxesImage)): Used for re-drawing the spike plots.
		(list(matplotlib.axes.Axes)): Used for re-drawing the spike plots.
	'''
	n_subplots = len(spikes.keys())
    
	# Confirm only 2 values for time were given
	if time is not None: 
		assert(len(time) == 2)
		assert(time[0] < time[1])

	else: # Set it for entire duration
		for key in spikes.keys():
			time = (0, spikes[key].shape[1])
			break
	
	if not ims:
		fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
		ims = []
		
		if n_subplots == 1: # Plotting only one image
			for key in spikes.keys():
				ims.append(axes.imshow(spikes[key][:, time[0]:time[1]], cmap='binary'))
				plt.title('%s spikes from t = %1.2f ms to %1.2f ms' % (key, time[0], time[1]))
				plt.xlabel('Time (ms)'); plt.ylabel('Neuron index')

		else: # Plot each layer at a time
			for i, datum in enumerate(spikes.items()):
				ims.append(axes[i].imshow(datum[1][:, time[0]:time[1]], cmap='binary'))
				axes[i].set_title('%s spikes from t = %1.2f ms to %1.2f ms' % (datum[0], time[0], time[1]))

		plt.setp(axes, xticks=[], yticks=[], xlabel='Simulation time', ylabel='Neuron index')
		
		for ax in axes:
			ax.set_aspect('auto')
		
		plt.tight_layout()
           
	else: # Plotting figure given
		assert(len(ims) == n_subplots)
		for i, datum in enumerate(spikes.items()):
			if time is None:
				ims[i].set_data(datum[1])
				axes[i].set_title('%s spikes from t = %1.2f ms to %1.2f ms' % (datum[0], time[0], time[1]))
			else: # Plot for given time
				ims[i].set_data(datum[1][time[0]:time[1]])
				axes[i].set_title('%s spikes from t = %1.2f ms to %1.2f ms' % (datum[0], time[0], time[1]))
	
	return ims, axes
        

def plot_weights(weights, im=None, figsize=(6, 6)):
	'''
	Plot a (possibly reshaped) connection weight matrix.
	
	Inputs:
		weights (torch.Tensor or torch.cuda.Tensor): Weight matrix of Connection object.
		im (matplotlib.image.AxesImage): Used for re-drawing the weights plot.
		figsize (tuple(int)): Horizontal, vertical figure size in inches.
	
	Returns:
		(matplotlib.image.AxesImage): Used for re-drawing the weights plot.
	'''
	if not im:
		fig, ax = plt.subplots(figsize=figsize)
		
		im = ax.imshow(weights, cmap='hot_r', vmin=weights.min(), vmax=weights.max())
		div = make_axes_locatable(ax)
		cax = div.append_axes("right", size="5%", pad=0.05)
		
		ax.set_xticks(()); ax.set_yticks(())
		
		plt.colorbar(im, cax=cax)
		fig.tight_layout()
	else:
		im.set_data(weights)

	return im


def plot_assignments(assignments, im=None, figsize=(6, 6)):
	'''
	Plot the two-dimensional neuron assignments.
	
	Inputs:
		assignments (torch.Tensor or torch.cuda.Tensor): Matrix of neuron label assignments.
		im (matplotlib.image.AxesImage): Used for re-drawing the assignments plot.
		figsize (tuple(int)): Horizontal, vertical figure size in inches.
	
	Returns:
		(matplotlib.image.AxesImage): Used for re-drawing the assigments plot.
	'''
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
	'''
	Plot training accuracy curves.
	
	Inputs:
		performances (dict(list(float))): Lists of training accuracy estimates per voting scheme.
		ax (matplotlib.axes.Axes): Used for re-drawing the performance plot.
		figsize (tuple(int)): Horizontal, vertical figure size in inches.
	
	Returns:
		(matplotlib.axes.Axes): Used for re-drawing the performance plot.
	'''
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