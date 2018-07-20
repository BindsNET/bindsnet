import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_weights_movie(ws, sample_every=1):
    """
    Create and plot movie of weights (:code:`ws`).
    
    Inputs:
    
        | :code:`ws` (:code:`numpy.array`): Numpy array
        of shape :code:`[n_examples, source, target, time]`
        | :code:`sample_every` (:code:`int`): Sub-sample using this parameter. 
    """
    weights = []
    
    # Obtain samples from the weights for every example
    for i in range(ws.shape[0]):
        sub_sampled_weight = ws[i, :, :, range(0, ws[i].shape[2], sample_every)]
        weights.append(sub_sampled_weight)
    else:
        weights = np.concatenate(weights, axis=0)
    
    # Initialize plot
    fig = plt.figure()
    im = plt.imshow(weights[0, :, :], cmap='hot_r', animated=True, vmin=0, vmax=1)
    plt.axis('off'); plt.colorbar(im)
        
    # Update function for the animation
    def update(j):
        im.set_data(weights[j, :, :])
        return [im]
    
    # Initialize animatino
    global ani; ani=0
    ani = animation.FuncAnimation(fig, update, frames=weights.shape[-1], interval=1000, blit=True)
    plt.show()
    
def plot_spike_trains_for_example(spikes, n_ex=None, top_k=None, indices=None):
    '''
    Plot spike trains for top-k neurons or for specific indices.
    
    Inputs:
        
        | :code:`spikes` (:code:`torch.Tensor (n_examples, n_neurons, time)`):
        Spiking train data for a population of neurons for one example.
        | :code:`n_ex` (:code:`int`): Allows user to pick
        which example to plot spikes for. Must be >= 0.
        | :code:`top_k` (:code:`int`): Plot k neurons that spiked the most for n_ex example.
        | :code:`indices` (:code:`list(int)`): Plot specific neurons'
        spiking activity instead of top_k. Meant to replace top_k. 
    '''

    assert (n_ex is not None and n_ex >= 0 and n_ex < spikes.shape[0])
    
    plt.figure()
    
    if top_k is None and indices is None: # Plot all neurons' spiking activity
        spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, :, :]]
        plt.title('Spiking activity for all %d neurons'%spikes.shape[1])
        
    elif top_k is None: # Plot based on indices parameter
        assert (indices is not None)
        spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, indices, :]]
        
    elif indices is None: # Plot based on top_k parameter
        assert (top_k is not None)
        # Obtain the top k neurons that fired the most
        top_k_loc = np.argsort(np.sum(spikes[n_ex,:,:], axis=1), axis=0)[::-1]
        spike_per_neuron = [np.argwhere(i==1).flatten() for i in spikes[n_ex, top_k_loc[0:top_k], :]]
        plt.title('Spiking activity for top %d neurons'%top_k)
        
    plt.eventplot(spike_per_neuron, linelengths= [0.5]*len(spike_per_neuron))
    plt.xlabel('Simulation Time'); plt.ylabel('Neuron index')
    plt.show()

def plot_voltage(voltage, n_ex=0, n_neuron=0, time=None, threshold=None):
    '''
    Plot voltage for a single neuron on a specific example.
    
    Inputs:
        
        | :code:`voltage` (:code:`torch.Tensor` or :code:`numpy.array`):
        Tensor or array of shape :code:`[n_examples, n_neurons, time]`.
        | :code:`n_ex` (:code:`int`): Allows user
        to pick which example to plot voltage for.
        | :code:`n_neuron` (:code:`int`): Neuron
        index for which to plot voltages for.
        | :code:`time` (:code:`tuple(int)`): Plot spiking
        activity of neurons between the given range of time. 
        | :code:`threshold` (:code:`float`): Neuron
        spiking threshold. Will be shown on the plot.
    '''
    
    assert (n_ex >= 0 and n_neuron >= 0)
    assert (n_ex < voltage.shape[0] and n_neuron < voltage.shape[1])

    if time is None:
        time = (0, voltage.shape[-1])
    else:
        assert (time[0] < time[1])
        assert (time[1] <= voltage.shape[-1])
    
    timer = np.arange(time[0], time[1])
    time_ticks = np.arange(time[0], time[1]+1, 10)
    
    plt.figure()
    plt.plot(voltage[n_ex, n_neuron, timer])
    plt.xlabel('Simulation Time')
    plt.ylabel('Voltage')
    plt.title('Membrane voltage of neuron %d for example %d' % (n_neuron, n_ex + 1))
    locs, labels = plt.xticks()
    locs = range(int(locs[1]), int(locs[-1]), 10)
    plt.xticks(locs, time_ticks)
    
    # Draw threshold line only if given
    if threshold is not None:
        plt.axhline(threshold, linestyle='--', color='black', zorder=0)
        
    plt.show()
