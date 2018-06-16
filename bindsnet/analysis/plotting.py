import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.ion()

def plot_input(image, inpt, label=None, axes=None, ims=None, figsize=(8, 4)):
    '''
    Plots a two-dimensional image and its corresponding spike-train representation.
    
    Inputs:
        
        | :code:`image` (:code:`torch.Tensor`): A 2D array of floats depicting an input image.
        | :code:`inpt` (:code:`torch.Tensor`): A 2D array of floats depicting an image's spike-train encoding.
        | :code:`ims` (:code:`list(matplotlib.image.AxesImage)`): Used for re-drawing the input plots.
        | :code:`figsize` (:code:`tuple(int)`): Horizontal, vertical figure size in inches.
    
    Returns:
        
        | (:code:`axes` (:code:`list(matplotlib.image.AxesImage)`): Used for re-drawing the input plots.
        | (:code:`ims` (:code:`list(matplotlib.axes.Axes)): Used for re-drawing the input plots.
    '''
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ims = axes[0].imshow(image, cmap='binary'), axes[1].imshow(inpt, cmap='binary')
        
        if label is None:
            axes[0].set_title('Current image')
        else:
            axes[0].set_title('Current image (label = %d)' % label)
        
        axes[1].set_title('Reconstruction')
        
        for ax in axes:
            ax.set_xticks(()); ax.set_yticks(())
        
        fig.tight_layout()
    else:
        if label is not None:
            axes[0].set_title('Current image (label = %d)' % label)
        
        ims[0].set_data(image)
        ims[1].set_data(inpt)

    return axes, ims


def plot_spikes(spikes, ims=None, axes=None, time=None, n_neurons={}, figsize=(8, 4.5)):
    '''
    Plot spikes for any group(s) of neurons.

    Inputs:
        
        | :code:`spikes` (:code:`dict(torch.Tensor)`): Contains
        spiking data for groups of neurons of interest.
        | :code:`ims` (:code:`list(matplotlib.image.AxesImage)`): Used for re-drawing the spike plots.
        | :code:`axes` (:code:`list(matplotlib.axes.Axes)`): Used for re-drawing the spike plots.
        | :code:`time` (:code:`tuple(int)`): Plot spiking activity of neurons
        in the given time range. Default is entire simulation time.
        | :code:`n_neurons` (:code:`dict(tuple(int))`): Plot spiking activity
        of neurons in the given range of neurons. Default is all neurons.
        | :code:`figsize` (:code:`tuple(int)`): Horizontal, vertical figure size in inches.
    
    Returns:
        
        | (:code:`ims` (:code:`list(matplotlib.axes.Axes)): Used for re-drawing the spike plots.
        | (:code:`axes` (:code:`list(matplotlib.image.AxesImage)`): Used for re-drawing the spike plots.
        
    '''
    n_subplots = len(spikes.keys())
    spikes = {k : v.view(-1, v.size(-1)) for (k, v) in spikes.items()}
   
    if time is not None: 
        assert len(time) == 2, 'Need (start, stop) values for time argument'
        assert time[0] < time[1], 'Need start < stop in time argument'
    else:
        # Set it for entire duration
        for key in spikes.keys():
            time = (0, spikes[key].shape[1])
            break

    if len(n_neurons.keys()) != 0:
        assert len(n_neurons.keys()) <= n_subplots, \
            'n_neurons argument needs fewer entries than n_subplots'
        assert all(key in spikes.keys() for key in n_neurons.keys()), \
            'n_neurons keys must be subset of spikes keys'
    
    # Use all neurons if no argument provided.
    for key, val in spikes.items():
        if key not in n_neurons.keys():
            n_neurons[key] = (0, val.shape[0])
    
    if not ims:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        ims = []
        
        if n_subplots == 1:
            for datum in spikes.items():
                ims.append(axes.imshow(spikes[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                       time[0]:time[1]],
                                       cmap='binary'))

                args = (datum[0], n_neurons[datum[0]][0], n_neurons[datum[0]][1], time[0], time[1])
                plt.title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
                plt.xlabel('Simulation time'); plt.ylabel('Neuron index')
                axes.set_aspect('auto')
        else:
            for i, datum in enumerate(spikes.items()):
                ims.append(axes[i].imshow(datum[1][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                          time[0]:time[1]],
                                          cmap='binary'))
                
                args = (datum[0], n_neurons[datum[0]][0], n_neurons[datum[0]][1], time[0], time[1])
                axes[i].set_title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
            
            for ax in axes:
                ax.set_aspect('auto')
            
        plt.setp(axes, xticks=[], yticks=[], xlabel='Simulation time', ylabel='Neuron index')
        plt.tight_layout()
           
    else:
        if n_subplots == 1:
            for datum in spikes.items():
                ims[0].set_data(datum[1][n_neurons[datum[0]][0]:n_neurons[datum[0]][1], time[0]:time[1]])
                ims[0].autoscale()
                
                args = (datum[0], n_neurons[datum[0]][0], n_neurons[datum[0]][1], time[0], time[1])
                axes.set_title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
    
        else:
            for i, datum in enumerate(spikes.items()):
                ims[i].set_data(datum[1][n_neurons[datum[0]][0]:n_neurons[datum[0]][1], time[0]:time[1]])
                ims[i].autoscale()
                
                args = (datum[0], n_neurons[datum[0]][0], n_neurons[datum[0]][1], time[0], time[1])
                axes[i].set_title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
        
    plt.draw()
    return ims, axes


def plot_weights(weights, wmin=0.0, wmax=1.0, im=None, figsize=(5, 5)):
    '''
    Plot a connection weight matrix.
    
    Inputs:
        
        | :code:`weights` (:code:`torch.Tensor`): Weight matrix of Connection object.
        | :code:`wmin` (:code:`float`): Minimum allowed weight value.
        | :code:`wmax` (:code:`float`): Maximum allowed weight value.
        | :code:`im` (:code:`matplotlib.image.AxesImage`): Used for re-drawing the weights plot.
        | :code:`figsize` (:code:`tuple(int)`): Horizontal, vertical figure size in inches.
        
    Returns:
        
        | (:code:`im` (:code:`matplotlib.image.AxesImage`): Used for re-drawing the weights plot.
    '''
    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Connection weights')
        
        im = ax.imshow(weights, cmap='hot_r', vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        
        ax.set_xticks(()); ax.set_yticks(())
        ax.set_aspect('auto')
        
        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(weights)

    return im


def plot_conv2d_weights(weights, wmin=0.0, wmax=1.0, im=None, figsize=(5, 5)):
    '''
    Plot a connection weight matrix of a Conv2dConnection.
    
    Inputs:
        
        | :code:`weights` (:code:`torch.Tensor`): Weight matrix of Conv2dConnection object.
        | :code:`wmin` (:code:`float`): Minimum allowed weight value.
        | :code:`wmax` (:code:`float`): Maximum allowed weight value.
        | :code:`im` (:code:`matplotlib.image.AxesImage`): Used for re-drawing the weights plot.
        | :code:`figsize` (:code:`tuple(int)`): Horizontal, vertical figure size in inches.
        
    Returns:
        
        | (:code:`im` (:code:`matplotlib.image.AxesImage`): Used for re-drawing the weights plot.
    '''
    n_sqrt = int(np.ceil(np.sqrt(weights.size(0))))
    height = weights.size(2)
    width = weights.size(3)
    reshaped = torch.zeros(n_sqrt * weights.size(2), n_sqrt * weights.size(3))
    
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            if i * n_sqrt + j < weights.size(0):
                fltr = weights[i * n_sqrt + j].view(height, width)
                reshaped[i * height : (i + 1) * height,
                        (j % n_sqrt) * width : ((j % n_sqrt) + 1) * width] = fltr
    
    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(reshaped, cmap='hot_r', vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        
        for i in range(height, n_sqrt * height, height):
            ax.axhline(i - 0.5, color='g', linestyle='--')
        
        for i in range(width, n_sqrt * width, width):
            ax.axvline(i - 0.5, color='g', linestyle='--')
        
        ax.set_xticks(()); ax.set_yticks(())
        ax.set_aspect('auto')
        
        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im


def plot_assignments(assignments, im=None, figsize=(5, 5), classes=None):
    '''
    Plot the two-dimensional neuron assignments.
    
    Inputs:
        
        | :code:`assignments` (:code:`torch.Tensor`): Vector of neuron label assignments.
        | :code:`im` (:code:`matplotlib.image.AxesImage`):
        Used for re-drawing the assignments plot.
        | :code:`figsize` (:code:`tuple(int)`):
        Horizontal, vertical figure size in inches.
        | :code:`classes` (:code:`iterable`): Iterable of
        labels for colorbar ticks corresponding to data labels.
    
    Returns:
        
        | (:code:`im` (:code:`matplotlib.image.AxesImage`):
        Used for re-drawing the assigments plot.
    '''
    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Categorical assignments')

        color = plt.get_cmap('RdBu', 11)
        im = ax.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
        div = make_axes_locatable(ax); cax = div.append_axes("right",
                                                             size="5%",
                                                             pad=0.05)
        
        if classes is None:
            plt.colorbar(im, cax=cax, ticks=np.arange(-1, 10))
        else:
            cbar = plt.colorbar(im,
                                cax=cax,
                                ticks=np.arange(-1, len(classes)))

            cbar.ax.set_yticklabels(classes)   
            
        ax.set_xticks(()); ax.set_yticks(())
        fig.tight_layout()
    else:
        im.set_data(assignments)

    return im


def plot_performance(performances, ax=None, figsize=(7, 4)):
    '''
    Plot training accuracy curves.
    
    Inputs:
        
        | :code:`performances` (:code:`dict(list(float))`):
        Lists of training accuracy estimates per voting scheme.
        | :code:`ax` (:code:`matplotlib.axes.Axes`):
        Used for re-drawing the performance plot.
        | :code:`figsize` (:code:`tuple(int)`):
        Horizontal, vertical figure size in inches.
    
    Returns:
        
        | (:code:`ax` (:code:`matplotlib.axes.Axes`):
        Used for re-drawing the performance plot.
    '''
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
    else:
        ax.clear()

    for scheme in performances:
        ax.plot(range(len(performances[scheme])), [p for p in performances[scheme]], label=scheme)

    ax.set_ylim([0, 100])
    ax.set_title('Estimated classification accuracy')
    ax.set_xlabel('No. of examples'); ax.set_ylabel('Accuracy')
    ax.set_xticks(()); ax.set_yticks(range(0, 110, 10))
    ax.legend()

    return ax


def plot_general(monitor=None, ims=None, axes=None, labels=None, parameters=None, figsize=(8,4.5)):
    '''
    General plotting function for variables being monitored.
    
    Inputs:
        
        | :code:`monitor` (:code:`monitors.Monitor`):
        Contains state variables to be plotted.
        | :code:`ims` (:code:`list(matplotlib.image.AxesImage)`):
        Used for re-drawing plots.
        | :code:`axes` (:code:`list(matplotlib.axes.Axes)`):
        Used for re-drawing plots.
        | :code:`labels` (:code:`dict(dict(string))`):
        Used to set axis labels and titles for plotted variables.
        | :code:`parameters` (:code:`dict(dict(tuples(int)))`):
        Set time, number of neurons for plotted variables.
        | :code:`figsize` (:code:`tuple(int)`):
        Horizontal, vertical figure size in inches.
        
    Returns:
        
        | (:code:`ims` (:code:`list(matplotlib.axes.Axes)):
        Used for re-drawing plots.
        | (:code:`axes` (:code:`list(matplotlib.image.AxesImage)`):
        Used for re-drawing plots. 
    '''
    default = {'xlabel' : 'Simulation time', 'ylabel' : 'Index'}
    
    if monitor is None:
        print("Did you forget to provide monitors?")
        raise TypeError
    
    if labels is None:
        labels = {var : {'title' : 'Recording of %s'%(var),
                         'xlabel' : 'Simulation time',
                         'ylabel' : 'Index'} for var in monitor.state_vars}
    
    # Default axis parameters
    else:
        for var in monitor.state_vars:
            for lb in ['title', 'xlabel', 'ylabel']:
                
                if lb not in labels[var].keys() and lb == 'title':
                    labels[var][lb] = 'Recording of %s'%var
                    
                elif lb not in labels[var].keys():
                    labels[var][lb] = default[lb]
                    
    if parameters is None:
        # Monitor object is of a class in nodes
        parameters = {var : {'time' : (0, monitor.get(var).shape[1]),
                           'n_neurons' : (0, monitor.get(var).shape[0]),
                           'cmap' : 'binary'} for var in monitor.state_vars}
        
    else:
        #if type(monitor.obj) in nodes.__dict__.values():
        for var in monitor.state_vars:
            if 'time' not in parameters[var].keys():
                parameters[var]['time'] = (0, monitor.get(var).shape[1])
            
            if 'n_neurons' not in parameters[var].keys():
                parameters[var]['n_neurons'] = (0, monitor.get(var).shape[0])
            
            if 'cmap' not in parameters[var].keys():
                parameters[var]['cmap'] = 'binary'
        
    n_subplots = len(monitor.state_vars)
    if not ims:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        ims = []
        
        if n_subplots == 1:
            for var in monitor.state_vars:
                # For Weights
                if parameters[var]['cmap'] == 'hot_r' or parameters[var]['cmap'] == 'hot':
                    ims.append(axes.matshow(monitor.get(var)[parameters[var]['n_neurons'][0]:parameters[var]['n_neurons'][1],
                                            parameters[var]['time'][0]:parameters[var]['time'][1]]))
                else:
                    ims.append(axes.imshow(monitor.get(var)[parameters[var]['n_neurons'][0]:parameters[var]['n_neurons'][1],
                                           parameters[var]['time'][0]:parameters[var]['time'][1]]))
            
                plt.title(labels[var]['title'])
                plt.xlabel(labels[var]['xlabel'])
                plt.ylabel(labels[var]['ylabel'])
            
            axes.set_aspect('auto')

        else: # Plot each monitor variable at a time
            for i, var in enumerate(monitor.state_vars):
                if parameters[var]['cmap'] == 'hot_r' or parameters[var]['cmap'] == 'hot':
                    ims.append(axes[i].matshow(monitor.get(var)[parameters[var]['n_neurons'][0]:parameters[var]['n_neurons'][1],
                                               parameters[var]['time'][0]:parameters[var]['time'][1]]))
                else:
                    ims.append(axes[i].imshow(monitor.get(var)[parameters[var]['n_neurons'][0]:parameters[var]['n_neurons'][1],
                                              parameters[var]['time'][0]:parameters[var]['time'][1]]))
                
                axes.set_title(labels[var]['title'])
                axes.set_xlabel(labels[var]['xlabel'])
                axes.set_ylabel(labels[var]['ylabel'])
                
            axes.set_aspect('auto') 

    # axes given        
    else:
        assert(len(ims) == n_subplots)
        
    return ims, axes


def plot_voltages(voltages, ims=None, axes=None, time=None, n_neurons={}, figsize=(8, 4.5)):
    '''
    Plot voltages for any group(s) of neurons.

    Inputs:
        
        | :code:`voltages` (:code:`dict(torch.Tensor`)): Contains voltage data by neuron layers.
        | :code:`ims` (:code:`list(matplotlib.image.AxesImage)`): Used for re-drawing the spike plots.
        | :code:`axes` (:code:`list(matplotlib.axes.Axes)`): Used for re-drawing the spike plots.
        | :code:`time` (:code:`tuple(int)`): Plot voltages of neurons in given time range. Default is entire simulation time.
        | :code:`n_neurons` (:code:`dict(tuple(int))`): Plot voltages of neurons in given range of neurons. Default is all neurons.
        | :code:`figsize` (:code:`tuple(int)`): Horizontal, vertical figure size in inches.
    
    Returns:
        
        | (:code:`ims` (:code:`list(matplotlib.axes.Axes)): Used for re-drawing the voltage plots.
        | (:code:`axes` (:code:`list(matplotlib.image.AxesImage)`): Used for re-drawing the voltage plots.
        
    '''
    n_subplots = len(voltages.keys())
    
    # Confirm only 2 values for time were given
    if time is not None: 
        assert(len(time) == 2)
        assert(time[0] < time[1])

    else:  # Set it for entire duration
        for key in voltages.keys():
            time = (0, voltages[key].shape[1])
            break
    
    # Number of neurons setup
    if len(n_neurons.keys()) != 0:
        # Don't have to give numbers for all keys
        assert(len(n_neurons.keys()) <= n_subplots)
        # Keys given must be same as the ones used in spikes dict
        assert(all(key in voltages.keys() for key in n_neurons.keys()))
    
    for key, val in voltages.items():
        if key not in n_neurons.keys():
            n_neurons[key] = (0, val.shape[0])
            
    if not ims:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        ims = []

        if n_subplots == 1:  # Plotting only one image
            for datum in voltages.items():
                ims.append(axes.matshow(voltages[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                        time[0]:time[1]]))
                plt.title('%s voltages for neurons (%d - %d) from t = %d to %d ' % (datum[0],
                                                                                    n_neurons[datum[0]][0],
                                                                                    n_neurons[datum[0]][1],
                                                                                    time[0],
                                                                                    time[1]))
                plt.xlabel('Time (ms)'); plt.ylabel('Neuron index')
                
                axes.set_aspect('auto')
                
        else:  # Plot each layer at a time
            for i, datum in enumerate(voltages.items()):
                    ims.append(axes[i].matshow(datum[1][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                               time[0]:time[1]]))
                    axes[i].set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % (datum[0],
                                                                                               n_neurons[datum[0]][0],
                                                                                               n_neurons[datum[0]][1], 
                                                                                               time[0], 
                                                                                               time[1]))
            
            for ax in axes:
                ax.set_aspect('auto')

        plt.setp(axes, xticks=[], yticks=[], xlabel='Simulation time', ylabel='Neuron index')
        plt.tight_layout()
           
    else:  # Plotting figure given
        if n_subplots == 1:  # Plotting only one image
            for datum in voltages.items():
                axes.clear()
                axes.matshow(voltages[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                             time[0]:time[1]])
                axes.set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % (datum[0],
                                                                                         n_neurons[datum[0]][0],
                                                                                         n_neurons[datum[0]][1],
                                                                                         time[0],
                                                                                         time[1]))
                
                axes.set_aspect('auto')

        else: # Plot each layer at a time
            for i, datum in enumerate(voltages.items()):
                axes[i].clear()
                axes[i].matshow(voltages[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                time[0]:time[1]])
                axes[i].set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % (datum[0],
                                                                                            n_neurons[datum[0]][0],
                                                                                            n_neurons[datum[0]][1],
                                                                                            time[0],
                                                                                            time[1]))

            for ax in axes:
                ax.set_aspect('auto')
        
        plt.setp(axes, xticks=[], yticks=[], xlabel='Simulation time', ylabel='Neuron index')
        plt.tight_layout()
    
    return ims, axes
