import torch
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.network import AbstractMonitor, Network

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, List, Optional, Any, Sized, Dict

from ..utils import reshape_locally_connected_weights

plt.ion()


def plot_input(image: torch.Tensor, inpt: torch.Tensor, label: Optional[int] = None, axes: List[Axes] = None,
               ims: List[AxesImage] = None,
               figsize: Tuple[int, int]=(8, 4)) -> Tuple[List[Axes], List[AxesImage]]:
    # language=rst
    """
    Plots a two-dimensional image and its corresponding spike-train representation.

    :param image: A 2D array of floats depicting an input image.
    :param inpt: A 2D array of floats depicting an image's spike-train encoding.
    :param label: Class label of the input data.
    :param axes: Used for re-drawing the input plots.
    :param ims: Used for re-drawing the input plots.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: Tuple of ``(axes, ims)`` used for re-drawing the input plots.
    """
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


def plot_spikes(network: Optional[Network] = None, spikes: Optional[Dict[str, torch.Tensor]] = None,
                layer_to_monitor: Optional[Dict[str, str]] = None, layers: Optional[List[str]] = None,
                time: Optional[Dict[str, Tuple[int, int]]] = None,
                n_neurons: Optional[Dict[str, Tuple[int, int]]] = None, ims=None, axes: List[AxesImage] = None,
                figsize: Tuple[float, float] = (8.0, 4.5)) -> Tuple[List[AxesImage], List[Axes]]:
    # language=rst
    """
    Plot spikes for any group(s) of neurons.

    :param spikes: Mapping from layer names to spiking data.
    :param time: Plot spiking activity of neurons in the given time range. Default is entire simulation time.
    :param n_neurons: Plot spiking activity of neurons in the given range of neurons. Default is all neurons.
    :param ims: Used for re-drawing the plots.
    :param axes: Used for re-drawing the plots.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: ``ims, axes``: Used for re-drawing the plots.
    """

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


def plot_spikes_new(network=None, spikes=None, layer_to_monitor={}, layers=[], time={}, n_neurons={}, ims=None, axes=None, figsize=(8, 4.5)):
    """
    Plot spikes for any group(s) of neurons. Default behavior will plot everything.

    :param network: ``Network`` containing spike monitor objects.
    :param spikes: Mapping from layer names to spiking data.
    :param layer_to_monitor: Mapping of layer names to monitors.
    :param layers: List of network's layer names to plot spikes for.
    :param time: Plot spiking activity of neurons in the given time range. Default is entire simulation time.
    :param n_neurons: Plot spiking activity of neurons in the given range of neurons. Default is all neurons.
    :param ims: Used for re-drawing the plots.
    :param axes: Used for re-drawing the plots.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: ``ims, axes``: Used for re-drawing the plots.
    """
    assert network is not None or spikes is not None, 'Need either "network" or "spikes" argument.'

    if layer_to_monitor is None:
        layer_to_monitor = {}

    assert network is not None or spikes is not None, 'No plotting information'

    if time is None:
        time = {}

    if n_neurons is None:
        n_neurons = {}

    # Set to all layers if no layers were requested
    if layers is None:
        layers = []
        if network is not None: # If network is provided
            for layer in network.layers:
                layers.append(layer)
        else: # Use layers from spikes
            for layer in spikes.keys():
                layers.append(layer)
    else: # Specific layers in network or spikes were provided
        for layer in layers:
            if network is not None:
                assert layer in network.layers,\
                    f"'{layer}' does not exist within network's layers. \
                    Network contains layers: {list(network.layers.keys())}"
            else:
                assert layer in spikes.keys(), \
                    f"'{layer}' does not exist within given spiking information.\
                    Spikes contain layers: {list(spikes.keys())}"

    # Pre-setup plots
    if network is not None:
        n_subplots = len(layers)
    else: # Obtain information from spikes
        n_subplots = len(layers)
        spikes = {k : v.view(-1, v.size(-1)) for (k, v) in spikes.items()}

    # Set up monitor to layer names
    if network is not None:
        for layer, monitor_name in layer_to_monitor.items():
            assert monitor_name in network.monitors, \
                f"'{monitor_name}' does not exist within network. Network contains layers: {list(network.layers.keys())}"
            assert layer in network.layers, \
                f"'{layer}' does not exist within network. Network contains monitors: {list(network.monitors.keys())}"

        # Not all mappings were provided. Assume layer names.
        if len(layer_to_monitor) != len(layers):
            for layer in set(network.layers) - set(layer_to_monitor):
                layer_to_monitor[layer] = layer

    assert len(n_neurons) <= n_subplots, \
        'n_neurons argument needs fewer entries than n_subplots'
    assert len(time) <= n_subplots, \
        'time argument needs fewer entries than n_subplots'

    # Check if appropriate values for time were provided
    if time != {}:
        for key, val in time.items():
            if network is not None:
                assert layer in network.layers,\
                    f"'{layer}' given in 'time' paramter does not exist within \
                    network's layers. Network contains layers: {list(network.layers.keys())}"
            else:
                assert layer in spikes.keys(), \
                    f"'{layer}' given in 'time' paramter does not exist within \
                    spiking information. Spikes contain layers: {list(spikes.keys())}"
            assert len(val) == 2, 'Need (start, stop) values for time argument'
            assert val[0] < val[1], 'Need start < stop in time argument'

    # Set it for entire duration
    for layer in layers:
        if layer not in time.keys():
            if network is not None: # Take from monitors
                time[layer] = (0, network.monitors[layer_to_monitor[layer]].get('s').shape[1])
            else: # Take from spikes
                time[layer] = (0, spikes[layer].shape[1])

    # Check if appropriate values for n_neurons were provided
    if n_neurons != {}:
        for key, val in n_neurons.items():
            if network is not None:
                assert layer in network.layers,\
                    f"'{layer}' given in 'n_neurons' paramter does not exist within \
                    network's layers. Network contains layers: {list(network.layers.keys())}"
            else:
                assert layer in spikes.keys(), \
                    f"'{layer}' given in 'n_neurons' paramter does not exist within \
                    spiking information. Spikes contain layers: {list(spikes.keys())}"

    # Set to use all neurons
    for layer in layers:
        if layer not in n_neurons.keys():
            if network is not None: # Take from monitors
                n_neurons[layer] = (0, network.monitors[layer_to_monitor[layer]].get('s').shape[0])
            else:
                n_neurons[layer] = (0, spikes[layer].shape[0])

    # Create subplots
    if not ims:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        ims = []

        if n_subplots == 1:
            # Assuming monitor names and layer names are matching
            for layer in layers:
                if network is not None: # Plot using network monitors
                    ims.append(axes.imshow(network.monitors[layer_to_monitor[layer]].get('s')[n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]],
                                   cmap='binary'))
                else: # Plot using spikes
                    ims.append(axes.imshow(spikes[layer][n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]],
                                   cmap='binary'))

                args = (layer, n_neurons[layer][0], n_neurons[layer][1], time[layer][0], time[layer][1])
                plt.title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
                plt.xlabel('Simulation time'); plt.ylabel('Neuron index')
                axes.set_aspect('auto')
        else: # Multiple subplots
            for i, layer in enumerate(layers):
                if network is not None: # Plot using network monitors
                    ims.append(axes[i].imshow(network.monitors[layer_to_monitor[layer]].get('s')[n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]],
                                   cmap='binary'))
                else: # Plot using spikes
                    ims.append(axes[i].imshow(spikes[layer][n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]],
                                   cmap='binary'))

                    args = (layer, n_neurons[layer][0], n_neurons[layer][1], time[layer][0], time[layer][1])
                    axes[i].set_title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)

            for ax in axes:
                ax.set_aspect('auto')

        plt.setp(axes, xticks=[], yticks=[], xlabel='Simulation time', ylabel='Neuron index')
        plt.tight_layout()
    else: # Update subplots
        if n_subplots == 1:
            for layer in layers:
                if network is not None: # Plot using network monitors
                    ims[0].set_data(network.monitors[layer_to_monitor[layer]].get('s')[n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]])
                else: # Plot using spikes
                    ims[0].set_data(spikes[layer][n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]])

                ims[0].autoscale()

                args = (layer, n_neurons[layer][0], n_neurons[layer][1], time[layer][0], time[layer][1])
                axes.set_title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
        else: # Multiple subplots
            for i, layer in enumerate(layers):
                if network is not None: # Plot using network monitors
                    ims[i].set_data(network.monitors[layer_to_monitor[layer]].get('s')[n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]])
                else: # Plot using spikes
                    ims[i].set_data(spikes[layer][n_neurons[layer][0]:n_neurons[layer][1],
                                   time[layer][0]:time[layer][1]])

                ims[i].autoscale()

                args = (layer, n_neurons[layer][0], n_neurons[layer][1], time[layer][0], time[layer][1])
                axes[i].set_title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)

    plt.draw()
    return ims, axes


def plot_weights(weights: torch.Tensor, wmin: Optional[float] = None, wmax: Optional[float] = None,
                 im: Optional[AxesImage] = None, figsize: Tuple[int, int] = (5, 5)) -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix.

    :param weights: Weight matrix of ``Connection`` object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: ``AxesImage`` for re-drawing the weights plot.
    """
    if wmin is None:
        wmin = weights.min()

    if wmax is None:
        wmax = weights.max()

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


def plot_conv2d_weights(weights: torch.Tensor, wmin: float = 0.0, wmax: float = 1.0, im: Optional[AxesImage] = None,
                        figsize: Tuple[int, int] = (5, 5)) -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix of a Conv2dConnection.

    :param weights: Weight matrix of Conv2dConnection object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: Used for re-drawing the weights plot.
    """
    n_sqrt = int(np.ceil(np.sqrt(weights.size(0))))
    height = weights.size(2)
    width = weights.size(3)
    reshaped = torch.zeros(n_sqrt * weights.size(2), n_sqrt * weights.size(3))

    for i in range(n_sqrt):
        for j in range(n_sqrt):
            if i * n_sqrt + j < weights.size(0):
                fltr = weights[i * n_sqrt + j].view(height, width)
                reshaped[i * height: (i + 1) * height, (j % n_sqrt) * width: ((j % n_sqrt) + 1) * width] = fltr

    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Connection weights')

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


def plot_locally_connected_weights(weights: torch.Tensor, n_filters: int, kernel_size: int, conv_size: int,
                                   locations: torch.Tensor, input_sqrt: int, wmin: float = 0.0, wmax: float = 1.0,
                                   im: Optional[AxesImage] = None, figsize: Tuple[int, int] = (5, 5)) -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix of a :code:`Connection` with `locally connected structure
    <http://yann.lecun.com/exdb/publis/pdf/gregor-nips-11.pdf>_.

    :param weights: Weight matrix of Conv2dConnection object.
    :param n_filters: No. of convolution kernels in use.
    :param kernel_size: Side length of 2D convolution kernels.
    :param conv_size: Side length of 2D convolution population.
    :param locations: Indices of input receptive fields for convolution population neurons.
    :param input_sqrt: Side length of 2D input data.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: Used for re-drawing the weights plot.
    """
    reshaped = reshape_locally_connected_weights(weights, n_filters, kernel_size, conv_size, locations, input_sqrt)

    n_sqrt = int(np.ceil(np.sqrt(n_filters)))

    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Connection weights')

        im = ax.imshow(reshaped, cmap='hot_r', vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        for i in range(n_sqrt * kernel_size, n_sqrt * conv_size * kernel_size, n_sqrt * kernel_size):
            ax.axhline(i - 0.5, color='g', linestyle='--')

        for i in range(n_sqrt * kernel_size, n_sqrt * conv_size * kernel_size, n_sqrt * kernel_size):
            ax.axvline(i - 0.5, color='g', linestyle='--')

        ax.set_xticks(()); ax.set_yticks(())
        ax.set_aspect('auto')

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im


def plot_assignments(assignments: torch.Tensor, im: Optional[AxesImage] = None, figsize: Tuple[int, int] = (5, 5),
                     classes: Optional[Sized] = None) -> AxesImage:
    # language=rst
    """
    Plot the two-dimensional neuron assignments.

    :param assignments: Vector of neuron label assignments.
    :param im: Used for re-drawing the assignments plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param classes: Iterable of labels for colorbar ticks corresponding to data labels.
    :return: Used for re-drawing the assigments plot.
    """
    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Categorical assignments')

        color = plt.get_cmap('RdBu', 11)
        im = ax.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        if classes is None:
            plt.colorbar(im, cax=cax, ticks=np.arange(-1, 10))
        else:
            cbar = plt.colorbar(im, cax=cax, ticks=np.arange(-1, len(classes)))
            cbar.ax.set_yticklabels(classes)

        ax.set_xticks(()); ax.set_yticks(())
        fig.tight_layout()
    else:
        im.set_data(assignments)

    return im


def plot_performance(performances: Dict[str, List[float]], ax: Optional[Axes] = None,
                     figsize: Tuple[int, int] = (7, 4)) -> Axes:
    # language=rst
    """
    Plot training accuracy curves.

    :param performances: Lists of training accuracy estimates per voting scheme.
    :param ax: Used for re-drawing the performance plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: Used for re-drawing the performance plot.
    """
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


def plot_voltages(voltages: Dict[str, torch.Tensor], ims: Optional[List[AxesImage]] = None,
                  axes: Optional[List[Axes]] = None, time: Tuple[int, int] = None,
                  n_neurons: Optional[Dict[str, Tuple[int, int]]] = None,
                  figsize: Tuple[float, float] = (8.0, 4.5)) -> Tuple[List[Axes], List[AxesImage]]:
    # language=rst
    """
    Plot voltages for any group(s) of neurons.

    :param voltages: Contains voltage data by neuron layers.
    :param ims: Used for re-drawing the plots.
    :param axes: Used for re-drawing the plots.
    :param time: Plot voltages of neurons in given time range. Default is entire simulation time.
    :param n_neurons: Plot voltages of neurons in given range of neurons. Default is all neurons.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: ``ims, axes``: Used for re-drawing the plots.
    """
    n_subplots = len(voltages.keys())

    if time is None:
        for key in voltages.keys():
            time = (0, voltages[key].shape[1])
            break

    if n_neurons is None:
        n_neurons = {}

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
                ims.append(axes[i].matshow(datum[1][n_neurons[datum[0]][0]:n_neurons[datum[0]][1], time[0]:time[1]]))
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
                axes.matshow(voltages[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1], time[0]:time[1]])
                axes.set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % (datum[0],
                                                                                         n_neurons[datum[0]][0],
                                                                                         n_neurons[datum[0]][1],
                                                                                         time[0],
                                                                                         time[1]))

                axes.set_aspect('auto')

        else: # Plot each layer at a time
            for i, datum in enumerate(voltages.items()):
                axes[i].clear()
                axes[i].matshow(voltages[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1], time[0]:time[1]])
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
