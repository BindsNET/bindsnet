import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from torch.nn.modules.utils import _pair
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, List, Optional, Sized, Dict, Union

from ..utils import reshape_locally_connected_weights

plt.ion()


def plot_input(image: torch.Tensor, inpt: torch.Tensor, label: Optional[int] = None, axes: List[Axes] = None,
               ims: List[AxesImage] = None, figsize: Tuple[int, int]=(8, 4)) -> Tuple[List[Axes], List[AxesImage]]:
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
            ax.set_xticks(())
            ax.set_yticks(())

        fig.tight_layout()
    else:
        if label is not None:
            axes[0].set_title('Current image (label = %d)' % label)

        ims[0].set_data(image)
        ims[1].set_data(inpt)

    return axes, ims


def plot_spikes(spikes: Dict[str, torch.Tensor], time: Optional[Tuple[int, int]] = None,
                n_neurons: Optional[Dict[str, Tuple[int, int]]] = None, ims: Optional[List[AxesImage]] = None,
                axes: Optional[Union[Axes, List[Axes]]] = None,
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
    if n_neurons is None:
        n_neurons = {}

    spikes = {k: v.view(-1, v.size(-1)) for (k, v) in spikes.items()}
    if time is None:
        # Set it for entire duration
        for key in spikes.keys():
            time = (0, spikes[key].shape[1])
            break

    # Use all neurons if no argument provided.
    for key, val in spikes.items():
        if key not in n_neurons.keys():
            n_neurons[key] = (0, val.shape[0])

    if ims is None:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        ims = []
        if n_subplots == 1:
            for datum in spikes.items():
                ims.append(axes.imshow(spikes[datum[0]][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                       time[0]:time[1]], cmap='binary'))
                args = (datum[0], n_neurons[datum[0]][0], n_neurons[datum[0]][1], time[0], time[1])
                plt.title('%s spikes for neurons (%d - %d) from t = %d to %d ' % args)
                plt.xlabel('Simulation time'); plt.ylabel('Neuron index')
                axes.set_aspect('auto')
        else:
            for i, datum in enumerate(spikes.items()):
                ims.append(axes[i].imshow(datum[1][n_neurons[datum[0]][0]:n_neurons[datum[0]][1],
                                          time[0]:time[1]], cmap='binary'))
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


def plot_weights(weights: torch.Tensor, wmin: Optional[float] = 0, wmax: Optional[float] = 1,
                 im: Optional[AxesImage] = None, figsize: Tuple[int, int] = (5, 5), cmap: str = 'hot_r') -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix.

    :param weights: Weight matrix of ``Connection`` object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: ``AxesImage`` for re-drawing the weights plot.
    """
    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(weights, cmap=cmap, vmin=wmin, vmax=wmax)
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
                        figsize: Tuple[int, int] = (5, 5), cmap: str = 'hot_r') -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix of a Conv2dConnection.

    :param weights: Weight matrix of Conv2dConnection object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """
    sqrt1 = int(np.ceil(np.sqrt(weights.size(0))))
    sqrt2 = int(np.ceil(np.sqrt(weights.size(1))))
    height, width = weights.size(2), weights.size(3)
    reshaped = torch.zeros(sqrt1 * sqrt2 * weights.size(2), sqrt1 * sqrt2 * weights.size(3))

    for i in range(sqrt1):
        for j in range(sqrt1):
            for k in range(sqrt2):
                for l in range(sqrt2):
                    if i * sqrt1 + j < weights.size(0) and k * sqrt2 + l < weights.size(1):
                        fltr = weights[i * sqrt1 + j, k * sqrt2 + l].view(height, width)
                        reshaped[i * height + k * height * sqrt1:
                                 (i + 1) * height + k * height * sqrt1,
                                 (j % sqrt1) * width + (l % sqrt2) * width * sqrt1:
                                 ((j % sqrt1) + 1) * width + (l % sqrt2) * width * sqrt1] = fltr

    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(reshaped, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        for i in range(height, sqrt1 * sqrt2 * height, height):
            ax.axhline(i - 0.5, color='g', linestyle='--')
            if i % sqrt1 == 0:
                ax.axhline(i - 0.5, color='g', linestyle='-')

        for i in range(width, sqrt1 * sqrt2 * width, width):
            ax.axvline(i - 0.5, color='g', linestyle='--')
            if i % sqrt1 == 0:
                ax.axvline(i - 0.5, color='g', linestyle='-')

        ax.set_xticks(()); ax.set_yticks(())
        ax.set_aspect('auto')

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im

def plot_conv2d_local_weights(weights: torch.Tensor, wmin: float = 0.0, wmax: float = 1.0, im: Optional[AxesImage] = None,
                        figsize: Tuple[int, int] = (5, 5), cmap: str = 'hot_r') -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix of a Conv2dConnection.

    :param weights: Weight matrix of Conv2dConnection object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """
    sqrt1 = int(np.ceil(np.sqrt(weights.size(2))))
    sqrt2 = int(np.ceil(np.sqrt(weights.size(3))))
    height, width = weights.size(4), weights.size(5)
    conv_h = weights.size(0)
    conv_w = weights.size(1)

    reshaped = torch.zeros(sqrt1 * sqrt2 * height * conv_h, sqrt1 * sqrt2 * width * conv_w)

    for i in range(sqrt1):
        for j in range(sqrt1):
            for k in range(sqrt2):
                for l in range(sqrt2):
                    #if i * sqrt1 + j < weights.size(0) and k * sqrt2 + l < weights.size(1):
                    for m in range(conv_h):
                        for n in range(conv_w):
                            fltr = weights[m, n, i * sqrt1 + j, k * sqrt2 + l].view(height, width)
                            reshaped[i * conv_h* height + k * height * sqrt1 + m*height:
                                     (i) * conv_h* height + k * height * sqrt1 + (m+1)*height,
                                     (j) *conv_w* width + (l % sqrt2) * width * sqrt1 + n*width:
                                     ((j)) * conv_w*width + (l % sqrt2) * width * sqrt1 + (n+1)*width] = fltr


    if not im:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(reshaped, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        ax.set_xticks(()); ax.set_yticks(())
        ax.set_aspect('auto')

        plt.colorbar(im, cax=cax)
        fig.tight_layout()
    else:
        im.set_data(reshaped)

    return im



def plot_locally_connected_weights(weights: torch.Tensor, n_filters: int, kernel_size: Union[int, Tuple[int, int]],
                                   conv_size: Union[int, Tuple[int, int]], locations: torch.Tensor,
                                   input_sqrt: Union[int, Tuple[int, int]], wmin: float = 0.0, wmax: float = 1.0,
                                   im: Optional[AxesImage] = None, lines: bool = True,
                                   figsize: Tuple[int, int] = (5, 5), cmap: str = 'hot_r') -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix of a :code:`Connection` with `locally connected structure
    <http://yann.lecun.com/exdb/publis/pdf/gregor-nips-11.pdf>_.

    :param weights: Weight matrix of Conv2dConnection object.
    :param n_filters: No. of convolution kernels in use.
    :param kernel_size: Side length(s) of 2D convolution kernels.
    :param conv_size: Side length(s) of 2D convolution population.
    :param locations: Indices of input receptive fields for convolution population neurons.
    :param input_sqrt: Side length(s) of 2D input data.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param lines: Whether or not to draw horizontal and vertical lines separating input regions.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: Used for re-drawing the weights plot.
    """
    kernel_size = _pair(kernel_size)
    conv_size = _pair(conv_size)
    input_sqrt = _pair(input_sqrt)

    reshaped = reshape_locally_connected_weights(weights, n_filters, kernel_size, conv_size, locations, input_sqrt)
    n_sqrt = int(np.ceil(np.sqrt(n_filters)))

    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(reshaped.cpu(), cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        if lines:
            for i in range(n_sqrt * kernel_size[0], n_sqrt * conv_size[0] * kernel_size[0], n_sqrt * kernel_size[0]):
                ax.axhline(i - 0.5, color='g', linestyle='--')

            for i in range(n_sqrt * kernel_size[1], n_sqrt * conv_size[1] * kernel_size[1], n_sqrt * kernel_size[1]):
                ax.axvline(i - 0.5, color='g', linestyle='--')

        ax.set_xticks(())
        ax.set_yticks(())
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

        if classes is None:
            color = plt.get_cmap('RdBu', 11)
            im = ax.matshow(assignments, cmap=color, vmin=-1.5, vmax=9.5)
        else:
            color = plt.get_cmap('RdBu', len(classes) + 1)
            im = ax.matshow(assignments, cmap=color, vmin=-1.5, vmax=len(classes) - 0.5)

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        if classes is None:
            cbar = plt.colorbar(im, cax=cax, ticks=list(range(-1, 11)))
            cbar.ax.set_yticklabels(['none'] + list(range(10)))
        else:
            cbar = plt.colorbar(im, cax=cax, ticks=np.arange(-1, len(classes)))
            cbar.ax.set_yticklabels(['none'] + list(classes))

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
                  cmap: Optional[str] = 'jet', plot_type: str = 'color', threshold: Dict[str,float] = None,
                  figsize: Tuple[float, float] = (8.0, 4.5),) -> Tuple[List[AxesImage], List[Axes]]:
    # language=rst
    """
    Plot voltages for any group(s) of neurons.

    :param voltages: Contains voltage data by neuron layers.
    :param ims: Used for re-drawing the plots.
    :param axes: Used for re-drawing the plots.
    :param time: Plot voltages of neurons in given time range. Default is entire simulation time.
    :param n_neurons: Plot voltages of neurons in given range of neurons. Default is all neurons.
    :param cmap: Matplotlib colormap to use.
    :param figsize: Horizontal, vertical figure size in inches.
    :param plot_type: The way how to draw graph. 'color' for pcolormesh, 'line' for curved lines.
    :param threshold: Threshold of each layer.
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
            for v in voltages.items():
                if plot_type == 'line':
                    ims.append(axes.plot(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]].cpu().numpy().T))

                    if threshold is not None:
                        ims.append(axes.axhline(y=threshold[v[0]], c='r', linestyle='--'))
                else:
                    ims.append(axes.pcolormesh(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]], cmap=cmap))

                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                plt.title('%s voltages for neurons (%d - %d) from t = %d to %d ' % args)
                plt.xlabel('Time (ms)'); plt.ylabel('Neuron index')
                axes.set_aspect('auto')

        else:  # Plot each layer at a time
            for i, v in enumerate(voltages.items()):
                if plot_type == 'line':
                    ims.append(axes[i].plot(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]].cpu().numpy().T))
                    if threshold is not None:
                        ims.append(axes[i].axhline(y=threshold[v[0]], c='r', linestyle='--'))
                else:
                    ims.append(axes[i].matshow(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]], cmap=cmap))
                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                axes[i].set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % args)

            for ax in axes:
                ax.set_aspect('auto')

        plt.setp(axes, xlabel='Simulation time', ylabel='Neuron index')
        plt.tight_layout()

    else:
        # Plotting figure given
        if n_subplots == 1:  # Plotting only one image
            for v in voltages.items():
                axes.clear()
                if plot_type == 'line':
                    axes.plot(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]].cpu().numpy().T)
                    if threshold is not None:
                        axes.axhline(y=threshold[v[0]], c='r', linestyle='--')
                else:
                    axes.matshow(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]], cmap=cmap)
                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                axes.set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % args)
                axes.set_aspect('auto')

        else:
            # Plot each layer at a time
            for i, v in enumerate(voltages.items()):
                axes[i].clear()
                if plot_type == 'line':
                    axes[i].plot(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]].cpu().numpy().T)
                    if threshold is not None:
                        axes[i].axhline(y=threshold[v[0]], c='r', linestyle='--')
                else:
                    axes[i].matshow(v[1][n_neurons[v[0]][0]:n_neurons[v[0]][1], time[0]:time[1]], cmap=cmap)
                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                axes[i].set_title('%s voltages for neurons (%d - %d) from t = %d to %d ' % args)

            for ax in axes:
                ax.set_aspect('auto')

        if plot_type == 'color':
            plt.setp(axes, xlabel='Simulation time', ylabel='Neuron index')
        elif plot_type == 'line':
            plt.setp(axes, xlabel='Simulation time', ylabel='Voltage')

        plt.tight_layout()

    return ims, axes
