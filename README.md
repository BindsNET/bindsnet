# SpikeTorch

Python package used for simulating spiking neural networks (SNNs) in [PyTorch](http://pytorch.org/).

At the moment, the focus is on replicating the SNN described in [Unsupervised learning of digit recognition using spike-timing-dependent plasticity](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#) (original code found [here](https://github.com/peter-u-diehl/stdp-mnist), extensions thereof found in my previous project repository [here](https://github.com/djsaunde/stdp-mnist)).

We are currently interested in applying SNNs to simple machine learning (ML) tasks, but the code can be used for any purpose.

## Requirements

All code was developed using Python 3.6.x, and will fail if run with Python 2.x. Use `pip install -r requirements.txt` to download all project dependencies. You may have to consult the [PyTorch webpage](http://pytorch.org/) in order to get the right installation for your machine. 

## Setting things up

To begin, download and unzip the MNIST dataset by running `./data/get_MNIST.sh`. To build the `spiketorch` package from source, change directory to the top level of this project and issue `pip install .` (PyPI support *hopefully* coming soon). After making changing to code in the `spiketorch` directory, issue `pip install . -U` or `pip install . --upgrade` at the top level of the project.

To replicate the SNN from the [above paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#), run `python examples/eth.py`. There are a number of optional command-line arguments which can be passed in, including `--plot` (displays useful monitoring figures), `--n_neurons [int]` (number of excitatory, inhibitory neurons simulated), `--mode ['train' | 'test']` (sets network operation to the training or testing phase), and more. Run `python code/eth.py --help` for more information on the command-line arguments.

__Note__: This is a work in progress, including the replication script `examples/eth.py` and other modifications in `examples/`.

## Background

One computational challenge is simulating time-dependent neuronal dynamics. This is typically done by solving ordinary differential equations (ODEs) which describe said dynamics. PyTorch does not explicitly support the solution of differential equations (as opposed to [`brian2`](https://github.com/brian-team/brian2), for example), but we can convert the ODEs defining the dynamics into difference equations and solve them at regular, short intervals (a `dt` on the order of 1 millisecond) as an approximation. Of course, under the hood, packages like `brian2` are doing the same thing. Doing this in [`PyTorch`](http://pytorch.org/) is exciting for a few reasons:

1. We can use the powerful and flexible [`torch.Tensor`](http://pytorch.org/) object, a wrapper around the [`numpy.ndarray`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html) which can be transferred to and from GPU devices.

2. We can avoid "reinventing the wheel" by repurposing functions from the [`torch.nn.functional`](http://pytorch.org/docs/master/nn.html#torch-nn-functional) PyTorch submodule in our SNN architectures; e.g., convolution or pooling functions.

The concept that the neuron spike ordering and their relative timing encode information is a central theme in neuroscience. [Markram et al. (1997)](http://www.caam.rice.edu/~caam415/lec_gab/g4/markram_etal98.pdf) proposed that synapses between neurons should strengthen or degrade based on this relative timing, and prior to that, [Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb) proposed the theory of Hebbian learning, often simply stated as "Neurons that fire together wire together." Markram et al.'s extension of the Hebbian theory is known as spike-timing-dependent plasticity (STDP).

We are interested in applying SNNs to machine learning problems. We use STDP to modify weights of synapses connecting pairs or populations of neurons in SNNs. In the context of ML, we want to learn a setting of synapse weights which will generate appropriate data-dependent spiking activity in SNNs. This activity will allow us to subsequently perform some ML task of interest; e.g., discriminating or clustering input data.

For now, we use the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/), which, though somewhat antiquated, is simple enough to develop new machine learning techniques on. The goal is to find a setting of synapse weights which will allow us to discriminate categories of input data. Based on historical spiking activity on training examples, we assign each neuron in an excitatory population an input category and subsequently classify test data based on these assignments.

## Contributors

- Daniel Saunders ([email](mailto:djsaunde@cs.umass.com) | [webpage](https://djsaunde.github.io))

- Hananel Hazan ([email](mailto:hhazan@cs.umass.edu))

- Darpan Sanghavi ([email](mailto:dsanghavi@cs.umass.edu))

- Hassaan Khan ([email](mailto:hqkhan@umass.edu))