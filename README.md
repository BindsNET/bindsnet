<p align="center"><img width="25%" src="docs/logo.png"/></p>

A Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using [PyTorch](http://pytorch.org/) `Tensor` functionality.

BindsNET is a spiking neural network simulation library geared towards the development of biologically inspired algorithms for machine learning.

This package is used as part of ongoing research on applying SNNs to machine learning (ML) and reinforcement learning (RL) problems in the [Biologically Inspired Neural & Dynamical Systems (BINDS) lab](http://binds.cs.umass.edu/).

[![Build Status](https://travis-ci.com/Hananel-Hazan/bindsnet.svg?token=trym5Uzx1rs9Ez2yENEF&branch=master)](https://travis-ci.com/Hananel-Hazan/bindsnet)
[![Documentation Status](https://readthedocs.org/projects/bindsnet-docs/badge/?version=latest)](https://bindsnet-docs.readthedocs.io/?badge=latest)

## Requirements

- Python 3.6
- `torch`
- `numpy`
- `matplotlib`
- `scikit_image`
- `opencv-python`
- `gym` (optional)

## Setting things up

### Using pip
BindsNET is available on PyPI. Issue

```
pip install bindsnet
```

to get the most recent stable release. Or, to build the `bindsnet` package from source, clone the GitHub repository, change directory to the top level of this project, and issue

```
pip install .
```

Or, to install in editable mode (allows modification of package without re-installing):

```
pip install -e .
```

To install the packages necessary to interface with the [OpenAI gym RL environments library](https://github.com/openai/gym), follow their instructions for installing the packages needed to run the RL environments simulator (on Linux / MacOS).

### Using Docker
[Link](https://hub.docker.com/r/hqkhan/bindsnet/) to Docker repository.

We also provide a Dockerfile in which BindsNET and all of its dependencies come installed in. Issue

```
docker image build .
```
at the top level directory of this project to create a docker image. 

To change the name of the newly built image, issue
```
docker tag <IMAGE_ID> <NEW_IMAGE_ID>
```

To run a container and get a bash terminal inside it, issue

```
docker run -it <NEW_IMAGE_ID> bash
```

## Getting started

To run a near-replication of the SNN from [this paper](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full#), issue

```
cd examples/mnist
python eth_mnist.py
```

There are a number of optional command-line arguments which can be passed in, including `--plot` (displays useful monitoring figures), `--n_neurons [int]` (number of excitatory, inhibitory neurons simulated), `--mode ['train' | 'test']` (sets network operation to the training or testing phase), and more. Run the script with the `--help` or `-h` flag for more information.

A number of other examples are available in the `examples` directory that are meant to showcase BindsNET's functionality. Take a look, and let us know what you think!

## Running the tests

Issue the following to run the tests:

```
python -m pytest test/
```

Some tests will fail if Open AI `gym` is not installed on your machine.

## Background

The simulation of biologically plausible spiking neuron dynamics can be challenging. It is typically done by solving ordinary differential equations (ODEs) which describe said dynamics. PyTorch does not explicitly support the solution of differential equations (as opposed to [`brian2`](https://github.com/brian-team/brian2), for example), but we can convert the ODEs defining the dynamics into difference equations and solve them at regular, short intervals (a `dt` on the order of 1 millisecond) as an approximation. Of course, under the hood, packages like `brian2` are doing the same thing. Doing this in [`PyTorch`](http://pytorch.org/) is exciting for a few reasons:

1. We can use the powerful and flexible [`torch.Tensor`](http://pytorch.org/) object, a wrapper around the [`numpy.ndarray`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html) which can be transferred to and from GPU devices.

2. We can avoid "reinventing the wheel" by repurposing functions from the [`torch.nn.functional`](http://pytorch.org/docs/master/nn.html#torch-nn-functional) PyTorch submodule in our SNN architectures; e.g., convolution or pooling functions.

The concept that the neuron spike ordering and their relative timing encode information is a central theme in neuroscience. [Markram et al. (1997)](http://www.caam.rice.edu/~caam415/lec_gab/g4/markram_etal98.pdf) proposed that synapses between neurons should strengthen or degrade based on this relative timing, and prior to that, [Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb) proposed the theory of Hebbian learning, often simply stated as "Neurons that fire together, wire together." Markram et al.'s extension of the Hebbian theory is known as spike-timing-dependent plasticity (STDP).

We are interested in applying SNNs to ML and RL problems. We use STDP to modify weights of synapses connecting pairs or populations of neurons in SNNs. In the context of ML, we want to learn a setting of synapse weights which will generate data-dependent spiking activity in SNNs. This activity will allow us to subsequently perform some ML task of interest; e.g., discriminating or clustering input data. In the context of RL, we may think of the spiking neural network as an RL agent, whose spiking activity may be converted into actions in an environment's action space.

We have provided some simple starter scripts for doing unsupervised learning (learning a fully-connected or convolutional representation via STDP), supervised learning (clamping output neurons to desired spiking behavior depending on data labels), and reinforcement learning (converting observations from the Atari game Space Invaders to input to an SNN, and converting network activity back to actions in the game).

## Citation

If you use BindsNET in your research, please cite the following [article](https://arxiv.org/abs/1806.01423):

```
@ARTICLE{2018arXiv180601423H,
   author = {{Hazan}, H. and {Saunders}, D.~J. and {Khan}, H. and {Sanghavi}, D.~T. and 
	{Siegelmann}, H.~T. and {Kozma}, R.},
    title = "{BindsNET: A machine learning-oriented spiking neural networks library in Python}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1806.01423},
 keywords = {Computer Science - Neural and Evolutionary Computing, Quantitative Biology - Neurons and Cognition},
     year = 2018,
    month = jun,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180601423H},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contributors

- Daniel Saunders ([email](mailto:djsaunde@cs.umass.edu))

- Hananel Hazan ([email](mailto:hananel@hazan.org.il))

- Darpan Sanghavi ([email](mailto:dsanghavi@cs.umass.edu))

- Hassaan Khan ([email](mailto:hqkhan@umass.edu))

## License
GNU Affero General Public License v3.0


