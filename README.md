<p align="center"><img width="25%" src="docs/logo.png"/></p>

A Python package used for simulating spiking neural networks (SNNs) on CPUs or GPUs using [PyTorch](http://pytorch.org/) `Tensor` functionality.

BindsNET is a spiking neural network simulation library geared towards the development of biologically inspired algorithms for machine learning.

This package is used as part of ongoing research on applying SNNs, machine learning (ML) and reinforcement learning (RL) problems in the [Biologically Inspired Neural & Dynamical Systems (BINDS) lab](http://binds.cs.umass.edu/) and the Allen Discovery Center at Tufts University.


Check out the [BindsNET examples](https://github.com/BindsNET/bindsnet/tree/master/examples) for a collection of experiments, functions for the analysis of results, plots of experiment outcomes, and more. Documentation for the package can be found [here](https://bindsnet-docs.readthedocs.io).

![Build Status](https://github.com/BindsNET/bindsnet/actions/workflows/python-app.yml/badge.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/bindsnet-docs/badge/?version=latest)](https://bindsnet-docs.readthedocs.io/?badge=latest)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/bindsnet_/community)

## Requirements

- Python >=3.9,<3.12

## Setting things up

## Using Pip
To install the most recent stable release from the GitHub repository

```
pip install git+https://github.com/BindsNET/bindsnet.git
```

Or, to build the `bindsnet` package from source, clone the GitHub repository, change directory to the top level of this project, and issue

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
docker build .
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

## Benchmarking
We simulated a network with a population of n Poisson input neurons with firing rates (in Hertz) drawn randomly from U(0, 100), connected all-to-all with a equally-sized population of leaky integrate-and-fire (LIF) neurons, with connection weights sampled from N(0,1). We varied n systematically from 250 to 10,000 in steps of 250, and ran each simulation with every library for 1,000ms with a time resolution dt = 1.0. We tested BindsNET (with CPU and GPU computation), BRIAN2, PyNEST (the Python interface to the NEST SLI interface that runs the C++NEST core simulator), ANNarchy (with CPU and GPU computation), and BRIAN2genn (the BRIAN2 front-end to the GeNN simulator).

Several packages, including BRIAN and PyNEST, allow the setting of certain global preferences; e.g., the number of CPU threads, the number of OpenMP processes, etc. We chose these settings for our benchmark study in an attempt to maximize each library's speed, but note that BindsNET requires no setting of such options. Our approach, inheriting the computational model of PyTorch, appears to make the best use of the available hardware, and therefore makes it simple for practitioners to get the best performance from their system with the least effort.

<p align="middle">
<img src="https://github.com/Hananel-Hazan/bindsnet/blob/master/docs/BindsNET%20benchmark.png" alt="BindsNET%20Benchmark"  width="503" height="403">
</p>

All simulations run on Ubuntu 16.04 LTS with Intel(R) Xeon(R) CPU E5-2687W v3 @ 3.10GHz, 128Gb RAM @ 2133MHz, and two GeForce GTX TITAN X (GM200) GPUs. Python 3.6 is used in all cases. Clock time was recorded for each simulation run. 

## Citation

If you use BindsNET in your research, please cite the following [article](https://www.frontiersin.org/article/10.3389/fninf.2018.00089):

```
@ARTICLE{10.3389/fninf.2018.00089,
	AUTHOR={Hazan, Hananel and Saunders, Daniel J. and Khan, Hassaan and Patel, Devdhar and Sanghavi, Darpan T. and Siegelmann, Hava T. and Kozma, Robert},   
	TITLE={BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python},      
	JOURNAL={Frontiers in Neuroinformatics},      
	VOLUME={12},      
	PAGES={89},     
	YEAR={2018}, 
	URL={https://www.frontiersin.org/article/10.3389/fninf.2018.00089},       
	DOI={10.3389/fninf.2018.00089},      
	ISSN={1662-5196},
}

```

## Contributors

- Hava Siegelmann - Director of BINDS lab at UMass
- Robert Kozma - Co-Director of BINDS lab (2018-2019)
- Hananel Hazan ([email](mailto:hananel@hazan.org.il)) - Spearheaded BindsNET and its main maintainer.
- Daniel Saunders ([email](mailto:djsaunde@cs.umass.edu)) - MSc student, BindsNET core functions coder (2018-2019). 
- Darpan Sanghavi ([email](mailto:dsanghavi@cs.umass.edu)) - MSc student BINDS Lab (2018)
- Hassaan Khan ([email](mailto:hqkhan@umass.edu)) - MSc student BINDS Lab (2018)
- Devdhar Patel ([email](mailto:devdharpatel@cs.umass.edu)) - MSc student BINDS Lab (2018)
- Simon Caby [github](https://github.com/SimonInParis) 
- Christopher Earl ([email](mailto:cearl@umass.edu)) - MSc student (2021 - present)


<a href="https://github.com/Bindsnet/bindsnet/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bindsnet/bindsnet" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## License
GNU Affero General Public License v3.0
