.. bindsnet documentation master file, created by
   sphinx-quickstart on Wed Apr 11 13:44:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BindsNET's documentation!
====================================

BindsNET is built on top of the `PyTorch <http://pytorch.org/>`_ deep learning platform. It is used for the simulation
of spiking neural networks (SNNs) and is geared towards machine learning and reinforcement learning.

BindsNET takes advantage of the :code:`torch.Tensor` object to build spiking neurons and connections between them, and
simulate them on CPUs or GPUs (for strong acceleration / parallelization) without any extra work. Recently,
:code:`torchvision.datasets` has been integrated into the library to allow the use of popular vision datasets in
training SNNs for computer vision tasks. Neural network functionality contained in :code:`torch.nn.functional` module is
used to implement more complex connections between populations of spiking neurons.

Spiking neural networks are sometimes referred to as the `third generation of neural networks
<https://www.sciencedirect.com/science/article/pii/S0893608097000117>`_. Rather than the simple linear layers and nonlinear activation functions of deep learning neural networks, SNNs are composed of neural units which more accurately capture properties of their biological counterparts. An important difference between spiking neurons and the artificial neurons of deep learning are the former's integration of input *in time*; they are naturally short-term memory devices by their maintenance of a (possibly decaying) membrane voltage. As a result, some have argued that SNNs are particularly well-suited to model time-varying data.

Neurons are connected together with directed edges (*synapses*) which are (in general) plastic. Synapses may have their own dynamics as well, which may or may not `depend on pre- and post-synaptic neural activity https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3395004/` or `other biological signals https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4717313/`. The modification of synaptic strengths is thought to be an important mechanism by which organisms learn. Accordingly, BindsNET provides a module (**bindsnet.learning**) which contains functions used for the updating of synapse weights.

At its core, BindsNET provides software objects and methods which support the simulation of groups of different types of neurons (**bindsnet.network.nodes**), as well as different types of connections between them (**bindsnet.network.topology**). These may be arbitrarily combined together under a single **bindsnet.network.Network** object, which is responsible for the coordination of the simulation logic of all underlying components. On creation of a network, the user can specify a simulation timestep constant, :math:`dt`, which determines the granularity of the simulation. Choosing this parameter induces a trade-off between simulation speed and numerical precision: large values result in fast simulation, but poor simulation accuracy, and vice versa. Monitors (**bindsnet.network.monitors**) are available for recording state variables from arbitrary network components (e.g., the voltage :math:`v` of a group of neurons). 

The development of BindsNET is supported by the Defense Advanced Research Project Agency Grant DARPA/MTO HR0011-16-l-0006.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   quickstart
   guide

.. toctree::
   :maxdepth: 2
   :caption: Package reference

   bindsnet

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
