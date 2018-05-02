.. bindsnet documentation master file, created by
   sphinx-quickstart on Wed Apr 11 13:44:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BindsNET's documentation!
====================================

BindsNET is built on top of the `PyTorch <http://pytorch.org/>`_ neural networks library. It is used for the simulation of spiking neural networks (SNNs) and is geared towards machine learning and reinforcement learning with SNNs. BindsNET takes advantage of the **torch.Tensor** object to build generic network structures and simulate them on CPUs or GPUs (for strong acceleration / parallelization) without any extra engineering. Neural network functionality such as that of the **torch.nn.functional** module may be useful in the future for quickly building more complex network structures.

Spiking neural networks are sometimes referred to as the `third generation of neural networks <https://www.sciencedirect.com/science/article/pii/S0893608097000117>`_. Rather than the simple linear layers and nonlinear activation functions of deep learning neural networks, SNNs are composed of neural units which more accurately capture properties of the biological brain. A simple spiking neuron model is defined by a leaky integrator ODE which represents its membrane voltage potential (exponentially decaying to a rest potential :math:`v_{\textrm{rest}}`) and a firing threshold :math:`v_{\textrm{thresh}}`; i.e., the voltage at which it releases an action potential (spike) and resets back to a starting value :math:`v_{\textrm{reset}}`. At this point, the neuron may undergo a refractory period, during which time it either 1) cannot emit a spike, or 2) does not integrate its input. An important difference between spiking neurons and deep learning neural network units are their integration of input *in time*; they are naturally short-term memory devices by their maintenance of a (leaky) membrane voltage.

Neurons are connected together via directed edges known as synapses which are, in general, modifiable. Synapses may have their own dynamics as well, which may or may not depend on pre- and post-synaptic neural activity or other biological signals. The modification of synaptic strengths in the biological brain is thought to be a central mechanism by which organisms learn. To this end, BindsNET provides a module **bindsnet.learning** which contains functions used for the updating of synaptic strengths and / or supporting state variables.

At its core, BindsNET provides software objects and methods which support the simulation of groups of different types of neurons (**bindsnet.network.nodes**), as well as different types of connections between them (**bindsnet.network.topology**). These may be arbitrarily combined together under a single **bindsnet.network.Network** object, which is responsible for the organization of the simulation logic of all underlying components. On creation of the network, the user can specify a simulation timestep constant, :math:`dt`, on which all other components' time constants depend. Monitor objects are available for recording state variables from arbitrary network objects (e.g., the instantaneous voltage :math:`v` of a group of neurons). 

The development of BindsNET is supported by the Defense Advanced Research Project Agency Grant DARPA/MTO HR0011-16-l-0006.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Package reference

   bindsnet

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
