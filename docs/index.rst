.. bindsnet documentation master file, created by
   sphinx-quickstart on Wed Apr 11 13:44:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BindsNET's documentation!
====================================

BindsNET is built on top of the `PyTorch <http://pytorch.org/>`_ tensors and dynamic neural networks library. It is used for simulation spiking neural networks (SNNs) and is specifically geared towards machine learning and reinforcement learning with SNNs. BindsNET takes advantage of the **torch.Tensor** object to build generate network structures and simulate them on CPUs or GPUs (for strong acceleration) without any extra engineering. Neural network functionality such as that of the **torch.nn.functional** module may be useful in the future for quickly building more complex network structures.

Spiking neural networks are sometimes referred to as the `third generation of neural networks <https://www.sciencedirect.com/science/article/pii/S0893608097000117>`_. Rather than the simple linear layers and nonlinear activation functions of deep learning, SNNs contain neural units which more accurately capture properties of the biological brain. A simple spiking neuron model is defined by a leaky integrator ODE which represents its membrane voltage potential (exponentially decaying to a rest potential :math:`v_{\textrm{rest}}`) and a firing threshold :math:`v_{\textrm{thresh}}`; i.e., the voltage at which it releases an action potential (spike) and resets back to a starting value :math:`v_{\textrm{reset}}`. At this point, the neuron may undergo a refractory period, during which time it either 1) cannot emit a spike, or 2) does not integrate its input. Neurons are connected together via directed edges known as synapses which are, in general, modifiable. The modification of synapses in the biological brain is thought to be a central mechanism by which creatures learn.

At its core, BindsNET provides software objects and methods which support the simulation of groups of different types of neurons (**bindsnet.network.Nodes**), as well as different types of connections between them (**bindsnet.network.Connections**). These may be arbitrarily combined together under a single **bindsnet.network.Network** object, which is responsible for the organization of the simulation logic of all underlying components. On creation of the network, the user can specify a simulation timestep constant, :math:`dt`, on which all other components' time constants depend. Monitor objects are available for recording state variables from arbitrary network objects (e.g., the instantaneous voltage :math:`v` of a group of neurons). 

The development of BindsNET is supported by the Defense Advanced Research Project Agency Grant DARPA/MTO HR0011-16-l-0006.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
