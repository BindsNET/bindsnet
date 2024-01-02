# **BindsNet examples**
The examples are shortly described and main parameters are given.
***
## MNIST examples
*/examples/mnist/*  
This directory contains different spiking network structures to train on the MNIST dataset (70.000 handwritten samples). In all these examples, the (28, 28) pixels are frequency-encoded using Poisson distribution into firing rates.

```python eth_mnist.py```: performs a near-replication of Diehl & Cook 2015 (https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full).
Two layers with lateral inhibition, unsupervised STDP.  
>*Approx running time of 1 hour on intel x86 i7.*

```batch_eth_mnist.py```: uses the same network as eth_mnist.py, but performs multiple parallel trainings on a GPU, and ensembles the results for faster learning (https://arxiv.org/pdf/1909.02549.pdf).  
>*Training can be as fast as 1 minute/epoch on GPU (1080ti)*

|training parameters |default|description|
|-|:-:|-|
|--n_epochs|1|     number of epochs for training
|--gpu|False| enables running on GPU (CUDA)
|--plot|False| graphically monitors voltages, spikes, output neurons assignments and confusion matrix
|--time|100|duration (in ms) of each sample presentation
|--dt|1| SNN simulation time increment (ms)
|--update_step|256|number of samples trained between each accuracy estimation
|--seed|0|initial random seed
|--n_workers|-1|dataset spikes conversion CPU/GPU threads (-1 for auto)
|--batch_size|32|only for batch_eth_mnist.py: length of each training batch

|model parameters|default|description
|-|:-:|-|
|--n_neurons|100|number of neurons in both excitatory and inhibitory layers
|--intensity|128.0|input layer Poisson spikes maximum firing rate, in Hz
|--exc|22.5|strength of synapse weights from excitatory to inhibitory layer
|--inh|120.0|strength of synapse weights from inhibitory to excitatory layer
|--theta_plus|0.05|membrane threshold potential increment at each spike


##### Remarks:
* Rising the --n_neurons parameter yields to higher accuracy.
* Rising –-batch_size can drastically reduce learning time, but may lower final accuracy, STDP being averaged through the batch and performed only at the end of each batch.


```reservoir.py```: **Training MNIST dataset using a reservoir computing paradigm.**  
The reservoir is a liquid state machine (LSM) composed of a spiking network (SNN), read out by a simple linear neural model. The SNN structure is a two-layer input-output, with an additional recurrent connection on the output. The LSM output then fed to a perceptron to be classified.  
MNIST data are first Poisson-converted and fed to the input layer of the SNN.
Output from the SNN is used to train the 1-layer linear neural network, which learns the (higher dimensional) representations of the MNIST.

|model parameters|default|description
|-|:-:|-|
|--n_neurons|500|number of neurons in output layer of SNN

|training parameters|default|description
|-|:-:|-|
|--n_epochs|100| number of epochs for training
|--gpu|False|enables running on GPU (CUDA)
|--examples|500|number of MNIST samples used for learn SNN outputs, by epoch
|--plot|False|graphically monitors voltages, spikes, synapses weights matrix
|--time|250|duration (in ms) of each sample presentation
|--dt|1|simulation time increment (ms)
|--seed|0|initial random seed
|--n_workers|-1|dataset spikes conversion CPU/GPU threads (-1 for auto)


```SOM_LM-SNN.py``` ***Improving upon Diehl & Cook 2015***  
(https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full), with two layers with lateral inhibition, unsupervised STDP, and a Self-Organizing Maps (SOM) property by Teuvo Kohonen (https://link.springer.com/article/10.1007/BF00317973). This example achieves better accuracy than Diehl & Cook, with additional of clustering the digits by shape similarity. 

|model parameters|default|description
|-|:-:|-|
|--theta_plus|0.05|membrane threshold potential increment at each spike


|training parameters|default|description
|-|:-:|-|
|--seed|0| initial random seed
|--n_neurons |100| number of neurons in both excitatory and inhibitory layers
|--n_epochs |1| number of epochs for training
|--n_tests |10000| number of test examples to use
|--n_workers |-1| dataset spikes conversion CPU/GPU threads (-1 for auto)
|--time |100| duration (in ms) of each sample presentation
|--dt |1| SNN simulation time increment (ms)
|--intensity |64| input layer Poisson spikes maximum firing rate, in Hz
|--progress_interval |64| number of epochs to update progress
|--update_interval |250| number of iterations to update the graphs
|--update_inhibition_weights|500| update the strength of the inhibition layer. 
|--plot_interval|250| number of iterations to update plots 
|--gpu |False| enables running on GPU (CUDA)
|--plot |False| graphically monitors voltages, spikes, output neurons assignments and confusion matrix
***
## BreakOut examples
*/examples/breakout/*  
A reinforcement learning example.

**OpenAI's Gym environments** (https://gym.openai.com/) provide a standard benchmark for reinforcement learning algorithms. The video game used in this example is a simple **breakout** game (https://gym.openai.com/envs/Breakout-v0/), where the spiking network (SNN) tries to maximize its game score.

Here, we connect the _(80x80 gray scale)_ output 'screen pixels' from the emulated video game _(Atari 2600)_ to the SNN. Input of the SNN is a Bernoulli spike encoder, a middle layer of 100 leaky-fire-and-integrate (LIF) neurons. Output is a 4 neurons layer, simulating 4 possible actions on a simulated joystick.

>SNN are believed to perform as good as standard NN, providing enhanced robustness to noise and unseen patterns: (https://www.sciencedirect.com/science/article/pii/S0893608019302266?via%3Dihub)

You can experiment with your own hyper-parameters, learning rules or network structures, directly from within this simple Python code.
***


```random_baseline.py``` shows how to setup the Gym Environment, choses a video game and interacts with it.  It provides a baseline score, based on pure random actions.  
**Parameters:** None


```random_network_baseline.py``` same as above, but a complete pipeline (EnvironmentPipeline) is built, connecting the video game Gym Environment to a SNN through an spike encoder (Bernoulli), and connecting the SNN decisions to the videogame inputs (simulated joystick), therefore providing an end-to-end training pipeline. The pipeline is run for '--n' rounds. It shows the changes in the connections between the SNN and the videogame as the game runs.

parameters|default|description
-|:-:|-
--n_neurons |100| number of (LIF) neurons in the middle (excitatory) layer
--n |1000000| number of pipeline runs to perform
--plot_interval |10|nb rounds between each display of SNN connections changes
--render_interval |10|nb rounds between each display of SNN connections changes
--print_interval |100| nb rounds between each display of SNN connections changes
--seed |0| random seed initializer for SNN weights
--dt |1| SNN simulation time increment (ms)


```breakout.py``` same as above, just using Izhikevich neurons and the game is run for 100 episodes (new-game to game-over).
**Parameters**: None

```breakout_stdp.py``` same as above, with 'Reward modulated STDP' learning rule between middle and output layers (https://florian.io/papers/2007_Florian_Modulated_STDP.pdf). The SNN is trained for 100 episodes, then evaluated for another 100 episodes.
**Parameters**: None

***

## Tensorboard example
*/examples/tensorboard/*  
Google's Tensorboard is a powerful tool to analyze Deep Learning models. It helps visualizing data flows, or any changes happening during a training process.  
First developed for Google's Tensorflow, it is now available as **TensorboardX** (https://tensorboardx.readthedocs.io/en/latest/index.html) for Py-Torch or other DL frameworks *(under development)*.

```tensorboard.py``` shows how to use the ```TensorboardAnalyzer``` class, graphically monitoring the weights of 2D convolutional SNN during its training process.

>The example also shows how to do the same using the MatplotlibAnalyzer class, outputting the graphs rendered by Matplotlib.

parameters|default|description
-|:-:|-
--dataset |MNIST| selects the dataset ton train on: **"MNIST", "KMNIST", "FashionMNIST", "CIFAR10", "CIFAR100"**
--time |50| duration (in ms) of each sample presentation
--dt |1| SNN simulation time increment (in ms)
--seed |0| initial random seed
--tensorboard |True| whether to use **Tensorboard** or **Matplotlib** analyzer as output

***

## Dot Tracing example
*/examples/dotTracing/*  
dot_tracing.py trains a basic RNN on the Dot Simulator environment and demonstrates how to record reward and performance data (if desired) and plot spiking activity via monitors.

See the environments directory for documentation on the Dot Simulator.

parameters|default|description
-|:-:|-
--steps |100| number of timesteps in an episode
--dim |28| square dimensions of grid (actual simulator can specify rows vs columns)
--dt |1| SNN simulation time increment (in ms)
--seed |0| initial random seed
--granularity |100| spike train granularity
--trn_eps |1000| training episodes
--tst_eps |100| test episodes
--decay |4| length of decaying trail behind target dot
--diag |False| enables diagonal movements
--randr |.15| determines rate of randomization of movement
--boundh |'bounce'| bounds handling mode
--fit_func |'dir'| fitness function, defaulted to directional
--allow_stay |False| disable option for targets to remain in place
--pandas |False| true = pandas Dataframe printout; false = heatmap
--mute |False| prohibit graphical rendering
--write |True| save observed grids to file
--fcycle |100| number of episodes per save file
--gpu |True| Utilize cuda
--herrs |0| number of distraction dots
