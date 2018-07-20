import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from .action             import *
from ..analysis.plotting import *
from ..network.nodes     import Input
from ..network.monitors  import Monitor
from ..encoding          import bernoulli

plt.ion()

class Pipeline:
    '''
    Abstracts the interaction between network, environment (or dataset), input encoding, and environment feedback action.
    '''
    def __init__(self, network, environment, encoding=bernoulli, action_function=None, **kwargs):
        '''
        Initializes the pipeline.
        
        Inputs:
        
            | :code:`network` (:code:`bindsnet.Network`): Arbitrary network object.
            | :code:`environment` (:code:`bindsnet.Environment`): Arbitrary environment.
            | :code:`encoding` (:code:`function`): Function to encode observations into spike trains.
            | :code:`action_function` (:code:`function`): Function to convert network outputs into environment inputs.
            | :code:`kwargs`:
            
                | :code:`plot_interval` (:code:`int`): Interval to update plots.
                | :code:`save_dir` (:code:`str`): Directory to save network object to.
                | :code:`print_interval` (:code:`int`): Interval to print text output.
                | :code:`time` (:code:`int`): Time input is presented for to the network.
                | :code:`history` (:code:`int`): Number of observations to keep track of.
                | :code:`delta` (:code:`int`): Step size to save observations in history. 
                | :code:`render_interval` (:code:`bool`): Interval to render the environment.
                | :code:`save_interval` (:code:`int`): How often to save the network to disk.
                | :code:`output` (:code:`str`): String name of the layer from which to take output from.
        '''
        self.network = network
        self.env = environment
        self.encoding = encoding
        self.action_function = action_function
        
        self.iteration = 0
        self.history_index = 1
        self.s_ims, self.s_axes = None, None
        self.v_ims, self.v_axes = None, None
        self.obs_im, self.obs_ax = None, None
        
        # Setting kwargs.
        self.time = kwargs.get('time', 1)
        self.delta = kwargs.get('delta', 1)
        self.output = kwargs.get('output', None)
        self.save_dir = kwargs.get('save_dir', 'network.p')
        self.plot_interval = kwargs.get('plot_interval', None)
        self.save_interval = kwargs.get('save_interval', None)
        self.print_interval = kwargs.get('print_interval', None)
        self.history_length = kwargs.get('history_length', None)
        self.render_interval = kwargs.get('render_interval', None)
        
        if self.history_length is not None and self.delta is not None:
            self.history = {i : torch.Tensor() for i in range(1, 
                                                              self.history_length * self.delta + 1,
                                                              self.delta)}
        else:
            self.history = {}
        
        if self.plot_interval is not None:
            for l in self.network.layers:
                self.network.add_monitor(Monitor(self.network.layers[l],
                                                 's',
                                                 self.plot_interval * self.time),
                                         name=f'{l}_spikes')
                
                if 'v' in self.network.layers[l].__dict__:
                    self.network.add_monitor(Monitor(self.network.layers[l],
                                                     'v',
                                                     self.plot_interval * self.time),
                                             name=f'{l}_voltages')
            
            self.spike_record = {l : torch.ByteTensor() for l in self.network.layers}
            self.set_spike_data()
            self.plot_data()

        # Set up for multiple layers of input layers.
        self.encoded = {key: torch.Tensor() for key, val in network.layers.items() if type(val) == Input}
        
        self.first = True
        self.clock = time.time()

    def set_spike_data(self):
        '''
        Get the spike data from all layers in the pipeline's network.
        '''
        self.spike_record = {l : self.network.monitors[f'{l}_spikes'].get('s') for l in self.network.layers}

    def set_voltage_data(self):
        '''
        Get the voltage data from all applicable layers in the pipeline's network.
        '''
        self.voltage_record = {}
        for l in self.network.layers:
            if 'v' in self.network.layers[l].__dict__:
                self.voltage_record[l] = self.network.monitors[f'{l}_voltages'].get('v')

    def step(self, **kwargs):
        '''
        Run an iteration of the pipeline.
        '''
        clamp = kwargs.get('clamp', {})
        
        if self.print_interval is not None and self.iteration % self.print_interval == 0:
            print(f'Iteration: {self.iteration} (Time: {time.time() - self.clock:.4f})')
            self.clock = time.time()
        
        if self.save_interval is not None and self.iteration % self.save_interval == 0:
            print(f'Saving network to {self.save_dir}')
            self.network.save(self.save_dir)
        
        # Render game.
        if self.render_interval is not None and self.iteration % self.render_interval == 0:
            self.env.render()
            
        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            action = self.action_function(self, output=self.output)
        else:
            action = None
        
        # Run a step of the environment.
        self.obs, self.reward, self.done, info = self.env.step(action)

        # Store frame of history and encode the inputs.
        if len(self.history) > 0:
            self.update_history()
            self.update_index()
        
        # Encode the observation using given encoding function.
        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(self.obs, 
                                               time=self.time, 
                                               max_prob=self.env.max_prob)
        
        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=self.encoded,
                         time=self.time,
                         reward=self.reward,
                         clamp=clamp)
        
        # Plot relevant data.
        if self.plot_interval is not None and self.iteration % self.plot_interval == 0:
            self.plot_data()
            
            if self.iteration > len(self.history) * self.delta:  
                self.plot_obs()
            
        self.iteration += 1

    def plot_obs(self):
        '''
        Plot the processed observation after difference against history
        '''
        if self.obs_im is None and self.obs_ax is None:
            fig, self.obs_ax = plt.subplots(); self.obs_ax.set_title('Observation')
            self.obs_ax.set_xticks(()); self.obs_ax.set_yticks(())
            self.obs_im = self.obs_ax.imshow(self.env.reshape(), cmap='gray')
        else:
            self.obs_im.set_data(self.env.reshape())
    
    def plot_data(self):
        '''
        Plot desired variables.
        '''
        # Set latest data
        self.set_spike_data()
        self.set_voltage_data()
        
        # Initialize plots
        if self.s_ims is None and self.s_axes is None and self.v_ims is None and self.v_axes is None:
            self.s_ims, self.s_axes = plot_spikes(self.spike_record)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record)
        else: 
            # Update the plots dynamically
            self.s_ims, self.s_axes = plot_spikes(self.spike_record, ims=self.s_ims, axes=self.s_axes)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record, ims=self.v_ims, axes=self.v_axes)
        
        plt.pause(1e-8)
        plt.show()

    def update_history(self):
        '''
        Updates the observations inside history by performing subtraction from 
        most recent observation and the sum of previous observations.
        
        If there are not enough observations to take a difference from, simply 
        store the observation without any subtraction.
        '''
        # Recording initial observations
        if self.iteration < len(self.history) * self.delta:
            # Store observation based on delta value
            if self.iteration % self.delta == 0:
                self.history[self.history_index] = self.obs
        else:
            # Take difference between stored frames and current frame
            temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)
                    
            # Store observation based on delta value.
            if self.iteration % self.delta == 0:
                self.history[self.history_index] = self.obs
            
            assert (len(self.history) == self.history_length), 'History size is out of bounds'
            self.obs = temp
            
    def update_index(self):
        '''
        Updates the index to keep track of history.
        
        For example: history = 4, delta = 3 will produce self.history = {1, 4, 7, 10}
                          and self.history_index will be updated according to self.delta
                          and will wrap around the history dictionary.
        '''
        if self.iteration % self.delta == 0:
            if self.history_index != max(self.history.keys()):
                self.history_index += self.delta
            # Wrap around the history
            else:
                self.history_index = (self.history_index % max(self.history.keys())) + 1 
                    
    def _reset(self):
        '''
        Reset the pipeline.
        '''
        self.env.reset()
        self.network._reset()
        self.iteration = 0
        self.history = {i: torch.Tensor() for i in self.history}
