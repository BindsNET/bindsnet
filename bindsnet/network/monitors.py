import os
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Optional, Iterable, Dict

from .nodes import Nodes
from .topology import AbstractConnection


class AbstractMonitor(ABC):
    # language=rst
    """
    Abstract base class for state variable monitors.
    """

    @abstractmethod
    def __init__(self):
        # language=rst
        """
        Abstract method stub for monitor constructor.
        """
        super().__init__()

    @abstractmethod
    def get(self):
        # language=rst
        """
        Abstract method stub for retrieving monitored state recording.
        :return: State variable recording.
        """
        pass

    @abstractmethod
    def record(self):
        # language=rst
        """
        Abstract method stub for recording monitored state variables.
        """
        pass

    @abstractmethod
    def reset_(self):
        # language=rst
        """
        Abstract method stub for resetting monitored state recording.
        """
        pass


class Monitor(AbstractMonitor):
    # language=rst
    """
    Records state variables of interest.
    """
    def __init__(self, obj: Union[Nodes, AbstractConnection], state_vars: Iterable[str], time: Optional[int]=None):
        # language=rst
        """
        Constructs a ``Monitor`` object.

        :param obj: An object to record state variables from during network simulation.
        :param state_vars: Iterable of strings indicating names of state variables to record.
        :param time: If not ``None``, pre-allocate memory for state variable recording.
        """
        super().__init__()

        self.obj = obj
        self.state_vars = state_vars
        self.time = time

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            self.recording = {var: torch.Tensor() for var in self.state_vars}

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            self.recording = {var: torch.zeros(*self.obj.__dict__[var].size(), self.time) for var in self.state_vars}
            self.i = 0

    def get(self, var: str) -> torch.Tensor:
        # language=rst
        """
        Return recording to user.

        :param var: State variable recording to return.
        :return: Tensor of shape ``[n_1, ..., n_k, time]``, where ``[n_1, ..., n_k]`` is the shape of the recorded
                 state variable.
        """
        return self.recording[var]

    def record(self) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        if self.time is None:
            for v in self.state_vars:
                data = self.obj.__dict__[v].view(-1, 1).float()
                self.recording[v] = torch.cat([self.recording[v], data], -1)
        else:
            for v in self.state_vars:
                data = self.obj.__dict__[v].unsqueeze(-1)
                self.recording[v][..., self.i % self.time] = data.squeeze()

            self.i += 1

    def reset_(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``torch.Tensor``s.
        """
        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            self.recording = {v: torch.Tensor() for v in self.state_vars}

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            self.recording = {v: torch.zeros(*self.obj.__dict__[v].size(), self.time) for v in self.state_vars}
            self.i = 0


class NetworkMonitor(AbstractMonitor):
    # language=rst
    """
    Record state variables of all layers and connections.
    """
    def __init__(self, network: 'Network', layers: Optional[Iterable[str]]=None,
                 connections: Optional[Iterable[str]]=None, state_vars: Optional[Iterable[str]]=None,
                 time: Optional[int]=None):
        # language=rst
        """
        Constructs a ``NetworkMonitor`` object.

        :param network: Network to record state variables from.
        :param layers: Layers to record state variables from.
        :param connections: Connections to record state variables from.
        :param state_vars: List of strings indicating names of state variables to record.
        :param time: If not ``None``, pre-allocate memory for state variable recording.
        """
        super().__init__()

        self.network = network
        self.layers = layers if layers is not None else list(self.network.layers.keys())
        self.connections = connections if connections is not None else list(self.network.connections.keys())
        self.state_vars = state_vars if state_vars is not None else ('v', 's', 'w')
        self.time = time

        if self.time is not None:
            self.i = 0

        # Initialize empty recording.
        self.recording = {k : {} for k in self.layers + self.connections}

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if v in self.network.layers[l].__dict__:
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if v in self.network.connections[c].__dict__:
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if v in self.network.layers[l].__dict__:
                        self.recording[l][v] = torch.zeros(*self.network.layers[l].__dict__[v].size(), self.time)

                for c in self.connections:
                    if v in self.network.connections[c].__dict__:
                        self.recording[c][v] = torch.zeros(*self.network.connections[c].__dict__[v].size(), self.time)

    def get(self) -> Dict[str, Dict[str, Union[Nodes, AbstractConnection]]]:
        # language=rst
        """
        Return entire recording to user.

        :return: Dictionary of dictionary of all layers' and connections' recorded state variables.
        """
        return self.recording

    def record(self) -> None:
        # language=rst
        """
        Appends the current value of the recorded state variables to the recording.
        """
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if v in self.network.layers[l].__dict__:
                        data = self.network.layers[l].__dict__[v].unsqueeze(-1).float()
                        self.recording[l][v] = torch.cat([self.recording[l][v], data], -1)

                for c in self.connections:
                    if v in self.network.connections[c].__dict__:
                        data = self.network.connections[c].__dict__[v].unsqueeze(-1)
                        self.recording[c][v] = torch.cat([self.recording[c][v],  data], -1)

        else:
            for v in self.state_vars:
                for l in self.layers:
                    if v in self.network.layers[l].__dict__:
                        data = self.network.layers[l].__dict__[v].float()
                        self.recording[l][v][..., self.i % self.time] = data

                for c in self.connections:
                    if v in self.network.connections[c].__dict__:
                        data = self.network.connections[c].__dict__[v]
                        self.recording[c][v][..., self.i % self.time] = data

            self.i += 1

    def save(self, path: str, fmt: str='npz') -> None:
        # language=rst
        """
        Write the recording dictionary out to file.

        :param path: The directory to which to write the monitor's recording.
        :param fmt: Type of file to write to disk. One of ``"pickle"`` or ``"npz"``.
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if fmt == 'npz':
            # Build a list of arrays to write to disk.
            arrays = {}
            for o in self.recording:
                if type(o) == tuple:
                    arrays.update({'_'.join(['-'.join(o), v]): self.recording[o][v] for v in self.recording[o]})
                elif type(o) == str:
                    arrays.update({'_'.join([o, v]): self.recording[o][v] for v in self.recording[o]})

            np.savez_compressed(path, **arrays)

        elif fmt == 'pickle':
            with open(path, 'wb') as f:
                torch.save(self.recording, f)

    def reset_(self) -> None:
        # language=rst
        """
        Resets recordings to empty ``torch.Tensors``.
        """
        # Reset to empty recordings
        self.recording = {k: {} for k in self.layers + self.connections}

        if self.time is not None:
            self.i = 0

        # If no simulation time is specified, specify 0-dimensional recordings.
        if self.time is None:
            for v in self.state_vars:
                for l in self.layers:
                    if v in self.network.layers[l].__dict__:
                        self.recording[l][v] = torch.Tensor()

                for c in self.connections:
                    if v in self.network.connections[c].__dict__:
                        self.recording[c][v] = torch.Tensor()

        # If simulation time is specified, pre-allocate recordings in memory for speed.
        else:
            for v in self.state_vars:
                for l in self.layers:
                    if v in self.network.layers[l].__dict__:
                        self.recording[l][v] = torch.zeros(self.network.layers[l].n, self.time)

                for c in self.connections:
                    if v in self.network.connections[c].__dict__:
                        self.recording[c][v] = torch.zeros(*self.network.connections[c].w.size(), self.time)
