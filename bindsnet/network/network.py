import tempfile
from typing import Dict, Iterable, Optional, Type, Any

import torch
from numpy import dtype
from torch import Tensor

from bindsnet.learning.reward import AbstractReward
from bindsnet.network.monitors import AbstractMonitor
from bindsnet.network.nodes import CSRMNodes, Nodes
from bindsnet.network.topology import AbstractConnection, AbstractMulticompartmentConnection


def load(file_name: str, map_location: str = "cpu", learning: bool = None) -> "Network":
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    :param map_location: One of ``"cpu"`` or ``"cuda"``. Defaults to ``"cpu"``.
    :param learning: Whether to load with learning enabled. Default loads value from
        disk.
    """
    network = torch.load(
        open(file_name, "rb"), map_location=map_location, weights_only=False
    )
    if learning is not None and "learning" in vars(network):
        network.learning = learning

    return network


class Network(torch.nn.Module):
    # language=rst
    """
    Central object of the ``bindsnet`` package. Responsible for the simulation and
    interaction of nodes and connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inputs = {'X' : train}  # Create inputs mapping.
        network.run(inputs=inputs, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(
        self,
        dt: float = 1.0,
        batch_size: int = 1,
        learning: bool = True,
        reward_fn: Optional[Type[AbstractReward]] = None,
    ) -> None:
        # language=rst
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        :param reward_fn: Optional class allowing for modification of reward in case of
            reward-modulated learning.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        if reward_fn is not None:
            self.reward_fn = reward_fn()
        else:
            self.reward_fn = None

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(
        self, connection: AbstractConnection | AbstractMulticompartmentConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.serialization.add_safe_globals([self])
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.

        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            if c[1] in layers:
                # Fetch source and target populations.
                source = self.connections[c].source
                target = self.connections[c].target

                if not c[1] in inputs:
                    if isinstance(target, CSRMNodes):
                        inputs[c[1]] = torch.zeros(
                            self.batch_size,
                            target.res_window_size,
                            *target.shape,
                            device=target.s.device,
                        )
                    else:
                        inputs[c[1]] = torch.zeros(
                            self.batch_size, *target.shape, device=target.s.device
                        )

                # Add to input: source's spikes multiplied by connection weights.
                if isinstance(target, CSRMNodes):
                    inputs[c[1]] += self.connections[c].compute_window(source.s)
                else:
                    inputs[c[1]] += self.connections[c].compute(source.s)

        return inputs

    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        :param Bool progress_bar: Show a progress bar while running the network.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Check input type
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Run synapse updates.
        if "a_minus" in kwargs:
            A_Minus = kwargs["a_minus"]
            kwargs.pop("a_minus")
            if isinstance(A_Minus, dict):
                A_MD = True
            else:
                A_MD = False
        else:
            A_Minus = None

        if "a_plus" in kwargs:
            A_Plus = kwargs["a_plus"]
            kwargs.pop("a_plus")
            if isinstance(A_Plus, dict):
                A_PD = True
            else:
                A_PD = False
        else:
            A_Plus = None

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(
                        x=torch.zeros(
                            self.layers[l].s.shape, device=self.layers[l].s.device
                        )
                    )

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0

            for c in self.connections:
                flad_m = False
                if A_Minus != None and ((isinstance(A_Minus, float)) or (c in A_Minus)):
                    if A_MD:
                        kwargs["a_minus"] = A_Minus[c]
                    else:
                        kwargs["a_minus"] = A_Minus
                    flad_m = True

                flad_p = False
                if A_Plus != None and ((isinstance(A_Plus, float)) or (c in A_Plus)):
                    if A_PD:
                        kwargs["a_plus"] = A_Plus[c]
                    else:
                        kwargs["a_plus"] = A_Plus
                    flad_p = True

                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )
                if flad_m:
                    kwargs.pop("a_minus")
                if flad_p:
                    kwargs.pop("a_plus")

            # # Get input to all layers.
            # current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        # language=rst
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)


# ---------------------------
# Shaders
# ---------------------------
vertex_shader_src = """
#version 330 core

layout(location = 0) in float spike;

uniform float time_x;
uniform float neuron_count;

void main()
{
    // Convert neuron index to y-position
    float y = (float(gl_VertexID) / neuron_count) * 2.0 - 1.0;

    // Current time position
    float x = time_x;

    // Hide inactive neurons
    if (spike < 0.95)
    {
        gl_Position = vec4(-10.0, -10.0, 0.0, 1.0);
    }
    else
    {
        gl_Position = vec4(x, y, 0.0, 1.0);
    }

    gl_PointSize = 2.0;
}
"""
fragment_shader_src = """
#version 330 core

out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 0.0, 1.0, 1.0);
}
"""

import glfw
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
from cuda.bindings import driver
from cuda.bindings import runtime
import cupy as cp
import warnings
import numpy as np
pytorch_cp_type_map = {
    torch.float32: cp.float32,
    torch.float64: cp.float64,
    torch.int32: cp.int32,
    torch.int64: cp.int64,
    torch.uint8: cp.uint8,
    torch.bool: cp.bool_,
}
pytorch_opengl_type_map = {
    torch.float32: gl.GL_FLOAT,
    torch.float64: gl.GL_DOUBLE,
    torch.int32: gl.GL_INT,
    torch.uint8: gl.GL_UNSIGNED_BYTE,
    torch.bool: gl.GL_UNSIGNED_BYTE,
}
class GUINetwork(Network):
    # language=rst
    """
    Subclass of ``Network`` with added functionality for live plotting using VisPy.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        ### Setup GUI window ###
        glfw.init()
        self.window = glfw.create_window(800, 600, "Network GUI", None, None)
        glfw.make_context_current(self.window)

        # OpenGL buffers for plotting
        # Connections: Values of topology features
        # Layers: Spikes & voltages
        self.opengl_vaos = {'connections': {}, 'layers': {}}
        self.opengl_vao_dtypes = {}

        ### Initialize CUDA driver ###
        err, = driver.cuInit(0)  # Initialize CUDA driver
        if err != 0: raise RuntimeError(f"Failed to initialize CUDA: error code {err}")
        err, self.cuda_device = driver.cuDeviceGet(0)  # Get CUDA device
        if err != 0: raise RuntimeError(f"Failed to get CUDA device: error code {err}")
        err, self.cuda_context = driver.cuCtxCreate(None, 0, self.cuda_device)  # Create CUDA context
        if err != 0: raise RuntimeError(f"Failed to create CUDA context: error code {err}")

        ### Compile shaders & create OpengGL program ###
        # Spike Raster Plots
        raster_plot_vertex_shader_src = """
        #version 330 core

        layout(location = 0) in uint spike;

        uniform float time_x;
        uniform float neuron_count;

        void main()
        {
            // Convert neuron index to y-position
            float y = (float(gl_VertexID) / neuron_count) * 2.0 - 1.0;

            // Current time position
            float x = time_x;

            // Hide inactive neurons
            if (spike == uint(0))
            {
                gl_Position = vec4(-10.0, -10.0, 0.0, 1.0);
            }
            else
            {
                gl_Position = vec4(x, y, 0.0, 1.0);
            }

            gl_PointSize = 2.0;
        }
        """
        raster_plot_fragment_shader_src = """
        #version 330 core

        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(1.0, 0.0, 1.0, 1.0);
        }
        """
        self.raster_plot_program = compileProgram(
            compileShader(raster_plot_vertex_shader_src, gl.GL_VERTEX_SHADER),
            compileShader(raster_plot_fragment_shader_src, gl.GL_FRAGMENT_SHADER)
        )
        self.max_time_steps = 100       # Max time-steps (value on x-axis) plotted
        # vs = self._compile_shader(raster_plot_vertex_shader_src, gl.GL_VERTEX_SHADER)
        # fs = self._compile_shader(raster_plot_fragment_shader_src, gl.GL_FRAGMENT_SHADER)
        # self.program = gl.glCreateProgram()
        # gl.glAttachShader(self.program, vs)
        # gl.glAttachShader(self.program, fs)
        # gl.glLinkProgram(self.program)
        # gl.glDeleteShader(vs)
        # gl.glDeleteShader(fs)
        # self.x_pointer = gl.glGetUniformLocation(self.program, "x")

    # def _compile_shader(self, src, shader_type):
    #     shader = gl.glCreateShader(shader_type)
    #     gl.glShaderSource(shader, src)
    #     gl.glCompileShader(shader)
    #     if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
    #         raise RuntimeError(gl.glGetShaderInfoLog(shader))
    #     return shader

    def add_layer(self, layer: Nodes, name: str) -> None:
        super().add_layer(layer, name)

        ### Determine which data needs a shared buffer ###
        layer_data = {}
        if isinstance(layer, Input):
            layer_data['s'] = layer.s
        elif isinstance(layer, LIFNodes):
            layer_data['s'] = layer.s
            layer_data['v'] = layer.v
        else:
            raise NotImplementedError("GUINetwork only supports Input and LIFNodes layers for now.")

        ### Create shared buffers ###
        self.opengl_vaos['layers'][name] = {}
        for data_name, data in layer_data.items():
            shared_buffer, vao = self._create_shared_buffer(data)           # Generate buffer
            layer.__setattr__(data_name, shared_buffer)                     # Replace original tensor with shared buffer
            self.opengl_vaos['layers'][name][data_name] = vao            # Map VBO to layer attribute
            self.opengl_vao_dtypes[vao] = pytorch_opengl_type_map[data.dtype]    # Store OpenGL type for this buffer

    def _create_shared_buffer(self, org_tensor: torch.Tensor) -> tuple[Tensor, int]:
        # language=rst
        """
        Create a shared buffer for a class variable tensor/buffer.

        :param org_tensor: PyTorch tensor to create a shared buffer for.
        :return:
            ``shared_buffer``: New PyTorch tensor that shares memory with an OpenGL buffer registered with CUDA
            ``vao``: OpenGL buffer object ID that is shared with the new PyTorch tensor.
        """

        N = org_tensor.numel()
        buffer_size = N * org_tensor.element_size()

        ### Setup OpenGL buffer ###
        vbo = gl.glGenBuffers(1)  # Vertex Buffer Object
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)  # Bind to GL_ARRAY_BUFFER
        gl.glBufferData(target=gl.GL_ARRAY_BUFFER,  # Allocate buffer space
                        size=buffer_size,  # Size in bytes
                        data=None,  # No initial data
                        usage=gl.GL_DYNAMIC_DRAW)  # Frequent updates expected
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)  # Unbind buffer
        if gl.glIsBuffer(vbo) == 0:
            raise RuntimeError("Failed to create OpenGL buffer")

        ### Register OpenGL buffer with CUDA ###
        err, = driver.cuInit(0)  # Initialize CUDA driver
        if err != 0: raise RuntimeError(f"Failed to initialize CUDA: error code {err}")

        err, device = driver.cuDeviceGet(0)  # Get CUDA device
        if err != 0: raise RuntimeError(f"Failed to get CUDA device: error code {err}")

        err, context = driver.cuCtxCreate(None, 0, device)  # Create CUDA context
        if err != 0: raise RuntimeError(f"Failed to create CUDA context: error code {err}")

        err, cuda_resource = driver.cuGraphicsGLRegisterBuffer(
            buffer=vbo,
            Flags=2  # cuda.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        )
        if err != 0: raise RuntimeError(f"Failed to register OpenGL buffer with CUDA: error code {err}")

        err, = driver.cuGraphicsMapResources(1, cuda_resource, 0)
        if err != 0: raise RuntimeError(f"Failed to map CUDA graphics resource: error code {err}")

        err, cuda, size = driver.cuGraphicsResourceGetMappedPointer(cuda_resource)
        if err != 0: raise RuntimeError(f"Failed to get mapped pointer for CUDA graphics resource: error code {err}")

        ### Define VAO ###
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 1, pytorch_opengl_type_map[org_tensor.dtype], False, 0, None)
        gl.glBindVertexArray(0)

        ### Create PyTorch tensor from CUDA pointer ###
        cp_ptr = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(int(cuda), size, cuda_resource), 0)
        dtype = pytorch_cp_type_map[org_tensor.dtype]
        cp_array = cp.ndarray(N, dtype=dtype, memptr=cp_ptr)
        torch_tensor = torch.as_tensor(cp_array)                # Create tensor with shared memory location
        torch_tensor = torch_tensor.reshape(org_tensor.shape)   # Reshape to original tensor shape
        torch_tensor.copy_(org_tensor)  # Copy original tensor values to shared buffer

        return torch_tensor, vao

    def run(self, inputs: Dict[str, torch.Tensor], time: int, **kwargs) -> None:
        # Check input type
        assert type(inputs) == dict, (
                "'inputs' must be a dict of names of layers "
                + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )

        ### Compute reward ###
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        ### Simulate network activity for `time` timesteps ###
        timesteps = int(time / self.dt)
        for t in range(timesteps):
            current_inputs = {}
            current_inputs.update(self._get_inputs())
            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(
                        x=torch.zeros(
                            self.layers[l].s.shape, device=self.layers[l].s.device
                        )
                    )
            # self.layers['LIF'].s.copy_(torch.rand(self.layers['LIF'].s.shape, device=self.layers['LIF'].s.device) > 0.5)
            self._render_spikes(self.opengl_vaos['layers']['EXC_LIF']['s'], self.layers['EXC_LIF'].s.shape[1], t, time)

        # for i in range(500):
        #     self.layers['LIF'].s.copy_(torch.rand(self.layers['LIF'].s.shape, device=self.layers['LIF'].s.device) > 0.99)
        #     self._render_spikes(self.opengl_vaos['layers']['LIF']['s'], self.layers['LIF'].s.shape[1], i, time)

        # TODO: I don't care for this
        # Re-normalize connections.
        # for c in self.connections:
        #     self.connections[c].normalize()

    def _render_spikes(self, vao: np.uint32, layer_size: int, time_step: int) -> None:
        # language=rst
        """
        Render a raster plot

        :return: None
        """

        ### Wrapping over plotted index ###
        wrapped_time_step = time_step % self.max_time_steps
        # Clear screen if we've plotted to the edge of the window
        if wrapped_time_step == 0:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # Normalize to range [-1.0, +1.0]
        time_step_adjusted = (wrapped_time_step) / self.max_time_steps * 2.0 - 1.0

        ### Bind raster plot shader & variables ###
        gl.glUseProgram(self.raster_plot_program)
        gl.glUniform1f(
            gl.glGetUniformLocation(self.raster_plot_program, "time_x"),
            time_step_adjusted
        )
        gl.glUniform1f(
            gl.glGetUniformLocation(self.raster_plot_program, "neuron_count"),
            float(layer_size)
        )
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, layer_size)

        glfw.swap_buffers(self.window)
        glfw.poll_events()


if __name__ == '__main__':
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import MulticompartmentConnection
    from bindsnet.network.topology_features import Weight, Mask

    IN_SIZE = 100
    EXC_SIZE = 10_000
    INH_SIZE = 1_500
    SIM_TIME = 500
    BATCH_SIZE = 1

    device = torch.device('cuda:0')
    network = GUINetwork()
    network.add_layer(layer=Input(IN_SIZE), name='I')
    network.add_layer(layer=LIFNodes(EXC_SIZE), name='EXC_LIF')
    network.add_layer(layer=LIFNodes(INH_SIZE), name='INH_LIF')
    network.add_connection(
        connection=MulticompartmentConnection(
            source=network.layers['I'],
            target=network.layers['EXC_LIF'],
            device=device,
            pipeline=[
                Weight(
                    name='I_to_EXC_LIF_weight',
                    value=torch.rand(IN_SIZE, EXC_SIZE, device=device),
                ),
                Mask(
                    name='EXC_LIF_mask',
                    value=torch.rand(IN_SIZE, EXC_SIZE, device=device) > 0.95,
                )
            ]),
        source='I',
        target='EXC_LIF')
    network.add_connection(
        connection=MulticompartmentConnection(
            source=network.layers['I'],
            target=network.layers['INH_LIF'],
            device=device,
            pipeline=[
                Weight(
                    name='I_to_INH_LIF_weight',
                    value=torch.rand(IN_SIZE, INH_SIZE, device=device),
                ),
                Mask(
                    name='EXC_LIF_mask',
                    value=torch.rand(IN_SIZE, INH_SIZE, device=device) > 0.95,
                )
            ]),
        source='I',
        target='INH_LIF')
    network.add_connection(
        connection=MulticompartmentConnection(
            source=network.layers['INH_LIF'],
            target=network.layers['EXC_LIF'],
            device=device,
            pipeline=[
                Weight(
                    name='I_to_LIF_weight',
                    value=-torch.rand(INH_SIZE, EXC_SIZE, device=device),
                ),
                Mask(
                    name='LIF_to_mask',
                    value=torch.rand(INH_SIZE, EXC_SIZE, device=device) > 0.95,
                )
            ]),
        source='INH_LIF',
        target='EXC_LIF')
    network.add_connection(
        connection=MulticompartmentConnection(
            source=network.layers['EXC_LIF'],
            target=network.layers['INH_LIF'],
            device=device,
            pipeline=[
                Weight(
                    name='I_to_LIF_weight',
                    value=torch.rand(EXC_SIZE, INH_SIZE, device=device),
                ),
                Mask(
                    name='LIF_to_mask',
                    value=torch.rand(EXC_SIZE, INH_SIZE, device=device) > 0.95,
                )
            ]),
        source='EXC_LIF',
        target='INH_LIF')

    network.to(device)
    inputs = torch.rand(SIM_TIME, BATCH_SIZE, IN_SIZE, device=device) > 0.90
    network.run(inputs={'I': inputs}, time=SIM_TIME)
