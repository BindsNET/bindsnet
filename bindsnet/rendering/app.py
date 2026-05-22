import torch

from bindsnet.network.network import GUINetwork

from bindsnet.rendering.widgets import AbstractWidget

import time as time_lib
import OpenGL.GL as gl
import glfw

class Application():
    def __init__(self, network: GUINetwork, width=1400, height=900, title="BindsNET GUI"):
      self.width, self.height = width, height
      self.network = network
      self.widgets = []

      if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

      ### Set Preferred OpenGL version (4.6) ###
      # OPENGL_CORE_PROFILE removes deprecated functions
      glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
      glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
      glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

      ### Prepare window and OpenGL ###
      self.window = glfw.create_window(
        width,
        height,
        title,
        None,   # Windowed mode
        None    # No shared context (ie. no parallel computations)
      )
      if not self.window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
      glfw.make_context_current(self.window)

      # Disable VSync, we'll handle frame timing manually
      # glfw.swap_interval(0)

      # Blending for transparent drawing
      gl.glEnable(gl.GL_BLEND)
      gl.glBlendFunc(gl.GL_SRC_ALPHA,
                     gl.GL_ONE_MINUS_SRC_ALPHA)

      # Set background to dark gray (and clear color buffer)
      gl.glClearColor(0.05, 0.05, 0.05, 1.0)
      gl.glClear(gl.GL_COLOR_BUFFER_BIT)

      ### Migrate network tensors to shared buffers ###
      self.network.migrate()

      self.last_time = time_lib.time()

    def add_widget(self, widget: AbstractWidget):
      self.widgets.append(widget)
      widget.set_window(self.window)

    def run(self, inputs: dict[str, torch.Tensor], time):
      # Effective number of timesteps.
      timesteps = int(time / self.network.dt)

      for t in range(timesteps):

        # For calculating frames
        # current_time = time.time()
        # dt = current_time - self.last_time
        # self.last_time = current_time

        # Simulate one timestep in network
        tstep_inputs = {layer_name : layer_inputs[t] for layer_name, layer_inputs in inputs.items()}
        self.network.step(tstep_inputs)

        # Update widget renders
        for widget in self.widgets:
          # widget.update(dt)
          widget.render(t)

        # Swap front/back buffer to reveal new frame
        glfw.swap_buffers(self.window)

        # Collect events (like keyboard/mouse input, window close, etc.)
        glfw.poll_events()

      glfw.terminate()