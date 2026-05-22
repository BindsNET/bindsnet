import torch
import glfw
import OpenGL.GL as gl
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np

class AbstractWidget:
  def __init__(self, width: float, height: float, x:float, y:float):
    self.width = width
    self.height = height
    self.x = x
    self.y = y
    self.border_inset = 0.05    # % padding from edges of widget to border
    self.drawable_width = 2.0 - 2*self.border_inset   # OpenGL coordinates are normalized to [-1.0, 1.0], `
    self.drawable_height = 2.0 - 2*self.border_inset  # so drawable area is 2.0 minus border insets on both sides

    self.border_vao = self.create_border_geometry()
    self.line_shader = self.create_line_shader()

  def set_window(self, app_window: glfw._GLFWwindow):
    self.window = app_window

  def create_line_shader(self):
    vertex_shader = """
    #version 330 core
    layout(location = 0) in vec2 pos;
    void main()
    {
        gl_Position = vec4(pos, 0.0, 1.0);
    }
    """

    fragment_shader = """
    #version 330 core
    out vec4 FragColor;
    void main()
    {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    """

    return compileProgram(
      compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
      compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
    )

  def create_border_geometry(self):
    ### Coordinates slightly inset from border ###
    vertices = np.array([
      -1+self.border_inset, -1+self.border_inset,
      1-self.border_inset, -1+self.border_inset,
      1-self.border_inset, 1-self.border_inset,
      -1+self.border_inset, 1-self.border_inset,
    ], dtype=np.float32)

    ### Generate VAO for border geometry ###
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(
      gl.GL_ARRAY_BUFFER, # Target buffer
      vertices.nbytes,    # Size of data in bytes
      vertices,           # Data
      gl.GL_STATIC_DRAW   # Type of drawing (static data, not changing frequently)
    )
    gl.glVertexAttribPointer(
      0,            # VAO slot
      2,            # x,y
      gl.GL_FLOAT,  # Data type
      False,        # Normalized?
      0,            # Stride
      None          # Offset in buffer
    )
    gl.glEnableVertexAttribArray(0)
    gl.glBindVertexArray(0)

    return vao

  def render_border(self):
    gl.glViewport(
      self.x,
      self.y,
      self.width,
      self.height
    )
    gl.glUseProgram(self.line_shader)
    gl.glBindVertexArray(self.border_vao)
    gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 4)
    gl.glBindVertexArray(0)

  def render(self, time_step: int):
    # Set size of area we are rendering into
    gl.glViewport(
      self.x,
      self.y,
      self.width,
      self.height
    )

class RasterPlotWidget(AbstractWidget):
  # language=rst
  """
  Render a raster plot

  :param width: Width of the raster plot
  :param height: Height of the raster plot
  :param vao: Vertex Array Object index containing spike data
  :param layer_size: Number of neurons in the layer being plotted
  :return: None
  """
  def __init__(self,
      width: float,
      height: float,
      x:float,
      y:float,
      vao: int,
      layer_size: int,
    ):
    super().__init__(width, height, x, y)
    self.vao = vao
    self.layer_size = layer_size
    self.max_time_steps = width
    self.window = None  # Will be defined when widget added to Application

    ### Define shaders ###
    raster_texture_vertex_shader = """
        #version 330 core

        layout(location = 0) in vec2 pos;
        out vec2 uv;
        void main()
        {
            uv = pos * 0.5 + 0.5;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """
    raster_texture_fragment_shader = """
        #version 330 core

        in vec2 uv;
        out vec4 FragColor;
        uniform sampler2D raster_tex;
        uniform float write_head;
        uniform float history_width;

        void main()
        {   
            int x = int(uv.x * history_width);
            int y = int(uv.y * history_width);

            int shifted_x = int(
              mod((history_width + x + 1) + write_head, history_width)
            );

            ivec2 texel_coord = ivec2(shifted_x, y);
            float spike =
                texelFetch(
                    raster_tex,
                    texel_coord,
                    0
                ).r;
            vec3 color = vec3(spike);
            FragColor = vec4(color, 1.0);
        }
    """
    self.raster_plot_program = compileProgram(
      compileShader(raster_texture_vertex_shader, gl.GL_VERTEX_SHADER),
      compileShader(raster_texture_fragment_shader, gl.GL_FRAGMENT_SHADER)
    )

    ### Define texture for rolling spike buffer ###
    self.raster_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, self.raster_texture)
    gl.glTexImage2D(
      gl.GL_TEXTURE_2D,
      0,          # Mipmap level
      gl.GL_R8,   # Internal format (32-bit float)  TODO: Can this be bool?
      self.max_time_steps,  # Width of texture (time steps)
      layer_size,   # Height of texture (neurons)
      0,            # Border
      gl.GL_RED,    # Format of pixel data
      gl.GL_UNSIGNED_BYTE,  # Data type of pixel data
      np.zeros(
        (layer_size, self.max_time_steps),
        dtype=np.uint8
      )         # No initial data
    )
    gl.glTexParameteri(
      gl.GL_TEXTURE_2D,
      gl.GL_TEXTURE_MIN_FILTER,
      gl.GL_NEAREST
    )
    gl.glPixelStorei(
      gl.GL_UNPACK_ALIGNMENT,
      1
    )

    ### Vertex indices buffer (square covering widget) ###
    quad_vertices = np.array([
      -1.0, -1.0,
      1.0, -1.0,
      1.0, 1.0,

      -1.0, -1.0,
      1.0, 1.0,
      -1.0, 1.0,
    ], dtype=np.float32)
    self.quad_vao = gl.glGenVertexArrays(1)
    quad_vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(self.quad_vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad_vbo)
    gl.glBufferData(
      gl.GL_ARRAY_BUFFER,  # Target buffer
      quad_vertices.nbytes,  # Size of data in bytes
      quad_vertices,  # Data
      gl.GL_STATIC_DRAW  # Type of drawing (static data, not changing)
    )
    gl.glVertexAttribPointer(
      0,  # VAO slot
      2,  # x,y
      gl.GL_FLOAT,  # Data type
      False,  # Normalized?
      0,  # Stride
      None  # Offset in buffer
    )
    gl.glEnableVertexAttribArray(0)
    gl.glBindVertexArray(0)

  def render_ticks(self):
    ...

  def render_spikes(self, time_step: int):
    wrapped_t = time_step % self.max_time_steps
    spikes = (np.random.random(self.layer_size) > 0.95) * 255

    ### Migrate spike data to GPU ###
    gl.glBindTexture(gl.GL_TEXTURE_2D,
                     self.raster_texture)
    gl.glTexSubImage2D(
      gl.GL_TEXTURE_2D,
      0,
      wrapped_t,  # x offset
      0,  # y offset
      1,  # width
      self.layer_size,  # height
      gl.GL_RED,
      gl.GL_UNSIGNED_BYTE,
      spikes
    )

    ### Plot ###
    gl.glUseProgram(self.raster_plot_program)
    gl.glUniform1f(
      gl.glGetUniformLocation(self.raster_plot_program, "write_head"),
      wrapped_t
    )

    gl.glUniform1f(
      gl.glGetUniformLocation(self.raster_plot_program, "history_width"),
      self.max_time_steps
    )

    gl.glActiveTexture(gl.GL_TEXTURE0)

    gl.glBindTexture(
      gl.GL_TEXTURE_2D,
      self.raster_texture
    )

    gl.glBindVertexArray(self.quad_vao)

    gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    glfw.swap_buffers(self.window)
    glfw.poll_events()

  def render(self, time_step: int):
    super().render(time_step)
    # self.render_background()
    self.render_border()
    # self.render_ticks()
    self.render_spikes(time_step)
