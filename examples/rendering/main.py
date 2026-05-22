from bindsnet.rendering.app import Application
from bindsnet.rendering.widgets import RasterPlotWidget
from model import create_model
import torch

SIM_TIME = 500
BATCH_SIZE = 1
DEVICE = "cuda:0"

IN_SIZE = 100
EXC_SIZE = 10_000
INH_SIZE = 1_500
I_TO_EXC_CONNECTIVITY = 0.05
I_TO_INH_CONNECTIVITY = 0.05
INH_TO_EXC_CONNECTIVITY = 0.05
EXC_TO_INH_CONNECTIVITY = 0.05

network = create_model(
  IN_SIZE,
  EXC_SIZE,
  INH_SIZE,
  I_TO_EXC_CONNECTIVITY,
  I_TO_INH_CONNECTIVITY,
  INH_TO_EXC_CONNECTIVITY,
  EXC_TO_INH_CONNECTIVITY,
)
app = Application(network, 1400, 900)
inputs = {"I" : torch.rand(SIM_TIME, BATCH_SIZE, IN_SIZE, device=DEVICE) > 0.90}
app.add_widget(
  RasterPlotWidget(
    width=400,
    height=300,
    x=50,
    y=50,
    vao=network.opengl_vaos['layers']['EXC_LIF']['s'],      # TODO: Clean this up
    layer_size=EXC_SIZE
  )
)
app.run(inputs=inputs, time=SIM_TIME)
