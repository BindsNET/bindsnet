import torch

# global var set by Network.init() for the various
# modules to use. 'cpu' or 'cuda'
network_device = "cpu"

# for later use: floating point precision for PyTorch
# torch.float16 or torch.float32
network_float_type = torch.float16
