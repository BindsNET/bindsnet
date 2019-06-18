import argparse
import torch
import numpy as np
from tqdm import tqdm

from bindsnet.datasets import Davis

dataset = Davis(
    root="../../data/Davis", task="semi-supervised", subset="test-dev", download=True
)
