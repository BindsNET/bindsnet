import os
import time
import torch
from PIL import Image
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from bindsnet.datasets import Davis
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=600)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

size = (args.width, args.height)

# Create Sequence Dataset
seq_dataset = Davis(
    root="../../data/Davis",
    task="semi-supervised",
    subset="train",
    download=True,
    size=size,
)

# Create a dataloader to iterate sequences
seq_dataloader = torch.utils.data.DataLoader(
    seq_dataset, batch_size=1, shuffle=True, num_workers=0  # pin_memory=gpu
)

seqs = []

for step, batch in enumerate(seq_dataloader):

    seqs.append(((batch["images"], batch["masks"])))

fig = plt.figure()

for images, masks in seqs:
    for i in range(len(images)):
        img = mpimg.imread(images[i][0], "JPG")
        plt.subplot(211)
        imgplot = plt.imshow(img)
        if type(masks[i][0]) != torch.Tensor:
            mask = mpimg.imread(masks[i][0], "JPG")
            plt.subplot(212)
            plt.imshow(mask)
        plt.show()
        plt.pause(1e-8)
