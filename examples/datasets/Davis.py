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

# Create Sequence Dataset
seq_dataset = Davis(
    root="../../data/Davis", task="semi-supervised", subset="test-dev", download=True
)

# Create a dataloader to iterate sequences
seq_dataloader = torch.utils.data.DataLoader(
    seq_dataset, batch_size=1, shuffle=True, num_workers=0  # pin_memory=gpu
)

seqs = []

for step, batch in enumerate(seq_dataloader):

    seqs.append(tuple((batch["images"], batch["masks"])))
    print(len(seqs))

fig = plt.figure()

for images, masks in seqs:
    for i in range(len(images)):
        img = mpimg.imread(images[i][0], "JPG")
        plt.subplot(211)
        imgplot = plt.imshow(img)
        if not type(masks[i][0]) == torch.Tensor:
            mask = mpimg.imread(masks[i][0], "JPG")
            plt.subplot(212)
            plt.imshow(mask)
        plt.show()
        time.sleep(0.05)
        plt.pause(1e-8)
