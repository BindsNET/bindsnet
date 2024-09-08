import pickle as pkl
import torch
import numpy as np
from matplotlib import pyplot as plt

from Memory import Memory_SNN, sparsify

if __name__ == '__main__':
  ## Constants ##
  KEY_SIZE = 150
  VAL_SIZE = 150
  NUM_GRID_CELLS = 20
  IN_KEY_SHAPE = (NUM_GRID_CELLS, KEY_SIZE)
  IN_VAL_SHAPE = (NUM_GRID_CELLS, VAL_SIZE)
  ASSOC_SHAPE = (KEY_SIZE, VAL_SIZE)
  SIM_TIME = 50
  WINDOW_FREQ = 10
  WINDOW_SIZE = 10
  NUM_SAMPLES = 2_500    # Number of samples to store
  PLOT = True

  ## Initialize Memory SNN ##
  w_in_key = torch.rand(IN_KEY_SHAPE)
  w_in_val = torch.rand(IN_VAL_SHAPE)
  w_assoc = torch.rand(ASSOC_SHAPE)
  # w_in_key = assign_inhibition(w_in_key, 0.2, 1)  # (weights, %-inhib, scale)
  # w_in_val = assign_inhibition(w_in_val, 0.2, 1)
  w_in_key = sparsify(w_in_key, 0.5)  # (weights, %-zero)
  w_in_val = sparsify(w_in_val, 0.5)
  #  w_assoc = sparsify(w_assoc, 0.25)
  hyper_params = {
    'thresh': -40,
    'theta_plus': 5,
    'refrac': 5,
    'reset': -65,
    'tc_theta_decay': 500,
    'tc_decay': 30,  # time constant for neuron decay; smaller = faster decay
    'nu': [0.005, 0.005],
    'decay': 0.00001
  }
  memory_module = Memory_SNN(
    KEY_SIZE, VAL_SIZE, NUM_GRID_CELLS,
    w_in_key, w_in_val, w_assoc,
    hyper_params
  )

  ## Load grid cell spike-train samples ##
  with open('Data/grid_cell_spk_trains.pkl', 'rb') as f:
    grid_cell_data, labels = pkl.load(f)    # (samples, time, num_cells)

  ## Store memories ##
  # -> STDP active
  if PLOT:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(w_assoc)
    ax[0].set_title("Initial Association Weights")
    plt.colorbar(im, ax=ax[0])
    ax[0].set_xlabel("Value Neuron")
    ax[0].set_ylabel("Key Neuron")

  # Store samples
  sample_inds = np.random.choice(len(grid_cell_data), NUM_SAMPLES, replace=False)
  samples = grid_cell_data[sample_inds] # (#-samples, time, num-cells)
  labels = labels[sample_inds]
  for i, s in enumerate(samples):
    if i % 10 == 0:
      print(f"Storing sample {i} of {NUM_SAMPLES}")
    memory_module.store(torch.tensor(s), sim_time=SIM_TIME)
    memory_module.reset_state_variables()

  if PLOT:
    im = ax[1].imshow(w_assoc)
    ax[1].set_title("Final Association Weights")
    plt.colorbar(im, ax=ax[1])
    ax[1].set_xlabel("Value Neuron")
    ax[1].set_ylabel("Key Neuron")
    plt.tight_layout()
    plt.show()

  ## Save ##
  with open('Data/memory_module.pkl', 'wb') as f:
    pkl.dump(memory_module, f)
