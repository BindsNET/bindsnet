import torch
from Reservoir import Reservoir
from Memory import sparsify, assign_inhibition
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

def store_reservoir(exc_size, inh_size, num_samples, num_grid_cells, gc_multiples, sim_time,
                    hyper_params, plot=False):
  print("Storing memories...")

  ## Create synaptic weights ##
  in_size = num_grid_cells * gc_multiples
  w_in_exc = torch.rand(in_size, exc_size)    # Initialize weights
  w_in_inh = torch.rand(in_size, inh_size)
  w_exc_exc = torch.rand(exc_size, exc_size)
  w_exc_inh = torch.rand(exc_size, inh_size)
  w_inh_exc = -torch.rand(inh_size, exc_size)
  w_inh_inh = torch.rand(inh_size, inh_size)
  w_in_exc = sparsify(w_in_exc, 0.85)   # 0 x% of weights
  w_in_inh = sparsify(w_in_inh, 0.85)
  w_exc_exc = sparsify(w_exc_exc, 0.85)
  w_exc_inh = sparsify(w_exc_inh, 0.85)
  w_inh_exc = sparsify(w_inh_exc, 0.85)
  w_inh_inh = sparsify(w_inh_inh, 0.85)
  res = Reservoir(in_size, exc_size, inh_size, hyper_params,
                  w_in_exc, w_in_inh, w_exc_exc, w_exc_inh, w_inh_exc, w_inh_inh)

  ## Load grid cell spike-train samples ##
  with open('Data/grid_cell_spk_trains.pkl', 'rb') as f:
    grid_cell_data, labels = pkl.load(f)  # (samples, time, num_cells)

  ## Store memories ##
  # -> STDP active
  # if plot:
  #   fig, ax = plt.subplots(2, 2, figsize=(10, 5))
  #   im = ax[0, 0].imshow(w_in_res)
  #   ax[0, 0].set_title("Initial Input-to-Res")
  #   plt.colorbar(im, ax=ax[0, 0])
  #   ax[0, 0].set_xlabel("Res Neuron")
  #   ax[0, 0].set_ylabel("Input Neuron")
  #   im = ax[0, 1].imshow(w_res_res)
  #   ax[0, 1].set_title("Initial Res-to-Res")
  #   plt.colorbar(im, ax=ax[0, 1])
  #   ax[0, 1].set_xlabel("Res Neuron")
  #   ax[0, 1].set_ylabel("Res Neuron")

  # Store samples
  sample_inds = np.random.choice(len(grid_cell_data), num_samples, replace=False)
  samples = grid_cell_data[sample_inds]  # (#-samples, time, num-cells)
  labels = labels[sample_inds]
  np.random.shuffle(samples)
  for i, s in enumerate(samples):
    res.store(torch.tensor(s.reshape(sim_time, -1)), sim_time=sim_time)
    res.reset_state_variables()

  # if plot:
  #   im = ax[1, 0].imshow(w_in_res)
  #   ax[1, 0].set_title("Final Input-to-Res")
  #   plt.colorbar(im, ax=ax[1, 0])
  #   ax[1, 0].set_xlabel("Res Neuron")
  #   ax[1, 0].set_ylabel("Input Neuron")
  #   im = ax[1, 1].imshow(w_res_res)
  #   ax[1, 1].set_title("Final Res-to-Res")
  #   plt.colorbar(im, ax=ax[1, 1])
  #   ax[1, 1].set_xlabel("Res Neuron")
  #   ax[1, 1].set_ylabel("Res Neuron")
  #   plt.tight_layout()
  #   plt.show()

  ## Save ##
  with open('Data/reservoir_module.pkl', 'wb') as f:
    pkl.dump(res, f)
