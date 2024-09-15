import pickle as pkl
import numpy as np
import torch
from matplotlib import pyplot as plt

def recall_reservoir(exc_size, inh_size, sim_time, plot=False):
  print("Recalling memories...")

  ## Load memory module and memory keys ##
  with open('Data/reservoir_module.pkl', 'rb') as f:
    res_module = pkl.load(f)
  with open('Data/grid_cell_spk_trains.pkl', 'rb') as f:
    memory_keys, labels = pkl.load(f)

  ## Recall memories ##
  recalled_memories = np.zeros((len(memory_keys), sim_time, exc_size + inh_size))
  recalled_memories_sorted = {}
  for i, (key, label) in enumerate(zip(memory_keys, labels)):
    exc_spikes, inh_spikes = res_module.recall(torch.tensor(key.reshape(sim_time, -1)), sim_time=sim_time)  # Recall the sample
    all_spikes = torch.cat((exc_spikes, inh_spikes), dim=2).squeeze()
    recalled_memories[i] = all_spikes  # Store the recalled memory
    label = tuple(label.round())
    if label not in recalled_memories_sorted:
      recalled_memories_sorted[label] = [all_spikes]
    else:
      recalled_memories_sorted[label].append(all_spikes)

  ## Save recalled memories ##
  with open('Data/recalled_memories.pkl', 'wb') as f:
    pkl.dump((recalled_memories, labels), f)
  with open('Data/recalled_memories_sorted.pkl', 'wb') as f:
    pkl.dump(recalled_memories_sorted, f)

  # Plot recalls
  # if plot:
  #   positions = np.array([key for key in recalled_memories_sorted.keys()])
  #   rand_inds = np.random.choice(range(len(positions)), 5)
  #   for pos in positions[rand_inds]:
  #     fig = plt.figure(figsize=(10, 3))
  #     gs = fig.add_gridspec(1, 6)
  #     ax1 = fig.add_subplot(gs[0, 0])
  #     ax1.set_title(f"Position: {pos}")
  #     avg_mem = np.mean(recalled_memories_sorted[tuple(pos)], axis=0)
  #     ax1.imshow(avg_mem.T)
  #     random_inds = np.random.choice(range(len(recalled_memories_sorted[tuple(pos)])), 5)
  #     random_samples = np.array(recalled_memories_sorted[tuple(pos)])[random_inds]
  #     vmin = np.min(random_samples)
  #     vmax = np.max(random_samples)
  #     for i in range(1, 5):
  #       ax = fig.add_subplot(gs[0, i])
  #       rand_sample = recalled_memories_sorted[tuple(pos)][random_inds[i]]
  #       im = ax.imshow(np.expand_dims(rand_sample.T, axis=1).squeeze(), vmin=vmin, vmax=vmax)
  #       ax.set_title(f"S{i}")
  #       ax.set(xticklabels=[])
  #       ax.set(yticklabels=[])
  #     plt.show()
