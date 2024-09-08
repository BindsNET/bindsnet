import pickle as pkl
import numpy as np
import torch

from Memory import Memory_SNN

if __name__ == '__main__':
  ## Constants ##
  KEY_SIZE = 150
  VAL_SIZE = 150
  NUM_GRID_CELLS = 20
  SIM_TIME = 50

  ## Load memory module and memory keys ##
  with open('Data/memory_module.pkl', 'rb') as f:
    memory_module = pkl.load(f)
  with open('Data/grid_cell_spk_trains.pkl', 'rb') as f:
    memory_keys, labels = pkl.load(f)

  ## Recall memories ##
  recalled_memories = np.zeros((len(memory_keys), SIM_TIME, VAL_SIZE))
  recalled_memories_sorted = {}
  for i, (key, label) in enumerate(zip(memory_keys, labels)):
    if i % 100 == 0:
      print(f'Recalling memory {i}...')
    value_spike_train = memory_module.recall(torch.tensor(key), sim_time=SIM_TIME)    # Recall the sample
    recalled_memories[i] = value_spike_train.squeeze()   # Store the recalled memory
    label = tuple(label.round())
    if label not in recalled_memories_sorted:
      recalled_memories_sorted[label] = [value_spike_train.squeeze()]
    else:
      recalled_memories_sorted[label].append(value_spike_train.squeeze())

  ## Save recalled memories ##
  with open('Data/recalled_memories.pkl', 'wb') as f:
    pkl.dump((recalled_memories, labels), f)
  with open('Data/recalled_memories_sorted.pkl', 'wb') as f:
    pkl.dump(recalled_memories_sorted, f)
