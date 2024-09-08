import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np


def recalled_mem_preprocessing(window_freq, window_size, plot):
  print('Preprocessing recalled memories...')

  ## Load recalled memory spike-trains ##
  with open('Data/recalled_memories.pkl', 'rb') as f:
    samples, labels = pkl.load(f)  # Used as training data, hence samples & labels

  ## Load recalled memory spike-trains ##
  with open('Data/recalled_memories_sorted.pkl', 'rb') as f:
    recalled_memories_sorted = pkl.load(f)  # Used as training data, hence samples & labels

  ## Transformer (reduces sample dimensions) ##
  # def windowed_spike_train(spike_train):
  #   windowed_spikes = np.zeros((len(spike_train) // window_freq, spike_train.shape[1]))
  #   for i in range(0, len(windowed_spikes)):  # Iterate through windows
  #     if i * window_size + window_size > len(spike_train):  # Last window...
  #       window = spike_train[i * window_freq:]  # ...use remaining spikes
  #       windowed_spikes[i] = window.sum(0)
  #     else:
  #       window = spike_train[i * window_size:i * window_size + window_size]
  #     windowed_spikes[i] = window.sum(0)  # Sum spikes in window
  #   return windowed_spikes

  # new_samples = np.zeros((len(samples), len(samples[0]) // window_freq, samples[0].shape[1]))
  new_samples = np.zeros((len(samples), samples[0].shape[1]))
  new_samples_sorted = {}
  for i, s in enumerate(samples):  # Apply transformer to each sample
    # s = windowed_spike_train(s)
    s = s.sum(0)
    new_samples[i] = s
    label = tuple(labels[i].round())
    if label not in new_samples_sorted:
      new_samples_sorted[label] = [s]
    else:
      new_samples_sorted[label].append(s)

  ## Save transformed samples ##
  with open('Data/preprocessed_recalls.pkl', 'wb') as f:
    pkl.dump((new_samples, labels), f)

  if plot:
    # positions = np.array([key for key in new_samples_sorted.keys()])
    # fig = plt.figure(figsize=(10, 10))
    # gs = fig.add_gridspec(nrows=5, ncols=5)
    # for i, pos in enumerate(positions):
    #   ax = fig.add_subplot(gs[int(pos[0]), int(pos[1])])
    #   avg_mem = np.mean(new_samples_sorted[tuple(pos)], axis=0)
    #   ax.set_title(f"Conf-Mat: {pos[0] * 5 + pos[1]}")
    #   im = ax.imshow(np.expand_dims(avg_mem, axis=0))
    #   ax.set_aspect('auto')
    # plt.tight_layout()
    # plt.show()

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(nrows=5, ncols=5)
    for i, pos in enumerate(positions):
      ax = fig.add_subplot(gs[int(pos[0]), int(pos[1])])
      avg_mem = np.mean(recalled_memories_sorted[tuple(pos)], axis=0)
      ax.set_title(f"Conf-Mat: {pos[0] * 5 + pos[1]}")
      im = ax.imshow(np.expand_dims(avg_mem, axis=0).squeeze())
      ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()