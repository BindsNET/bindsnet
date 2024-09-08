import pickle as pkl
import numpy as np

# Take in grid cell activity vector and turn into spike train
# max_freq: Maximum frequency of spikes
def intensity_to_spike(intensity, time, max_freq, labels=None):
  # Normalize [0, 1]
  intensity = (intensity - min(intensity)) / (max(intensity) - min(intensity))

  # Convert to spike rate
  spike_rate = intensity * max_freq
  spike_train = np.zeros((time, len(intensity)))
  for i, rate in enumerate(spike_rate):
    if rate != 0:
      spike_train[:, i] = np.zeros(time)
      spike_train[:, i][np.random.rand(time) < rate] = 1
    else:
      spike_train[:, i] = np.zeros(time)

  return spike_train

def spike_train_generator(intensities, labels, sim_time, gc_multiples, max_freq):
  print("Generating Spike Trains...")

  ## Transform intensities to spike trains ##
  with open('Data/grid_cell_intensities.pkl', 'rb') as f:
    intensities, labels = pkl.load(f)
  # with open('Data/grid_cell_intensities_sorted.pkl', 'rb') as f:
  #   intensities_sorted = pkl.load(f)
  spike_trains = np.zeros(
    (len(intensities), sim_time, len(intensities[0]), gc_multiples))  # (num_samples, time, gc, num_gc)
  sorted_spike_trains = {}
  for i, intensity in enumerate(intensities):
    for j in range(gc_multiples):
      spike_trains[i, :, :, j] = intensity_to_spike(intensity, sim_time, max_freq)
    adjusted_label = (round(labels[i][0]), round(labels[i][1]))
    if adjusted_label not in sorted_spike_trains:
      sorted_spike_trains[adjusted_label] = [spike_trains[i]]
    else:
      sorted_spike_trains[adjusted_label].append(spike_trains[i])

  ## Save to file ##
  with open('Data/grid_cell_spk_trains.pkl', 'wb') as f:
    pkl.dump((spike_trains, labels), f)
  with open('Data/grid_cell_spk_trains_sorted.pkl', 'wb') as f:
    pkl.dump((sorted_spike_trains), f)

  return spike_trains, labels, sorted_spike_trains
