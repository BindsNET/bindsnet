import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from timeit import default_timer

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'datasets')))

# from datasets          import MNIST
from datasets		   import SpokenMNIST
from network           import Network
from encoding          import get_poisson
from encoding          import get_poisson_mixture
from encoding          import get_bernoulli_mixture
from connections       import Connection, post_pre
from nodes             import AdaptiveLIFNodes, LIFNodes, Input
from analysis.plotting import plot_input, plot_spikes, plot_weights
from evaluation 	   import *

model_name = 'eth_spokenMNIST'

# logs_path = os.path.join('..', 'logs', model_name)
# data_path = os.path.join('..', 'data', model_name)
params_path = os.path.join('..', 'params', model_name)
results_path = os.path.join('..', 'results', model_name)
assign_path = os.path.join('..', 'assignments', model_name)
proportions_path = os.path.join('..', 'proportions', model_name)
# perform_path = os.path.join('..', 'performances', model_name)

for path in [params_path, assign_path, results_path, proportions_path]:
	if not os.path.isdir(path):
		os.makedirs(path)

def get_square_weights(weights, n_sqrt):
	square_weights = torch.zeros_like(torch.Tensor(28 * n_sqrt, 28 * n_sqrt))
	for i in range(n_sqrt):
		for j in range(n_sqrt):
			filter_ = weights[:, i * n_sqrt + j].contiguous().view(28, 28)
			square_weights[i * 28 : (i + 1) * 28, (j % n_sqrt) * 28 : ((j % n_sqrt) + 1) * 28] = filter_
	
	return square_weights

def save_params(params_path, params, fname, prefix):
	'''
	Save network params to disk.

	Arguments:
		- params (numpy.ndarray): Array of params to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(params_path, '_'.join([prefix, fname]) + '.npy'), params)


def load_params(params_path, fname, prefix):
	'''
	Load network params from disk.

	Arguments:
		- fname (str): File name of file to read from.
		- prefix (str): Name of the parameters to read from disk.

	Returns:
		- params (numpy.ndarray): Params stored in file `fname`.
	'''
	return np.load(os.path.join(params_path, '_'.join([prefix, fname]) + '.npy'))


def save_assignments(assign_path, assignments, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- assignments (numpy.ndarray): Array of assignments to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(assign_path, '_'.join(['assignments', fname]) + '.npy'), assignments)


def load_assignments(assign_path, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- fname (str): File name of file to read from.

	Returns:
		- assignments (numpy.ndarray): Assignments stored in file `fname`.
	'''
	return np.load(os.path.join(assign_path, '_'.join(['assignments', fname]) + '.npy'))


def save_proportions(proportions_path, proportions, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- assignments (numpy.ndarray): Array of assignments to save.
		- fname (str): File name of file to write to.
	'''
	np.save(os.path.join(proportions_path, '_'.join(['proportions', fname]) + '.npy'), proportions)


def load_proportions(proportions_path, fname):
	'''
	Save network assignments to disk.

	Arguments:
		- fname (str): File name of file to read from.

	Returns:
		- assignments (numpy.ndarray): Assignments stored in file `fname`.
	'''
	return np.load(os.path.join(proportions_path, '_'.join(['proportions', fname]) + '.npy'))


print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=1000)
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--excite', type=float, default=22.5)
parser.add_argument('--inhib', type=float, default=17.5)
parser.add_argument('--wmax', type=float, default=1.0)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--dt', type=int, default=1.0)
parser.add_argument('--min_isi', type=float, default=25.0)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('--no-gpu', dest='gpu', action='store_false')
parser.set_defaults(plot=False, gpu=False, train=True)

args = vars(parser.parse_args())
locals().update(args)

# Build filename from command-line arguments.
fname = '_'.join([str(n_neurons), str(n_train), str(seed), str(inhib), str(excite), str(wmax)])

# Log argument values.
print('\nOptional argument values:')
for key, value in args.items():
	print('-', key, ':', value)

print('\n')

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

if not train:
	update_interval = n_test

# Build network.
network = Network(dt=dt)

n_sqrt = int(np.sqrt(n_neurons))

# Layers of neurons.
# Input layer.
input_layer = Input(n=40, traces=True, trace_tc=1 / 20)

# Excitatory layer.
exc_layer = AdaptiveLIFNodes(n=n_neurons, traces=True, rest=-65.0, reset=-65.0, threshold=-52.0, refractory=5,
                                    voltage_decay=1e-2, trace_tc=1 / 20, theta_plus=0.05, theta_decay=1e-7)

# Inhibitory layer.
inh_layer = LIFNodes(n=n_neurons, traces=False, rest=-60.0, reset=-45.0, threshold=-40.0,
                                 voltage_decay=1e-1, refractory=2, trace_tc=1 / 20)

# Connections between layers.
# Input -> excitatory.
if train:
	input_exc_w = 0.3 * torch.rand(input_layer.n, exc_layer.n)
else:
	if gpu:
		input_exc_w = torch.from_numpy(load_params(params_path, fname, 'X_Ae')).cuda()
	else:
		input_exc_w = torch.from_numpy(load_params(params_path, fname, 'X_Ae'))

input_exc_conn = Connection(source=input_layer, target=exc_layer, w=input_exc_w, update_rule=post_pre, wmin=0.0, wmax=wmax)

# Excitatory -> inhibitory.
exc_inh_w = excite * torch.diag(torch.ones(exc_layer.n))
exc_inh_conn = Connection(source=exc_layer, target=inh_layer, w=exc_inh_w, update_rule=None)

# Inhibitory -> excitatory.
inh_exc_w = -1 * inhib * (torch.ones(inh_layer.n, exc_layer.n) - torch.diag(torch.ones(inh_layer.n)))
inh_exc_conn = Connection(source=inh_layer, target=exc_layer, w=inh_exc_w, update_rule=None)

# Add all layers and connections to the network.
network.add_layer(input_layer, name='X')
network.add_layer(exc_layer, name='Ae')
network.add_layer(inh_layer, name='Ai')
network.add_connection(input_exc_conn, source='X', target='Ae')
network.add_connection(exc_inh_conn, source='Ae', target='Ai')
network.add_connection(inh_exc_conn, source='Ai', target='Ae')

# Load MNIST data.
if train:
	audios, labels = SpokenMNIST().get_train()
else:
	audios, labels = SpokenMNIST().get_test()

# Lazily encode data as Poisson spike trains.
# data_loader = get_poisson_mixture(data=audios, time=time, window=50)
data_loader = get_bernoulli_mixture(data=audios, time=time, window=30)

# Record spikes during the simulation.
spike_record = torch.zeros_like(torch.Tensor(update_interval, time, n_neurons))
spike_record_full = torch.zeros_like(torch.Tensor(n_train, time, n_neurons))

# Neuron assignments and spike proportions.
rates = torch.zeros_like(torch.Tensor(n_neurons, 10))

if train:
	assignments = -1 * torch.ones_like(torch.Tensor(n_neurons))
	proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
	
else:
	if gpu:
		assignments = torch.from_numpy(load_assignments(assign_path, fname)).cuda()
		proportions = torch.from_numpy(load_proportions(proportions_path, fname)).cuda()
	else:
		assignments = torch.from_numpy(load_assignments(assign_path, fname))
		proportions = torch.from_numpy(load_proportions(proportions_path, fname))

voting_schemes = ['all', 'proportion']
# Sequence of accuracy estimates.
accuracy = { scheme : [] for scheme in voting_schemes }

# Keep track of correct classifications for performance monitoring.
correct = { scheme : 0 for scheme in voting_schemes }
total_correct = { scheme : 0 for scheme in voting_schemes }

if train:
	n_samples = n_train
else:
	n_samples = n_test


n_images = len(audios) # TODO i%n_images & print epochs = i/n_images
best_accuracy = -np.inf

# Train the network.
print('Running the network with train = ', train, ".\n")
start = default_timer()
train_spikes = []
intensity = 1
for i in range(n_samples):    
	if i % progress_interval == 0:
		print('Progress: %d / %d (%.4f seconds)' % (i, n_samples, default_timer() - start))
		start = default_timer()
	
	if train:
		if i % update_interval == 0 and i > 0:
			# Assign labels to excitatory layer neurons based on the last update interval
			assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)

			# Get network predictions.
			all_activity_pred = all_activity(spike_record, assignments, 10)
			proportion_pred = proportion_weighting(spike_record, assignments, proportions, 10)
			
			# Compute network accuracy according to available classification strategies.
			accuracy['all'].append(100 * torch.sum(labels[i - update_interval:i].long() \
													== all_activity_pred) / update_interval)
			accuracy['proportion'].append(100 * torch.sum(labels[i - update_interval:i].long() \
															== proportion_pred) / update_interval)
			
			print('\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)' \
							% (accuracy['all'][-1], np.mean(accuracy['all']), np.max(accuracy['all'])))
			print('Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n' \
							% (accuracy['proportion'][-1], np.mean(accuracy['proportion']),
							  np.max(accuracy['proportion'])))
			
			new_best = False
			if performances['all'][-1] > best_accuracy:
				best_accuracy = performances['all'][-1]
				new_best = True
			if performances['proportion'][-1] > best_accuracy:
				best_accuracy = performances['proportion'][-1]
				new_best = True
			if new_best:
				if gpu:
					weights = network.get_weights(('X', 'Ae')).cpu().numpy()
					theta = network.get_theta('Ae').cpu().numpy()
					asgnmts = assignments.cpu().numpy()
					prprtns = proportions.cpu().numpy()
				else:
					weights = network.get_weights(('X', 'Ae')).numpy()
					theta = network.get_theta('Ae').numpy()
					asgnmts = assignments.numpy()
					prprtns = proportions.numpy()
				
				save_params(params_path, weights, fname, 'X_Ae')
				save_params(params_path, theta, fname, 'theta')
				save_assignments(assign_path, asgnmts, fname)
				save_proportions(proportions_path, prprtns, fname)
			
			# Save sequence of performance estimates to file.
			# p.dump(performances, open(os.path.join(perform_path, fname), 'wb'))
	
	# Get next input datum.
	sample = next(data_loader)
	inpts = {'X' : sample}
	if gpu:
		inpts['X'] = inpts['X'].cuda()
	
	# Run the network on the input for time `t`.
	spikes = network.run(inpts=inpts, time=time)
	train_spikes.append(spikes['Ae']) # TODO this is duplicated - also stored in spike_record and spike_record_full
	
	print("Total spikes in exc layer for example", i, "=", torch.sum(spikes['Ae']))
	# TODO how to rerun using yield?
	# # Re-run image if there isn't any network activity.
	# n_retries = 0
	# while torch.sum(spikes['Ae']) < 5 and n_retries < 3:
	# 	intensity += 1; n_retries += 1
	# 	inpts['X'] = torch.from_numpy(generate_spike_train(image, intensity * dt, int(image_time / dt)))
	# 	spikes = network.run(mode=mode, inpts=inpts, time=image_time)
	# intensity = 1



	network._reset()  # Reset state variables.
	network.connections[('X', 'Ae')].normalize()  # Normalize input -> excitatory weights
	
	# Record spikes.
	spike_record[i % update_interval] = spikes['Ae']
	spike_record_full[i] = spikes['Ae']

	# Optionally plot the excitatory, inhibitory spiking.
	if plot:
		# TODO handle gpu case
		inpt = inpts['X'].t()
		exc_spikes = spikes['Ae']; inh_spikes = spikes['Ai']
		input_exc_weights = network.connections[('X', 'Ae')].w
		square_weights = get_square_weights(input_exc_weights, n_sqrt)
		
		if i == 0:
			inpt_ims = plot_input(images[i].view(28, 28), inpt)
			spike_ims, spike_axes = plot_spikes({'Ae' : exc_spikes, 'Ai' : inh_spikes})
			weights_im = plot_weights(square_weights)
			assigns_im = plot_assignments(assignments)
			perf_ax = plot_performance(accuracy)
		else:
			inpt_ims = plot_input(images[i].view(28, 28), inpt, ims=inpt_ims)
			spike_ims, spike_axes = plot_spikes({'Ae' : exc_spikes, 'Ai' : inh_spikes}, ims=spike_ims, axes=spike_axes)
			weights_im = plot_weights(square_weights, im=weights_im)
			assigns_im = plot_assignments(assigassignmentsnments, im=assigns_im)
			perf_ax = plot_performance(accuracy, ax=perf_ax)
		
		plt.pause(1e-8)

print('Progress: %d / %d (%.4f seconds)\n' % (n_samples, n_samples, default_timer() - start))
print('Run complete.\n')

if train:
	assignments, proportions, _ = assign_labels(spike_record_full, labels[:n_train], 10)
	predictions_pw = proportion_weighting(spike_record_full, assignments, proportions, 10)
	predictions_all = all_activity(spike_record_full, assignments, 10)
	print("Accuracy Proportion Weighting = ", np.mean(np.array(predictions_pw)==np.array(labels[:n_train],dtype=np.int32)))
	print("Accuracy All Activity = ", np.mean(np.array(predictions_all)==np.array(labels[:n_train],dtype=np.int32)))

	print("Calculating ngram scores..")
	ngrams = estimate_ngram_probabilities(train_spikes, labels[:len(train_spikes)], 2)
	print("Accuracy = ", ngram(train_spikes, labels[:len(train_spikes)], ngrams, 2))

	# TODO Write all info to files to be later used for testing


else:
	print("Testing using the saved assignments, proportions and ngrams")
	# TODO use the saved assignments, proportions and ngrams for testing