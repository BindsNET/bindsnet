import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.datasets import MNIST
from bindsnet.encoding import bernoulli_loader
from bindsnet.network import Network
from bindsnet.learning import MSTDPET
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from bindsnet.network.nodes import AdaptiveLIFNodes, Input
from bindsnet.analysis.plotting import plot_spikes, plot_weights

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=100)
parser.add_argument('--iters', type=int, default=500 * 1000)
parser.add_argument('--intensity', type=float, default=2.0)
parser.add_argument('--clamped', type=int, default=5)
parser.add_argument('--target_rate', type=float, default=0.03)
parser.add_argument('--low_rate', type=float, default=0.0)
parser.add_argument('--plot_interval', type=int, default=500)
parser.add_argument('--print_interval', type=int, default=25)
parser.add_argument('--change_interval', type=int, default=500)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)
locals().update(vars(parser.parse_args()))

args = parser.parse_args()

n = args.n
iters = args.iters
intensity = args.intensity
clamped = args.clamped
target_rate = args.target_rate
low_rate = args.low_rate
plot_interval = args.plot_interval
print_interval = args.print_interval
change_interval = args.change_interval
plot = args.plot
gpu = args.gpu

clamped = min(clamped, int(n / 10))

sqrt = int(np.ceil(np.sqrt(n)))

network = Network(dt=1.0)

inpt = Input(n=784, traces=True)
exc = AdaptiveLIFNodes(n=n, traces=True)

ew = 0.3 * torch.rand(784, n)
econn = Connection(inpt, exc, w=ew, update_rule=MSTDPET, nu=0.1, wmin=0, wmax=1, norm=78.4)

network.add_layer(inpt, 'X')
network.add_layer(exc, 'Y')
network.add_connection(econn, source='X', target='Y')

spike_monitors = {layer : Monitor(network.layers[layer], ['s']) for layer in network.layers}
for layer in spike_monitors:
    network.add_monitor(spike_monitors[layer], '%s' % layer)

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'),
                       download=True).get_train()
images *= intensity
images /= 4

# Lazily encode data as Poisson spike trains.
ims = []
for j in range(0, iters, change_interval):
    ims.extend([images[j % 60000]] * change_interval)

lbls = []
for j in range(0, iters, change_interval):
    lbls.extend([int(labels[j % 60000])] * change_interval)

loader = bernoulli_loader(data=torch.stack(ims), time=1, max_prob=0.05)

reward = 0
a_plus = 1
a_minus = 0

avg_rates = torch.zeros(n)
target_rates = low_rate * torch.ones(n)
for j in np.random.choice(range(n)[lbls[0]::10], clamped, replace=False):
    target_rates[j] = target_rate

distances = [torch.sum(torch.sqrt((target_rates - avg_rates) ** 2))]
rewards = [reward]

perfs = [[0] * 2]

spike_record = {layer : torch.zeros(network.layers[layer].n, plot_interval) for layer in network.layers}

print()
for i in range(iters):
    if i > 0 and i % change_interval == 0:
        avg_rates = torch.zeros(n)
        target_rates = low_rate * torch.ones(n)
        for j in np.random.choice(range(n)[lbls[i]::10], clamped, replace=False):
            target_rates[j] = target_rate
        
    inpts = {'X': next(loader).view(1, 784)}
    kwargs = {'reward': reward, 'a_plus': a_plus, 'a_minus': a_minus}

    network.run(inpts, 1, **kwargs)
    
    spikes = {layer : spike_monitors[layer].get('s').view(-1) for layer in spike_monitors}
    for layer in spike_record:
        spike_record[layer][:, i % plot_interval] = spikes[layer]
    
    a = i % change_interval
    if a == 0:
        avg_rates = spikes['Y']
    else:
        avg_rates = ((a - 1) / a) * avg_rates + (1 / a) * spikes['Y']
    
    reward = target_rates - avg_rates
    reward[reward < 0] = 0
    reward[reward > 0] = 0.1
    rewards.append(reward.sum())
    distance = torch.sum(torch.sqrt((target_rates - avg_rates) ** 2))
    distances.append(distance)
    
    if i > 0 and i % change_interval == 0:
        sums = []
        for j in range(10):
            if gpu:
                s = []
                for k in range(10):
                    s.append(np.sum(spike_record['Y'].cpu().numpy()[k::10].sum(axis=1)))
                
                sums = [s, np.sum(spike_record['Y'].cpu().numpy(), axis=1)]
            else:
                s = []
                for k in range(10):
                    s.append(np.sum(spike_record['Y'].numpy()[k::10].sum(axis=1)))
                    
                sums = [s, np.sum(spike_record['Y'].numpy(), axis=1)]
        
        a = int(i / change_interval)
        if a == 1:
            p = [np.argmax(sums[0]) == lbls[i - 1],
                 np.argmax(sums[1]) % 10 == lbls[i - 1]]
        else:
            p = [((a - 1) / a) * p[0] + (1 / a) * int(np.argmax(sums[0]) == lbls[i - 1]),
                 ((a - 1) / a) * p[1] + (1 / a) * int(np.argmax(sums[1]) % 10 == lbls[i - 1])]
        
        perfs.append([item * 100 for item in p])
        
        print('Performance on iteration %d: (%.2f, %.2f)' % (i / change_interval, p[0] * 100, p[1] * 100))
        
    for m in spike_monitors:
        spike_monitors[m].reset_()
    
    if plot:
        if i == 0:
            spike_ims, spike_axes = plot_spikes(spike_record)
            weights_im = plot_weights(get_square_weights(econn.w, sqrt, side=28))

            fig, ax = plt.subplots()
            im = ax.matshow(torch.stack([avg_rates, target_rates]), cmap='hot_r')
            ax.set_xticks(()); ax.set_yticks([0, 1])
            ax.set_yticklabels(['Actual', 'Targets'])
            ax.set_aspect('auto')
            ax.set_title('Difference between target and actual firing rates.')
            
            plt.tight_layout()
            
            fig2, ax2 = plt.subplots()
            line2, = ax2.semilogy(distances, label='Exc. distance')
            ax2.axhline(0, ls='--', c='r')
            ax2.set_title('Sum of squared differences over time')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Sum of squared differences')
            plt.legend()
            
            fig3, ax3 = plt.subplots()
            
            lines = []
            for perf, label in zip(np.array(perfs).T, ['all fired exc', 'max fired exc']):
                line, = ax3.plot(range(len(perf)), perf, label=label)
                lines.append(line)
            
            ax3.set_title('Performance over time')
            ax3.set_xlabel('Examples')
            ax3.set_ylabel('Classification accuracy')
            plt.grid(); plt.legend()
            
            plt.pause(1e-8)

        elif (i + 1) % plot_interval == 0:
            spikes_ims, spike_axes = plot_spikes(spike_record, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(get_square_weights(econn.w, sqrt, side=28), im=weights_im)

            im.set_data(torch.stack([avg_rates, target_rates]))
            
            if not len(distances) > change_interval:
                line2.set_xdata(range(len(distances)))
                line2.set_ydata(distances)
            else:
                line2.set_xdata(range(len(distances) - change_interval, len(distances)))
                line2.set_ydata(distances[-change_interval:])
            
            ax2.relim() 
            ax2.autoscale_view(True, True, True)
            
            for perf, line in zip(np.array(perfs).T, lines):
                line.set_xdata(range(len(perf)))
                line.set_ydata(perf)
            
            ax3.relim() 
            ax3.autoscale_view(True, True, True)
            
            plt.pause(1e-8)

    i += 1
