import torch
import numpy as np
import argparse
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.network import Network
from bindsnet.encoding import poisson
from bindsnet.learning import MSTDP, PostPre




def main(seed = 0):
    print('----------------------- seed = ', seed, '--------------------------------------')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(seed)

    tau = 20
    plot_interval = 1
    dt = 1
    time = 500
    epochs = 200

    ONE = torch.ones(30) * 40
    ZERO = torch.zeros(30)


    Inp = Input(n=60, traces=True)
    Hidden = LIFNodes(n=60, thresh=-54.0, rest=-70.0, reset=-70.0, refrac=0, decay=1 - np.exp(-1 / tau), traces=True)
    Output = LIFNodes(n=1, thresh=-54.0, rest=-70.0, reset=-70.0, refrac=0, decay=1 - np.exp(-1 / tau), traces=True)

    layers = {'Input': Inp, 'Hidden': Hidden, 'Output': Output}

    inhibitory_neuron_indices = torch.randperm(60)[0:30]
    wmax = torch.ones(60, 1) * 5.0
    wmax[inhibitory_neuron_indices, :] = 0
    wmin = torch.zeros(60, 1)
    wmin[inhibitory_neuron_indices, :] = -5.0

    input_hidden_conn = Connection(
        source=layers['Input'], target=layers['Hidden'],
        wmax=wmax @ torch.ones(1, 60),
        wmin=wmin @ torch.ones(1, 60),
        update_rule=PostPre,
        nu=0.1
    )

    hidden_output_conn = Connection(
        source=layers['Hidden'], target=layers['Output'], wmax=5.0, wmin=0.0, update_rule=PostPre, nu=0.1
    )


    spikes = {}
    for layer in layers:
        spikes[layer] = Monitor(layers[layer], ['s'], time=time)

    voltages = {}
    for layer in set(layers.keys()) - {'Input'}:
        voltages[layer] = Monitor(layers[layer], ['v'], time=time)

    network = Network(dt=dt)

    for layer in layers:
        network.add_layer(layers[layer], name=layer)

    for layer in layers:
        network.add_monitor(spikes[layer], name=layer)

    network.add_connection(input_hidden_conn, source='Input', target='Hidden')
    network.add_connection(hidden_output_conn, source='Hidden', target='Output')


    data_x = [torch.cat((poisson(ZERO, time, dt), poisson(ZERO, time, dt)), dim=1),
              torch.cat((poisson(ZERO, time, dt), poisson(ONE, time, dt)), dim=1),
              torch.cat((poisson(ONE, time, dt), poisson(ZERO, time, dt)), dim=1),
              torch.cat((poisson(ONE, time, dt), poisson(ONE, time, dt)), dim=1)
              ]

    data_y = [0, 1, 1, 0]

    rewards = np.zeros(epochs)

    for epoch in range(epochs):
        seq = torch.randperm(4)
        for i in seq:
            inpts = {'Input':data_x[i]}
            network.run(inpts=inpts, time=time, output=data_y[i])
            if data_y[i] == 1:
                rewards[epoch] += torch.sum(spikes['Output'].get('s'))
            elif data_y[i] == 0:
                rewards[epoch] -= torch.sum(spikes['Output'].get('s'))
            network.reset_()
        print("Epoch: ", epoch, 'Reward: ', int(rewards[epoch]))

    # torch.save(network, open('trained_network_' + str(seed), 'wb'))
    np.savetxt('rewards_' + str(seed) + '.txt', rewards)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = vars(parser.parse_args())
    main(**args)