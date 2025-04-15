import torch
import time
import argparse

from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting


parser = argparse.ArgumentParser()
parser.add_argument("--benchmark_type", choices=['memory', 'runtime'], default='memory')
args = parser.parse_args()


assert torch.cuda.is_available(), 'Benchmark works only on cuda'
device = torch.device("cuda")
shape = (300, 500, 500)


def create_spikes_tensor(percent_of_true_values, sparse, return_memory_usage=True):
    spikes_tensor = torch.bernoulli(
        torch.full(shape, percent_of_true_values, device=device)
    ).bool()
    if sparse:
        spikes_tensor = spikes_tensor.to_sparse()

    if return_memory_usage:
        torch.cuda.reset_peak_memory_stats(device=device)
        return round(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))
    else:
        return spikes_tensor


def memory_benchmark():
    print('======================= ====================== ====================== ====================')
    print('Sparse (megabytes used) Dense (megabytes used) Ratio (Sparse/Dense) % % of non zero values')
    print('======================= ====================== ====================== ====================')
    percent_of_true_values = 0.005
    while percent_of_true_values < 0.1:
        result = {}
        for sparse in [True, False]:
            result[sparse] = create_spikes_tensor(percent_of_true_values, sparse)
        percent = round((result[True] / result[False]) * 100)

        row = [
            str(result[True]).ljust(23),
            str(result[False]).ljust(22),
            str(percent).ljust(22),
            str(round(percent_of_true_values * 100, 1)).ljust(20),
        ]
        print(' '.join(row))
        percent_of_true_values += 0.005

    print('======================= ====================== ====================== ====================')


def run(sparse):
    n_classes = 10
    proportions = torch.zeros((500, n_classes), device=device)
    rates = torch.zeros((500, n_classes), device=device)
    assignments = -torch.ones(500, device=device)
    spike_record = []
    for _ in range(5):
        tmp = torch.zeros(shape, device=device)
        spike_record.append(tmp.to_sparse() if sparse else tmp)

    spike_record_idx = 0

    delta = 0
    for _ in range(10):
        start = time.perf_counter()
        label_tensor = torch.randint(0, n_classes, (n_classes,), device=device)
        spike_record_tensor = torch.cat(spike_record, dim=0)
        all_activity(
            spikes=spike_record_tensor, assignments=assignments, n_labels=n_classes
        )
        proportion_weighting(
            spikes=spike_record_tensor,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )

        assignments, proportions, rates = assign_labels(
            spikes=spike_record_tensor,
            labels=label_tensor,
            n_labels=n_classes,
            rates=rates,
        )
        delta += time.perf_counter() - start
        spike_record[spike_record_idx] = create_spikes_tensor(
            0.03,
            sparse,
            return_memory_usage=False
        )
        spike_record_idx += 1
        if spike_record_idx == len(spike_record):
            spike_record_idx = 0
    return round(delta, 1)


def runtime_benchmark():
    print(f"Sparse runtime: {run(True)} seconds")
    print(f"Dense runtime: {run(False)} seconds")


if args.benchmark_type == 'memory':
    memory_benchmark()
else:
    runtime_benchmark()
