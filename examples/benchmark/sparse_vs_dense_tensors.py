import torch


assert torch.cuda.is_available(), 'Benchmark works only on cuda'
device = torch.device("cuda")


def create_spikes_tensor(percent_of_true_values, sparse):
    spikes_tensor = torch.bernoulli(
        torch.full((500, 500, 500), percent_of_true_values, device=device)
    ).bool()
    if sparse:
        spikes_tensor = spikes_tensor.to_sparse()

    torch.cuda.reset_peak_memory_stats(device=device)
    return round(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2))


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
