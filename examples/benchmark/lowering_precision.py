import re
import os
import subprocess
from statistics import mean

precision_sample_size = 4
precisions = ['float16', 'float32']

folder = os.path.dirname(os.path.dirname(__file__))
script = os.path.join(folder, 'mnist', 'batch_eth_mnist.py')
data = {}
for precision in precisions:
    for _ in range(precision_sample_size):
        result = subprocess.run(
            f"python {script} --n_train 100 --batch_size 50 --n_test 10 --n_updates 1 --w_dtype {precision}",
            shell=True, capture_output=True, text=True
        )
        output = result.stdout
        time_match = re.search(r'Progress: 1 / 1 \((\d+\.\d+) seconds\)', output)
        memory_match = re.search(r'Memory consumption: (\d+)mb', output)
        data.setdefault(precision, []).append([
            time_match.groups()[0],
            memory_match.groups()[0]
        ])
        print("+")


def print_table(data):
    column_widths = [max(len(str(item)) for item in col) for col in zip(*data)]
    for row in data:
        formatted_row = " | ".join(f"{str(item):<{column_widths[i]}}" for i, item in enumerate(row))
        print(formatted_row)


average_time = {}
average_memory = {}
for precision, rows in data.items():
    print(f"precision: {precision}")
    table = [
        ['Time (sec)', 'GPU memory (Mb)']
    ] + rows
    avg_time = mean(map(lambda i: float(i[0]), rows))
    avg_memory = mean(map(lambda i: float(i[1]), rows))
    print_table(table)
    print(f"Average time: {avg_time}")
    print(f"Average memory: {avg_memory}")
    average_memory[precision] = avg_memory
    average_time[precision] = avg_time
    print('')