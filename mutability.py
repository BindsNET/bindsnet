import torch
from torch.multiprocessing.spawn import spawn


def layer_update(z, self):
    self.x += 1


class Network:
    def __init__(self):
        self.x = torch.tensor([1, 1, 1], device='cpu')

    def run(self):
        spawn(layer_update, args=(self,))
        print(self.x)

if __name__ == '__main__':
    Network().run()

