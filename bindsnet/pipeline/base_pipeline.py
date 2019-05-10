from typing import Optional

from ..network import Network

class BasePipeline:
    """
    A generic pipeline that handles high level functionality
    """

    def __init__(self, network: Network, **kwargs):
        self.network = network
        self.save_dir = kwargs.get('save_dir', 'network.pt')

        self.plot_interval = kwargs.get('plot_interval', None)
        self.save_interval = kwargs.get('save_interval', None)

        self.step_count = 0

    def reset(self) -> None:
        """
        Reset the pipeline.
        """

        self.network.reset_()
        self.step_count = 0
        self.history = {i: torch.Tensor() for i in self.history}

    def _step(self, batch) -> None:
        self.step(batch)
        if self.print_interval is not None and self.step_count % self.print_interval == 0:
            print(f'Iteration: {self.iteration} (Time: {time.time() - self.clock:.4f})')
            self.clock = time.time()

        if self.plot_interval is not None and self.step_count % self.plot_interval == 0:
            self.plots(batch)

        if self.save_interval is not None and self.step_count % self.save_interval == 0:
            self.network.save(self.save_dir)

        if self.test_interval is not None and self.step_count % self.test_interval == 0:
            self.test()

        self.step_count += 1

    def step(self, batch):
        raise NotImplementedError('You need to provide a step method')

    def train(self):
        raise NotImplementedError('You need to provide a train method')

    def test(self):
        raise NotImplementedError('You need to provide a test method')

    def init_fn(self):
        raise NotImplementedError('You need to provide an init_fn method')

    def plots(self, batch, mode='train'):
        raise NotImplementedError('You need to provide a plots method')
