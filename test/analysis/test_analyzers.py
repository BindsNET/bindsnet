import os

import matplotlib.pyplot as plt
import torch

from bindsnet.analysis.pipeline_analysis import MatplotlibAnalyzer, TensorboardAnalyzer


class TestAnalyzer:
    """
    Sanity checks all plotting functions for analyzers
    """

    def test_init(self):
        ma = MatplotlibAnalyzer()
        assert plt.isinteractive()

        ta = TensorboardAnalyzer("./logs/init")

        # check to ensure path was written
        assert os.path.isdir("./logs/init")

        # check to ensure we can write data
        ta.writer.add_scalar("init_scalar", 100.0, 0)
        ta.writer.close()

    def test_plot_runs(self):
        ma = MatplotlibAnalyzer()
        ta = TensorboardAnalyzer("./logs/runs")

        for analyzer in [ma, ta]:
            obs = torch.rand(1, 28, 28)
            analyzer.plot_obs(obs)

            # 4 channels out, 1 channel in, 8x8 kernels
            conv_weights = torch.rand(4, 1, 8, 8)
            analyzer.plot_conv2d_weights(conv_weights)

            rewards = [0, 0, 0, 0, 0]
            analyzer.plot_reward(rewards)

            # Monitors have time as last dimension
            v = torch.rand(50, 1, 1, 28, 28)
            voltage_dict = {"X": v}
            threshold_dict = {"X": torch.tensor(0.75)}
            analyzer.plot_voltages(voltage_dict, threshold_dict)

            # The monitors have time as last dimension
            spikes = torch.rand(50, 1, 1, 28, 28) > 0.5
            spike_dict = {"X": spikes}
            analyzer.plot_spikes(spike_dict)

            analyzer.finalize_step()

        ta.writer.close()


if __name__ == "__main__":
    tester = TestAnalyzer()

    tester.test_init()
    tester.test_plot_runs()
