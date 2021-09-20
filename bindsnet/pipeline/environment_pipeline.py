import itertools
from typing import Callable, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from bindsnet.analysis.pipeline_analysis import MatplotlibAnalyzer
from bindsnet.environment import Environment
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import AbstractInput
from bindsnet.pipeline.base_pipeline import BasePipeline


class EnvironmentPipeline(BasePipeline):
    # language=rst
    """
    Abstracts the interaction between ``Network``, ``Environment``, and environment
    feedback action.
    """

    def __init__(
        self,
        network: Network,
        environment: Environment,
        action_function: Optional[Callable] = None,
        encoding: Optional[Callable] = None,
        **kwargs,
    ):
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        :param environment: Arbitrary environment.
        :param action_function: Function to convert network outputs into environment inputs.
        :param encoding: Function to encoding input.

        Keyword arguments:

        :param str device: PyTorch computing device
        :param encode_factor: coefficient for the input before encoding.
        :param int num_episodes: Number of episodes to train for. Defaults to 100.
        :param str output: String name of the layer from which to take output.
        :param int render_interval: Interval to render the environment.
        :param int reward_delay: How many iterations to delay delivery of reward.
        :param int time: Time for which to run the network. Defaults to the network's
        :param int overlay_input: Overlay the last X previous input
        :param float percent_of_random_action: chance to choose random action
        :param int random_action_after: take random action if same output action counter reach

            timestep.
        """
        super().__init__(network, **kwargs)

        self.episode = 0

        self.env = environment
        self.action_function = action_function
        self.encoding = encoding

        self.accumulated_reward = 0.0
        self.reward_list = []

        # Setting kwargs.
        self.num_episodes = kwargs.get("num_episodes", 100)
        self.output = kwargs.get("output", None)
        self.render_interval = kwargs.get("render_interval", None)
        self.plot_interval = kwargs.get("plot_interval", None)
        self.reward_delay = kwargs.get("reward_delay", None)
        self.time = kwargs.get("time", int(network.dt))
        self.overlay_t = kwargs.get("overlay_input", 1)
        self.percent_of_random_action = kwargs.get("percent_of_random_action", 0.0)
        self.encode_factor = kwargs.get("encode_factor", 1.0)

        if torch.cuda.is_available() and self.allow_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # var for overlay process
        if self.overlay_t > 1:
            self.overlay_time_effect = torch.tensor(
                [i / self.overlay_t for i in range(1, self.overlay_t + 1)],
                dtype=torch.float,
                device=self.device,
            )
        self.overlay_start = True

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)

        # Set up for multiple layers of input layers.
        self.inputs = [
            name
            for name, layer in network.layers.items()
            if isinstance(layer, AbstractInput)
        ]

        self.action = torch.tensor(-1, device=self.device)
        self.last_action = torch.tensor(-1, device=self.device)
        self.action_counter = 0
        self.random_action_after = kwargs.get("random_action_after", self.time)

        self.voltage_record = None
        self.threshold_value = None
        self.reward_plot = None
        self.first = True

        self.analyzer = MatplotlibAnalyzer(**self.plot_config)

        if self.output is not None:
            self.network.add_monitor(
                Monitor(self.network.layers[self.output], ["s"], time=self.time),
                self.output,
            )

            self.spike_record = {
                self.output: torch.zeros((self.time, self.env.action_space.n)).to(
                    self.device
                )
            }

    def init_fn(self) -> None:
        pass

    def train(self, **kwargs) -> None:
        # language=rst
        """
        Trains for the specified number of episodes. Each episode can be of arbitrary
        length.
        """
        while self.episode < self.num_episodes:
            self.reset_state_variables()

            for _ in itertools.count():
                obs, reward, done, info = self.env_step()

                self.step((obs, reward, done, info), **kwargs)

                if done:
                    break

            print(
                f"Episode: {self.episode} - "
                f"accumulated reward: {self.accumulated_reward:.2f}"
            )
            self.episode += 1

    def env_step(self) -> Tuple[torch.Tensor, float, bool, Dict]:
        # language=rst
        """
        Single step of the environment which includes rendering, getting and performing
        the action, and accumulating/delaying rewards.

        :return: An OpenAI ``gym`` compatible tuple with modified reward and info.
        """
        # Render game.
        if (
            self.render_interval is not None
            and self.step_count % self.render_interval == 0
        ):
            self.env.render()

        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            self.last_action = self.action
            if torch.rand(1) < self.percent_of_random_action:
                self.action = torch.randint(
                    low=0, high=self.env.action_space.n, size=(1,)
                )[0]
            elif self.action_counter > self.random_action_after:
                if self.last_action == 0:  # last action was start b
                    self.action = 1  # next action will be fire b
                    tqdm.write(f"Fire -> too many times {self.last_action} ")
                else:
                    self.action = torch.randint(
                        low=0, high=self.env.action_space.n, size=(1,)
                    )[0]
                    tqdm.write(f"too many times {self.last_action} ")
            else:
                self.action = self.action_function(self, output=self.output)

            if self.last_action == self.action:
                self.action_counter += 1
            else:
                self.action_counter = 0

        # Run a step of the environment.
        obs, reward, done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            reward = self.rewards[-1]

        # Accumulate reward.
        self.accumulated_reward += reward

        info["accumulated_reward"] = self.accumulated_reward

        return obs, reward, done, info

    def step_(
        self, gym_batch: Tuple[torch.Tensor, float, bool, Dict], **kwargs
    ) -> None:
        # language=rst
        """
        Run a single iteration of the network and update it and the reward list when
        done.

        :param gym_batch: An OpenAI ``gym`` compatible tuple.
        """
        obs, reward, done, info = gym_batch

        if self.overlay_t > 1:
            if self.overlay_start:
                self.overlay_last_obs = (
                    obs.view(obs.shape[2], obs.shape[3]).clone().to(self.device)
                )
                self.overlay_buffer = torch.stack(
                    [self.overlay_last_obs] * self.overlay_t, dim=2
                ).to(self.device)
                self.overlay_start = False
            else:
                obs = obs.to(self.device)
                self.overlay_next_stat = torch.clamp(
                    self.overlay_last_obs - obs, min=0
                ).to(self.device)
                self.overlay_last_obs = obs.clone()
                self.overlay_buffer = torch.cat(
                    (
                        self.overlay_buffer[:, :, 1:],
                        self.overlay_next_stat.view(
                            [
                                self.overlay_next_stat.shape[2],
                                self.overlay_next_stat.shape[3],
                                1,
                            ]
                        ),
                    ),
                    dim=2,
                )
            obs = (
                torch.sum(self.overlay_time_effect * self.overlay_buffer, dim=2)
                * self.encode_factor
            )

        # Place the observations into the inputs.
        if self.encoding is None:
            obs = obs.unsqueeze(0).unsqueeze(0)
            obs_shape = torch.tensor([1] * len(obs.shape[1:]), device=self.device)
            inputs = {
                k: self.encoding(
                    obs.repeat(self.time, *obs_shape).to(self.device),
                    device=self.device,
                )
                for k in self.inputs
            }
        else:
            obs = obs.unsqueeze(0)
            inputs = {
                k: self.encoding(obs, self.time, device=self.device)
                for k in self.inputs
            }

        # Run the network on the spike train-encoded inputs.
        self.network.run(inputs=inputs, time=self.time, reward=reward, **kwargs)

        if self.output is not None:
            self.spike_record[self.output] = (
                self.network.monitors[self.output].get("s").float()
            )

        if done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(
                    accumulated_reward=self.accumulated_reward,
                    steps=self.step_count,
                    **kwargs,
                )
            self.reward_list.append(self.accumulated_reward)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_state_variables()
        self.accumulated_reward = 0.0
        self.step_count = 0
        self.overlay_start = True
        self.action = torch.tensor(-1)
        self.last_action = torch.tensor(-1)
        self.action_counter = 0

    def plots(self, gym_batch: Tuple[torch.Tensor, float, bool, Dict], *args) -> None:
        # language=rst
        """
        Plot the encoded input, layer spikes, and layer voltages.

        :param gym_batch: An OpenAI ``gym`` compatible tuple.
        """
        if self.plot_interval is None:
            return

        obs, reward, done, info = gym_batch

        for key, item in self.plot_config.items():
            if key == "obs_step" and item is not None:
                if self.step_count % item == 0:
                    self.analyzer.plot_obs(obs[0, ...].sum(0))
            elif key == "data_step" and item is not None:
                if self.step_count % item == 0:
                    self.analyzer.plot_spikes(self.get_spike_data())
                    self.analyzer.plot_voltages(*self.get_voltage_data())
            elif key == "reward_eps" and item is not None:
                if self.episode % item == 0 and done:
                    self.analyzer.plot_reward(self.reward_list)

        self.analyzer.finalize_step()
