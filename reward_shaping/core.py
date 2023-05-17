"""Reward shaping package"""

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from typing import Any, Optional, List, DefaultDict, Dict
from collections import defaultdict
from copy import deepcopy

import json
import pathlib

from .plotting import plot_evals


class Learner:
    """Abstract class for an agent in a MDP.

    For now this library only contains the tabular
    Q-learning instance. It may be expanded in the future."""

    def __init__(self) -> None:
        ...

    def act(self, obs) -> int:
        raise NotImplementedError

    def off_act(self, obs) -> int:
        raise NotImplementedError

    def update(self, obs, action, next_obs, rew, term) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class DataManager:
    """Data collection and saving for experiments.

    Records evaluations over many training runs, as well
    as, for each run and state, the first timestep that an
    update to the value function occurs. (This only makes
    sense for Q-learning; may be changed later.)"""

    def __init__(self) -> None:
        self.value_update_timesteps = []
        self.evaluations = []
        self.curr_run_evaluations = []
        self.curr_run_value_update_timesteps = defaultdict(lambda: float("inf"))

    def update_data(self) -> None:
        self.evaluations.append(self.curr_run_evaluations)
        self.curr_run_evaluations = []
        self.value_update_timesteps.append(self.curr_run_value_update_timesteps)
        self.curr_run_value_update_timesteps = defaultdict(lambda: float("inf"))

    def save(self, folder: str):
        """Save data to given folder.

        Args
            folder: Name of folder (omit trailing backslash)
        Returns
            None. Saves evaluations to csv file named 'evaluations.csv'
            and saves value update timesteps as
            'value_update_timesteps.json'.
        """
        df = pd.DataFrame(self.evaluations)
        df.to_csv(folder + "/evaluations.csv", mode="w")
        with open(folder + "/value_update_timesteps.json", "w") as f:
            json.dump(self.value_update_timesteps, fp=f)


class Experiment:
    """Class for running a specified experiment, collecting data, etc."""

    def __init__(
        self,
        name: str,
        env_id: str,
        env_options: dict,
        q_learner_params: dict,
        modification: Optional[gym.Wrapper] = None,
        modification_params: dict = {},
    ) -> None:
        self.name = name
        self.env_id = env_id
        self.env_options = env_options
        self.q_learner_params = q_learner_params
        self.modification = modification
        self.modification_params = modification_params
        self.initial_qtable = None

        self.folder = f"logs/experiment_{self.name}"
        p = pathlib.Path(self.folder)
        p.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        runs: int,
        episodes_per_run: int,
        save_rate: int = 100,
    ) -> None:
        """Performs training run a given number of times, saving along the way.

        Args:
            runs: Number of training runs.
            episodes_per_run: Number of episodes to train for each run.
            save_rate: How often we save the results of the run (default every 100).

        Returns
            None. Saves csv and json files as in `DataManager` class
            every save_rate many runs, and always saves at the end
            (overwriting files each time.)
        """
        self.dm = DataManager()
        for run in tqdm(range(runs)):
            self.single_run(episodes=episodes_per_run)
            self.dm.update_data()
            if run % save_rate == 0:
                self.save()
        self.save()

    def save(self):
        self.dm.save(self.folder)
        self.learner.save_qtable(self.folder)

    def record_video(self):
        """Records and saves video of current Q-learner in environment.

        Found at `logs/experiment_{experiment name}/rl-video-episode-0.mp4`.
        """
        self.evaluate(video=True)

    def single_run(self, episodes: int, eval_rate: int = 1) -> None:
        """Perform a single training run.

        Args
            episodes: Number of episodes to train the agent for.
            eval_rate: How often to evaluate the agent.
        Returns: None

        Creates new instance of environment for the run and
        closes it at the end. When evaluating, again
        creates a new instance (see `evaluate` method.)
        """
        env = gym.make(self.env_id, **self.env_options)
        if self.modification:
            env = self.modification(
                env,
                **self.modification_params,
            )
        self.learner = QLearner(**self.q_learner_params)
        if self.initial_qtable:
            self.learner.load_qtable(self.initial_qtable)
        self.curr_timestep = 0
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "Only discrete action spaces are supported for now."
        for episode in tqdm(range(episodes), leave=False):
            obs, _ = env.reset()
            done = False
            while not done:
                action = self.learner.off_act(obs)
                next_obs, rew, term, trunc, _ = env.step(action)
                self.curr_timestep += 1
                self.learner.update(obs, action, next_obs, rew, term)
                if (
                    self.dm.curr_run_value_update_timesteps[obs] == float("inf")
                ) and self.learner.curr_update_target != 0:
                    self.dm.curr_run_value_update_timesteps[obs] = self.curr_timestep
                done = term or trunc
                obs = next_obs
            if episode % eval_rate == 0:
                self.dm.curr_run_evaluations.append(self.evaluate())
        env.close()

    def evaluate(self, video=False) -> float:
        """Evaluate agent on task, optionally recording a video.

        Args
            video: Boolean, whether to record a video.
        Returns
            Total reward obtained in one episode of play.

        Creates a new instance of environment to train on.
        """
        rewards = 0
        temp_options = self.env_options
        if not video:
            env = gym.make(self.env_id, **temp_options)
        elif video:
            temp_options["render_mode"] = "rgb_array"
            env = RecordVideo(
                gym.make(self.env_id, **temp_options), self.folder, disable_logger=True
            )
        obs, _ = env.reset()
        done = False
        while not done:
            action = self.learner.act(obs)
            next_obs, rew, term, trunc, _ = env.step(action)
            rewards += rew
            done = term or trunc
            obs = next_obs
        env.close()
        return rewards

    def plot_evals(self) -> None:
        df = pd.DataFrame(self.dm.evaluations).T
        return plot_evals(df=df)

    def set_initial_qtable(self, qtable: defaultdict) -> None:
        self.initial_qtable = deepcopy(qtable)

    def __repr__(self) -> str:
        """Prints info about the experiment."""
        msg = []
        msg.append(f"Experiment {self.name}")
        msg.append("_______________")
        msg.append(f"Task: {self.env_id}")
        return "\n".join(msg)


class QLearner(Learner):
    """An agent learning from environment via the tabular Q-learning algorithm.

    Vanilla implementation of tabular Q-learning. The Q-table
    is stored in a default dictionary to be agnostic about
    the observation space (though it should be some
    flavor of discrete). The value at an observation
    is a numpy array- all zero by default.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        lr: float = 0.7,
        eps_init: float = 1,
        eps_final: float = 0.1,
        eps_decay: float = 0.0005,
        discount: float = 0.95,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.eps = eps_init
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.discount = discount
        self.action_space = action_space
        self.qtable = defaultdict(lambda: np.zeros(self.action_space.n))

    def off_act(self, obs) -> int:
        """Off-policy action for exploration.

        At the moment we use epsilon-greedy only. We may
        later allow more options (e.g. Boltzmann, mellowmax, etc.)
        """
        if np.random.uniform() < self.eps:
            return self.action_space.sample()
        else:
            self.act(obs)
        self.decay_eps()

    def act(self, obs) -> int:
        """On-policy, deterministic action given an observation."""
        return np.argmax(self.qtable[obs])

    def update_target(self, obs, action, next_obs, rew, term):
        """Computes the target as the td target minus the current estimate.

        Also records this answer in the property `curr_update_target`
        for the purposes of identifying the first nonzero update
        when logging.
        """
        if term:
            target = rew - self.qtable[obs][action]
        else:
            target = (
                rew
                + self.discount * np.max(self.qtable[next_obs])
                - self.qtable[obs][action]
            )
        self.curr_update_target = target
        return target

    def update(self, obs, action, next_obs, rew, term) -> None:
        self.qtable[obs][action] += self.lr * self.update_target(
            obs, action, next_obs, rew, term
        )

    def decay_eps(self) -> None:
        """Subtracts decay rate from epsilon until final value is reached."""
        self.eps = max(self.eps_final, self.eps - self.eps_decay)

    def load_qtable(self, qtable: defaultdict):
        """Loads pre-specified qtable, given as a defaultdict."""
        self.qtable = deepcopy(qtable)

    def save_qtable(self, folder: str):
        """Saves qtable to csv, columns labeled by *encountered* observations."""
        df = pd.DataFrame(self.qtable)
        df.to_csv(folder + "/qtable.csv")
