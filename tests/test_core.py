from collections import defaultdict
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

import reward_shaping as rs


@pytest.fixture
def example_qlearner_params() -> dict:
    return {
        "lr": 0.7,
        "discount": 0.95,
        "eps_init": 1,
        "eps_final": 0.1,
        "eps_decay": 0.0005,
        "action_space": gym.spaces.Discrete(4),
    }


@pytest.fixture
def example_experiment_params(example_qlearner_params: dict) -> dict:
    return {
        "name": "Best_Ever",
        "env_id": "FrozenLake-v1",
        "env_options": {
            "desc": None,
            "map_name": "4x4",
            "is_slippery": False,
        },
        "q_learner_params": example_qlearner_params,
        "modification": rs.PBRS,
        "modification_params": {
            "potential": lambda x: 100,
            "discount": 0.95,
        },
    }


class TestLearner:
    """Test Learner base class."""

    learner = rs.Learner()

    def test_init(self):
        assert self.learner is not None

    def test_not_implemented(self):
        """Test all methods raise `NotImplementedError`"""
        with pytest.raises(NotImplementedError):
            self.learner.act(1)
        with pytest.raises(NotImplementedError):
            self.learner.off_act(1)
        with pytest.raises(NotImplementedError):
            self.learner.update(1, 2, 3, 4, False)
        with pytest.raises(NotImplementedError):
            self.learner.reset()


class TestDataManager:
    """Test Data Manager base class."""

    dm = rs.DataManager()

    def test_init(self):
        """Test properties initialize correctly."""
        assert self.dm is not None
        assert self.dm.value_update_timesteps == []
        assert self.dm.evaluations == []
        assert self.dm.curr_run_evaluations == []
        assert isinstance(self.dm.curr_run_value_update_timesteps, defaultdict)

    def test_update_data(self):
        """Test data is updated correctly."""
        self.dm.curr_run_evaluations.append(10)
        self.dm.curr_run_value_update_timesteps[1] = 20
        self.dm.update_data()
        assert self.dm.evaluations[0] == [10]
        assert self.dm.value_update_timesteps[0][1] == 20
        assert self.dm.curr_run_value_update_timesteps[0] == float("inf")
        assert self.dm.curr_run_evaluations == []

    def test_save_passes(self):
        """Test we can save without incident."""
        # TODO Make and cleanup all files during testing.
        self.dm.save("tmp")


class TestExperiment:
    """Test Experiment class."""

    exp = rs.Experiment(
        **{
            "name": "Best_Ever",
            "env_id": "FrozenLake-v1",
            "env_options": {
                "desc": None,
                "map_name": "4x4",
                "is_slippery": False,
            },
            "q_learner_params": {
                "lr": 0.7,
                "discount": 0.95,
                "eps_init": 1,
                "eps_final": 0.1,
                "eps_decay": 0.0005,
                "action_space": gym.spaces.Discrete(4),
            },
            "modification": rs.PBRS,
            "modification_params": {
                "potential": lambda x: 100,
                "discount": 0.95,
            },
        }
    )

    def test_init(self):
        assert self.exp is not None

    def test_experiment_properties_access(self):
        assert self.exp.name == "Best_Ever"
        assert self.exp.env_id == "FrozenLake-v1"
        assert self.exp.env_options == {
            "desc": None,
            "map_name": "4x4",
            "is_slippery": False,
        }
        assert self.exp.q_learner_params == {
            "lr": 0.7,
            "discount": 0.95,
            "eps_init": 1,
            "eps_final": 0.1,
            "eps_decay": 0.0005,
            "action_space": gym.spaces.Discrete(4),
        }
        assert self.exp.modification == rs.PBRS
        assert self.exp.initial_qtable is None
        assert self.exp.folder == "logs/experiment_Best_Ever"

    def test_experiment_record_video_passes(self):
        # TODO: Cleanup files here and below.
        self.exp.run(runs=2, episodes_per_run=3)
        self.exp.record_video()

    def test_experiment_run_passes(self):
        self.exp.run(runs=2, episodes_per_run=2)
        self.exp.run(runs=2, episodes_per_run=2, seed=1)
        self.exp.run(runs=2, episodes_per_run=2, seed=[1, 3])

    def test_experiment_single_run_passes(self):
        self.exp.dm = rs.DataManager()
        self.exp.single_run(episodes=2, seed=1)

    def test_experiment_evaluations_reproducibility(
        self, example_experiment_params: dict
    ):
        new_params = deepcopy(example_experiment_params)
        new_params["modification_params"] = None
        new_params["modification"] = None
        new_exp = rs.Experiment(**new_params)
        new_exp.run(runs=2, episodes_per_run=400, seed=[1, 3])
        first_results = deepcopy(new_exp.dm.evaluations)
        new_exp.run(runs=2, episodes_per_run=400, seed=[1, 3])
        second_results = deepcopy(new_exp.dm.evaluations)
        assert first_results[0][200:] == second_results[0][200:]

    def test_experiment_repr(self):
        assert repr(self.exp) == "\n".join(
            ["Experiment Best_Ever", "_______________", "Task: FrozenLake-v1"]
        )

    def test_experiment_set_initial_qtable(self, example_qlearner_params: dict):
        number = example_qlearner_params["action_space"].n
        qtable = defaultdict(lambda: np.zeros(number))
        qtable[0] = np.ones(number)
        self.exp.set_initial_qtable(qtable)
        self.exp.run(runs=2, episodes_per_run=3)
        for key in qtable:
            assert (self.exp.initial_qtable[key] == qtable[key]).all()  # pyright:ignore


class TestQLearner:
    qlearner = rs.QLearner(
        **{
            "lr": 0.7,
            "discount": 0.95,
            "eps_init": 1,
            "eps_final": 0.1,
            "eps_decay": 0.0005,
            "action_space": gym.spaces.Discrete(4),
            "seed": 1,
        }
    )

    def test_init(self):
        assert self.qlearner is not None
        assert isinstance(self.qlearner.action_space, gym.spaces.Discrete)
        assert self.qlearner.discount == 0.95
        assert self.qlearner.eps == 1
        assert self.qlearner.eps_final == 0.1
        assert self.qlearner.eps_decay == 0.0005
        assert (self.qlearner.qtable[1] == np.zeros(4)).all()

    def test_off_act(self):
        assert self.qlearner.off_act(obs=0, explore=0) == 1
        assert pytest.approx(self.qlearner.eps) == 1 - self.qlearner.eps_decay
        assert self.qlearner.off_act(obs=0, explore=2) == 0
        assert pytest.approx(self.qlearner.eps) == 1 - 2 * self.qlearner.eps_decay

    def test_act(self):
        assert self.qlearner.act(1) == 0

    def test_update_target(self):
        self.qlearner.qtable[0][0] = 10
        self.qlearner.qtable[1][0] = 20
        assert self.qlearner.update_target(0, 0, 1, 1, True) == -9
        assert self.qlearner.update_target(0, 0, 1, 1, False) == pytest.approx(
            1 + 0.95 * 20 - 10
        )

    def test_update(self):
        self.qlearner.qtable[0][0] = 10
        self.qlearner.qtable[1][0] = 20
        self.qlearner.update(0, 0, 1, 1, True)
        assert self.qlearner.qtable[0][0] == (pytest.approx(10 + 0.7 * (-9)))
        self.qlearner.qtable[0][0] = 10
        self.qlearner.update(0, 0, 1, 1, False)
        assert self.qlearner.qtable[0][0] == (
            pytest.approx(10 + (0.7) * (1 + 0.95 * 20 - 10))
        )

    def test_decay_eps(self):
        self.qlearner.eps = 1
        self.qlearner.decay_eps()
        assert self.qlearner.eps == 1 - 0.0005

    def test_qlearner_load_qtable(self):
        qtable = defaultdict(lambda: np.zeros(self.qlearner.action_space.n))
        qtable[0] = np.ones(self.qlearner.action_space.n)
        self.qlearner.load_qtable(qtable)
        for key in qtable:
            assert (self.qlearner.qtable[key] == qtable[key]).all()
        assert self.qlearner.qtable[0][0] == 1
        assert self.qlearner.qtable[1][0] == 0
        self.qlearner.update(0, 0, 1, 100, False)
        assert self.qlearner.qtable[0][0] == (pytest.approx(1 + (0.7) * (100 - 1)))


def test_data_manager_length_after_experiment(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    exp.run(runs=2, episodes_per_run=3)
    assert len(exp.dm.evaluations) == 2
    assert len(exp.dm.curr_run_evaluations) == 0
    assert len(exp.dm.evaluations[0]) == 3


def test_run_experiment_twice_data_lengths(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    exp.run(runs=2, episodes_per_run=3)
    assert len(exp.dm.value_update_timesteps) == 2
    assert len(exp.dm.evaluations) == 2
    assert len(exp.dm.evaluations[0]) == 3
    exp.run(runs=2, episodes_per_run=3)
    assert len(exp.dm.value_update_timesteps) == 2
    assert len(exp.dm.evaluations) == 2
    assert len(exp.dm.evaluations[0]) == 3
