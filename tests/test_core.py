import pytest
import reward_shaping as rs
import gymnasium as gym
import numpy as np
import pandas as pd

from collections import defaultdict

from typing import List, Dict


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


def test_experiment_instantiation(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    assert exp


def test_experiment_properties_access(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    assert exp.env_id == "FrozenLake-v1"
    assert exp.env_options == {
        "desc": None,
        "map_name": "4x4",
        "is_slippery": False,
    }


def test_experiment_run_passes(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    exp.run(runs=2, episodes_per_run=2)


def test_experiment_single_run_passes(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    exp.dm = rs.DataManager()
    exp.single_run(episodes=2)


def test_experiment_repr(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    assert repr(exp) == "\n".join(
        ["Experiment Best_Ever", "_______________", "Task: FrozenLake-v1"]
    )


def test_learner_init():
    learner = rs.Learner()
    assert learner


def test_qlearner_init(example_qlearner_params: dict):
    qlearner = rs.QLearner(**example_qlearner_params)
    assert qlearner


def test_qlearner_load_qtable(example_qlearner_params: dict):
    qlearner = rs.QLearner(**example_qlearner_params)
    qtable = defaultdict(lambda: np.zeros(qlearner.action_space.n))
    qtable[0] = np.ones(qlearner.action_space.n)
    qlearner.load_qtable(qtable)
    assert qlearner.qtable == qtable
    assert qlearner.qtable[0][0] == 1
    assert qlearner.qtable[1][0] == 0


def test_qlearner_update(example_qlearner_params: dict):
    qlearner = rs.QLearner(**example_qlearner_params)
    qlearner.update(0, 0, 1, 100, True)
    qlearner.update(1, 0, 0, -100, False)
    assert qlearner.qtable[0][0] == 0.7 * 100
    assert qlearner.qtable[1][0] == -23.45


def test_data_manager_length_after_experiment(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    exp.run(runs=2, episodes_per_run=3)
    assert len(exp.dm.evaluations) == 2
    assert len(exp.dm.curr_run_evaluations) == 0
    assert len(exp.dm.evaluations[0]) == 3


def test_PBRS_wrapper():
    wrapped = rs.PBRS(gym.make("FrozenLake-v1"), potential=lambda x: 10, discount=0.95)
    assert wrapped


def test_experiment_record_video(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    exp.run(runs=2, episodes_per_run=3)
    exp.record_video()


def test_plot_evals():
    df = pd.DataFrame(
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 2, 3],
        ]
    )
    ax = rs.plot_evals(df)
    assert ax


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


def test_default_dict_to_list():
    dd = defaultdict(lambda: float("inf"))
    dd[0] = 1
    dd[1] = 2
    dd[2] = 3
    e = rs.default_dict_to_list(dd, 2)
    assert e[0] == 1
    assert e[1] == 2
    assert e[2] == 3
    assert e[3] == float("inf")
    assert len(e) == 4


def test_update_dict_to_array():
    dd = defaultdict(lambda: float("inf"))
    dd[0] = 1
    dd[1] = 2
    dd[2] = 3
    ff = defaultdict(lambda: float("inf"))
    ff[0] = 10
    ff[1] = 9
    v = [dd, ff]
    flat = rs.update_dict_to_array(v, 2)
    assert len(flat[0]) == 4
    assert len(flat[1]) == 4
    assert len(flat) == 2
    assert flat[0][3] == float("inf")
    assert flat[1][2] == float("inf")
    assert flat[1][0] == 10
    assert flat[0][1] == 2
    assert isinstance(flat, np.ndarray)
    assert isinstance(flat[0], np.ndarray)


def test_visualize_frozen_updates():
    dd = defaultdict(lambda: float("inf"))
    dd[0] = 1
    dd[1] = 2
    dd[2] = 3
    ff = defaultdict(lambda: float("inf"))
    ff[1] = 9
    ff[2] = 8
    ax, len1, len2 = rs.visualize_frozen_updates([dd, ff], 2)
    assert ax.get_title() == "Timesteps before Updating Value"
    assert len1 == 1
    assert len2 == 2


def test_filter_inf_start_value():
    arr = [
        [float("inf"), 8, 9],
        [1, 2, 3],
        [float("inf"), 5, 7],
        [0, 0, 0],
    ]
    arr = np.array(arr)
    finite, infinite = rs.filter_inf_start_value(arr)
    assert (
        infinite
        == np.array(
            [
                [float("inf"), 8, 9],
                [float("inf"), 5, 7],
            ]
        )
    ).all()
    assert (
        finite
        == np.array(
            [
                [1, 2, 3],
                [0, 0, 0],
            ]
        )
    ).all()


def test_bad_wrapper_init():
    env = gym.make("FrozenLake-v1")
    wrapped = rs.RewardGettingCloser(env, lambda x: 0)
    assert wrapped


def test_experiment_set_initial_qtable(example_experiment_params: dict):
    exp = rs.Experiment(**example_experiment_params)
    number = example_experiment_params["q_learner_params"]["action_space"].n
    qtable = defaultdict(lambda: np.zeros(number))
    qtable[0] = np.ones(number)
    exp.set_initial_qtable(qtable)
    exp.run(runs=2, episodes_per_run=3)
    assert exp.initial_qtable == qtable
