import gymnasium as gym

import reward_shaping as rs


def test_bad_wrapper_init():
    env = gym.make("FrozenLake-v1")
    wrapped = rs.RewardGettingCloser(env, lambda x: 0)
    assert wrapped


def test_PBRS_wrapper_init():
    wrapped = rs.PBRS(gym.make("FrozenLake-v1"), potential=lambda x: 10, discount=0.95)
    assert wrapped


def test_PBRS_wrapper_expected_reward():
    wrapped = rs.PBRS(
        gym.make("FrozenLake-v1", is_slippery=False),
        potential=lambda x: 10,
        discount=0.95,
    )
    wrapped.reset()
    next_obs, rew, term, trunc, _ = wrapped.step(1)
    assert next_obs == 4
    assert not term
    assert not trunc
    assert rew == (10 * 0.95 - 10)
    assert wrapped.curr_obs == 4
    wrapped.reset()
    assert wrapped.curr_obs == 0
    next_obs, rew, term, trunc, _ = wrapped.step(1)
    assert next_obs == 4
    assert not term
    assert not trunc
    assert rew == (10 * 0.95 - 10)
    assert wrapped.curr_obs == 4
