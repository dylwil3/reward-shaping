# def test_bad_wrapper_init():
#     env = gym.make("FrozenLake-v1")
#     wrapped = rs.RewardGettingCloser(env, lambda x: 0)
#     assert wrapped

# def test_PBRS_wrapper():
#     wrapped = rs.PBRS(gym.make("FrozenLake-v1"), potential=lambda x: 10, discount=0.95)
#     assert wrapped
