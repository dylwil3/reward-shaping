from gymnasium.core import Wrapper, Env
from typing import Callable, Any, SupportsFloat, Optional


class PBRS(Wrapper):
    """A wrapper for Gymnasium environments that modifies the reward by a potential.

    Implementation of Ng-Harada-Russell 'Policy invariance under reward transformations.'
    Given a potential function $\Phi: \text{States}-->\mathbb{R}$, we modify the reward function
    by the formula
        $$
        R_{\text{new}}(s,a,s') = R(s,a,s') + \gamma \cdot \Phi(s') - \Phi(s).
        $$
    """

    def __init__(
        self,
        env: Env,
        potential: Callable[[Any], float],
        discount: float,
    ):
        super().__init__(env)
        self.potential = potential
        self.discount = discount

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.curr_obs = obs
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        new_obs, rew, term, trunc, info = super().step(action)
        rew += self.discount * self.potential(new_obs) - self.potential(self.curr_obs)
        self.curr_obs = new_obs
        return new_obs, rew, term, trunc, info


class RewardGettingCloser(Wrapper):
    """A *bad* wrapper that rewards agents for moving closer to goal.

    Used to illustrate some pitfalls of reward shaping.
    """

    def __init__(self, env: Env, dist_to_goal: Callable[[Any], float]):
        super().__init__(env)
        self.dist_to_goal = dist_to_goal

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.curr_obs = obs
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        new_obs, rew, term, trunc, info = super().step(action)
        bonus = max(0, self.dist_to_goal(self.curr_obs) - self.dist_to_goal(new_obs))
        self.curr_obs = new_obs
        return new_obs, rew + bonus, term, trunc, info
