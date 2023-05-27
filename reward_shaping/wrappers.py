"""Two wrappers for reward shaping.

Potential Based Reward Shaping Wrapper (PBRS): Updates reward function according to
    $$
    R_{\text{new}}(s,a,s') = R(s,a,s') + \gamma \cdot \Phi(s') - \Phi(s).
    $$
    where $\Phi$ is a specified 'potential' function on the observation space.

RewardGettingCloser: For pedagogical use only, a *bad* reward update of the form
    $$
    R_{\text{new}} += max(0, \text{change in distance to goal})
    $$
"""

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
        """
        Args
            env: A gymnasium environment to wrap.
            potential: A function with domain the
                observation/state space and codomain the real numbers.
            discount: The discount used for converting the potential to the reward.
        
        Note: In the original paper it is implicit that the discount specified
        here agrees with the discount used when training. However, it may be more
        beneficial for training to make the potential discount value *smaller*
        than the training discount. 
        """
        super().__init__(env)
        self.potential = potential
        self.discount = discount

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # We need to track the current observation when updating rewards
        self.curr_obs = obs
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Adjusts reward function according to R += \gamma*potential(s') - potential(s).
        Also updates `curr_obs` as side-effect.
        """
        new_obs, rew, term, trunc, info = super().step(action)
        rew += self.discount * self.potential(new_obs) - self.potential(self.curr_obs)
        self.curr_obs = new_obs
        return new_obs, rew, term, trunc, info


class RewardGettingCloser(Wrapper):
    """A *bad* wrapper that rewards agents for moving closer to goal.

    Used to illustrate some pitfalls of reward shaping.
    """

    def __init__(self, env: Env, dist_to_goal: Callable[[Any], float]):
        """
        Args
            env: A Gymnasium environment
            dist_to_goal: A function from observations to reals, meant
                to be a notion of the distance to the goal state.
        """
        super().__init__(env)
        self.dist_to_goal = dist_to_goal

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        # We need to track the current observation when updating rewards
        self.curr_obs = obs
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Adjusts reward function according to R+=(nonnegative change in distance to goal).
        Also updates `curr_obs` as side-effect.
        """
        new_obs, rew, term, trunc, info = super().step(action)
        bonus = max(0, self.dist_to_goal(self.curr_obs) - self.dist_to_goal(new_obs))
        self.curr_obs = new_obs
        return new_obs, rew + bonus, term, trunc, info
