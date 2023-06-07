---
hide-toc: true
firstpage:
lastpage:
---

# Reward Shaping: learn faster by changing rewards.

**Reward Shaping** offers a simple API for running experiments in reinforcement learning that involve modifying the reward function of an environment. 

To run an experiment we:
- Give specifications, including a name and environment.
- Describe the desired reward shaping (e.g. Potential Based Reward Shaping).
- Give parameters for a tabular Q-learning agent
- Run and plot!

Here is an example:

```{code-block} python

import reward_shaping as rs
import gymnasium as gym
import matplotlib.pyplot as plt

# Define parameters for environment and reward shaping
experiment_specs = {
   "name": "Best_Experiment",
   # Environment specifications
   "env_id": "FrozenLake-v1",
   "env_options": {
      "is_slippery": False,
      "desc": None,
      "map_name": "4x4",
   },
   # Reward shaping modification (optional)
   "modification": rs.PBRS, # Potential based reward shaping
   "modification_params": {
      "potential": lambda x: (0.8)**(abs(x-15)),
      "discount": 0.95,
   }
}

# Define parameters for a tabular Q Learner
q_learner_params = {
   "lr": 0.7,
   "discount": 0.95,
   "eps_init": 1,
   "eps_final": 0.1,
   "eps_decay": 0.0005,
   "action_space": gym.spaces.Discrete(4),
}

# Initialize experiment
exp = rs.Experiment(**experiment_specs, q_learner_params=q_learner_params)

# Run
exp.run(runs=10,episodes_per_run=300)

# Plot evaluations
fig, ax = exp.plot_evals()
plt.show()
plt.clf()
```

```{toctree}
:hidden:
:caption: Introduction
```

```{toctree}
:hidden:
:caption: API

api/core
```
