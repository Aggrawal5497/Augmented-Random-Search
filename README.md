# Augmented-Random-Search

ARS is a random search method for training linear policies for continuous control problems, based on the paper "Simple random search provides a competitive approach to reinforcement learning."

# Prerequisites
Our ARS implementation relies on Python 3, OpenAI Gym, PyBullet-env and Numpy. Every Dependency is easily installed using pip
- `pip install gym`
- `pip install pybullet`
- `pip install numpy`

# Training in ARS
The following pseudo-code describes how to train an agent using Augmented Random Search.
- Initialize Envirnment, Policy to use and HyperParameters
- Generate `num_deltas` number of random noise and create two clones of current weights, one by adding the generated noise and other by subtracting the generated noise.
- Play episodes with both the clones for every pair of clones generated.
- Collect a rollout tuples as (reward_positive, reward_negative, delta)
- Calculate Standard deviation for all rewards (`sigma_reward`)
- Sort the collected rollouts by maximum reward and select `num_best_deltas` number of rollouts
- Calculate step as `step = sum((r[+] - r[-]) * delta)` for each best rollout.
- Update weights as `theta += learning_rate / (num_best_deltas * sigma_reward) * step`.
- Evaluate the new weights by playing an episode to measure performance.
- Repeat Until desired performance is reached.
