import numpy as np
import gym
import pybullet_envs
from gym import wrappers
import os

# Hyper Parameters for Augmented Random Search
class HyperParams():
    def __init__(self,
                 n_steps=1000,
                 episode_length=2000,
                 learning_rate=0.01,
                 num_deltas=20,
                 num_best_deltas=10,
                 noise=0.05,
                 env_name="HalfCheetahBulletEnv-v0",
                 seed=3,
                 record_every=50):
        self.n_steps = n_steps # Total number of steps in Training
        self.episode_length = episode_length # maximum length of episode
        self.learning_rate = learning_rate # Learning rate
        self.num_deltas = num_deltas # Number of clones created
        assert num_best_deltas <= num_deltas
        self.num_best_deltas = num_best_deltas # Number of clones selected for Learning
        self.noise = noise # Random noise
        self.env_name = env_name # Enviornment Name
        self.seed = seed # Seed for reproducibility
        self.record_every = record_every # Record after every 50 steps

# Normalization Class        
class Normalizer():
    def __init__(self, n_inputs):
        self.n = np.zeros(n_inputs) # Steps into the training
        self.mean = np.zeros(n_inputs) # Mean - Running Average
        self.mean_diff = np.zeros(n_inputs) # Last observed mean
        self.var = np.zeros(n_inputs) # Variance of input

    # Calculates Necessary stats from data
    def observe(self, x):
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(1e-03)

    # Normalize the data between [0-1]
    def normalize(self, x):
        obs_std = np.sqrt(self.var)
        return (x - self.mean) / obs_std

# Policy used by the Agent
class Policy():
    # Weights and HyperParameter declaration
    def __init__(self, n_inputs, n_outputs, hp):
        self.theta = np.zeros((n_outputs, n_inputs))
        self.hp = hp

    # Sample random noise
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    # Evaluate the policy
    def evaluate(self, x, deltas = None, direction = None):
        if direction is None:
            return self.theta.dot(x) # Inference from learned weights
        elif direction == '+':
            return (self.theta + self.hp.noise * deltas).dot(x) # Inference from positive noise weights 
        elif direction == '-':
            return (self.theta - self.hp.noise * deltas).dot(x) # Inference from negative noise weights

    # Update the Policy based on best clones (Rollouts)
    def update(self, best_rollouts, sigma_rewards):
        step = np.zeros_like(self.theta)
        for r_pos, r_neg, delta in best_rollouts:
            step += (r_pos - r_neg) * delta

        # Update Rule
        self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step
        

# Main Training Class for ARS        
class ARSTrainer():
    def __init__(self, hp=None,
                input_size=None, 
                output_size=None, 
                policy=None, 
                normalizer=None, 
                monitor_dir=None):
        self.hp = hp or HyperParams() #HyperParameters used
        np.random.seed(self.hp.seed) # Random seed initialize
        self.env = pybullet_envs.make(self.hp.env_name) #Making Environment
        # Setting Monitor to observe how agent plays
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.in_size = input_size or self.env.observation_space.shape[0] # Input size
        self.out_size = output_size or self.env.action_space.shape[0] # Output size
        self.policy = policy or Policy(self.in_size, self.out_size, self.hp) # Policy to be used
        self.normalizer = normalizer or Normalizer(self.in_size) # Normalizer to use
        self.record_video = False

    # Plays the Episode given direction and noises    
    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        done= False
        num_plays = 0
        total_reward = 0
        while not done and num_plays < self.hp.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state)
            action = self.policy.evaluate(state, direction=direction, deltas=delta)
            state, reward, done, _ = self.env.step(action)
            total_reward += max(min(reward, 1), -1)
            num_plays += 1
            
        return total_reward

    # Main Training loop
    def train(self):
        for step in range(self.hp.n_steps):
            # Sample random noise
            deltas = self.policy.sample_deltas()
            pos_rewards = [0] * self.hp.num_deltas
            neg_reward = [0] * self.hp.num_deltas

            # Play episode with sampled random noise in both positive and negative direction of noise
            for k in range(self.hp.num_deltas):
                pos_rewards[k] = self.explore('+', delta=deltas[k])
                neg_reward[k] = self.explore('-', delta=deltas[k])

            # Calculating standard deviation for all rewards                
            sigma_reward = np.array(pos_rewards + neg_reward).std()

            # Sorting (r_pos, r_neg, delta) rollouts in decreasing order and selecting best rollouts to be used to update the policy
            scores = {k : max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(pos_rewards, neg_reward))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse =True)[:self.hp.num_best_deltas]
            rollouts = [(pos_rewards[k], neg_reward[k], deltas[k]) for k in order]

            # Update Policy
            self.policy.update(rollouts, sigma_reward)

            # Record After every 50 steps
            if step % self.hp.record_every == 0:
                self.record_video = True

            # Print progress
            reward_evaluation = self.explore()
            print("Step:", step, "Reward:", reward_evaluation)
            self.record_video = False
            
# Creats directory for videos         
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Main code
if __name__ == '__main__':
    videos_dir = mkdir('.', 'videos')
    monitor_dir = mkdir(videos_dir, "HalfCheetahBulletEnv-v0")
    hp = HyperParams()
    trainer = ARSTrainer(hp=hp, monitor_dir=monitor_dir)
    trainer.train()
        
