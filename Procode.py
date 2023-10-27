import torch as th
import torch.nn as nn
import os
import gymnasium
import panda_gym
from gymnasium import spaces

import sys
sys.modules["gym"] = gymnasium

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym
import gym_panda_reach

#env_id = "panda-reach-v0" # our env_id

# Create the env
env = gym.make("panda-reach-v0")

# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

# env = make_vec_env(env_id, n_envs=4)
env = make_vec_env("panda-reach-v0", n_envs=50)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)