import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
import gym_panda_reach

env = gym.make('me5412')
model = DDPG(policy="MultiInputPolicy", env=env,device="cpu",verbose = 1)
model.learn(30_000)