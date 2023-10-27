import gym
import gym_panda_reach
import torch as th
import torch.nn as nn
import os
import panda_gym
from gymnasium import spaces

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def create_env():
    env_id = 'me5412'
    return gym.make(env_id)

env_id = 'me5412'
env=gym.make(env_id)
# # Get the state space and action space
# s_size = env.observation_space.shape
# a_size = env.action_space

# print("_____OBSERVATION SPACE_____ \n")
# print("The State Space is: ", s_size)
# print("Sample observation", env.observation_space.sample()) # Get a random observation

#env = make_vec_env(env_id, n_envs=4)
#env = make_vec_env(create_env, n_envs=50)
#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(
#                 th.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))

# policy_kwargs = dict(
#     features_extractor_class=CustomCNN,
#     features_extractor_kwargs=dict(features_dim=128),
# )

model = A2C(policy = "MultiInputPolicy",
            env = env,
            #policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=1)

model.learn(50_000)
# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-PandaReachDense-v3") # our model_save_id
env.save("vec_normalize.pkl") # our env_save_id