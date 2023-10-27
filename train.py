import gym
import gym_panda_reach
import numpy as np
from stable_baselines3 import DDPG
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_features = observation_space.shape[0]  # 获取输入特征的数量
        self.linear = nn.Sequential(nn.Linear(n_input_features, features_dim), nn.ReLU())  # 使用线性层处理数据

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(observations)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# 初始化环境
env = gym.make('me5412')

# 设置随机种子
np.random.seed(0)

# 初始化DDPG模型
model = DDPG('MlpPolicy', env,policy_kwargs=policy_kwargs, verbose=1)
# 在环境中训练模型
model.learn(total_timesteps=200000)
model.save("nnmodel")

# 评估模型
episode_rewards = []
for i in range(10):
    obs = env.reset()
    episode_reward = 0
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
    episode_rewards.append(episode_reward)

# 输出评估结果
print(f"Mean reward: {np.mean(episode_rewards)}")
