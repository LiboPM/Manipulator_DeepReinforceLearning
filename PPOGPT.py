import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym_panda_reach

# 创建 PandaEnv 环境
env = gym.make('me5412')

# 使用 make_vec_env 将单个环境转换为 VecNormalize 对象
env = make_vec_env(lambda: env, n_envs=1)

# 初始化 PPO 模型
model = PPO('MlpPolicy', env, verbose=1)

# 在环境中训练模型
model.learn(total_timesteps=20000)

# 保存训练后的模型
model.save("ppo_panda_model")

# 加载模型并评估
model = PPO.load("ppo_panda_model")

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
