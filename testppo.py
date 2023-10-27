import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym_panda_reach 

PandaEnv = gym.make('panda-reach-v0')

# 创建 Panda 环境
env = DummyVecEnv([lambda: PandaEnv()])

# 定义 PPO 模型
model = PPO('MlpPolicy', env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_panda_model")

# 加载模型
model = PPO.load("ppo_panda_model")

# 观察效果
obs = env.reset()
for i in range(1000):  # 运行1000个步骤
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')

env.close()
