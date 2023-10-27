import gym
import gym_panda_reach
from stable_baselines3 import DDPG

#env = gym.make('panda-reach-v0')
env = gym.make('me5412')

env.reset()
env.reward_type = "sparse" #default is "dense"
#model = DDPG(policy="MultiInputPolicy", env=env, verbose =1)
#model.learn(30_000)

for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(
        env.action_space.sample()) # take a random action
env.close()