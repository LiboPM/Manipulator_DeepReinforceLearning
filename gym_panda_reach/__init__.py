from gym.envs.registration import register

register(
    #id="panda-reach-v0",
    id="me5412",
    entry_point="gym_panda_reach.envs:PandaEnv",
)
