from gym.envs.registration import register

register(
    id='Forex-v0',
    entry_point='gym_forex.envs:ForexEnv',
)