from gym.envs.registration import register
import sys

register(
    id='Forex-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': sys.argv[2]}
)