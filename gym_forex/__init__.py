from gym.envs.registration import register

register(
    id='ForexTrainingSet-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts_15min_3m.CSV'}
)

register(
    id='ForexValidationSet-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/vs_15min_3m.CSV'}
)