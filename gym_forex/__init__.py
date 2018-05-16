from gym.envs.registration import register

register(
    id='ForexTrainingSet1-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts1_15min_3m.CSV'}
)

register(
    id='ForexTrainingSet2-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts2_15min_3m.CSV'}
)

register(
    id='ForexTrainingSet3-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts3_15min_3m.CSV'}
)

register(
    id='ForexTrainingSet4-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts4_15min_3m.CSV'}
)

register(
    id='ForexTrainingSet5-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts5_15min_3m.CSV'}
)

register(
    id='ForexTrainingSet6-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts6_15min_3m.CSV'}
)

register(
    id='ForexTrainingSet7-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts7_15min_3m.CSV'}
)
register(
    id='ForexTrainingSet8-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts8_15min_3m.CSV'}
)
register(
    id='ForexTrainingSet9-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts9_15min_3m.CSV'}
)
register(
    id='ForexTrainingSet10-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts10_15min_3m.CSV'}
)
register(
    id='ForexTrainingSet11-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts11_15min_3m.CSV'}
)
register(
    id='ForexTrainingSet12-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/ts12_15min_3m.CSV'}
)

register(
    id='ForexValidationSet-v0',
    entry_point='gym_forex.envs:ForexEnv',
    kwargs={'dataset': 'datasets/vs_15min_3m.CSV'}
)