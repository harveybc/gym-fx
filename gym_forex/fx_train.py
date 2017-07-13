import gym
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Forex-v0',
    entry_point='envs:ForexEnv',
    timestep_limit=1000000,
    reward_threshold=1.0,
    nondeterministic = True,
)

env = gym.make('Forex-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000000):
        #env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break