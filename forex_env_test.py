import gym
env = gym.make('Forex-v0')
env.reset()
episode_over=bool(0)
points=0 # for cummulative reward
while not episode_over: # run until episode over
    env.render()
    action = env.action_space.sample() # pick a random action
    ob, reward, episode_over, info = env.step(action) # take random action
    points += reward
print ('Total Reward =',points)
