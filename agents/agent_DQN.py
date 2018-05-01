# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from pastalog import Log

EPISODES = 2000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=128000) #originalmente 2k
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exporlation rate, orig=1.0
        self.epsilon_min = 0.001 # originalmente 0.01 (max 0.001 6m142@4y)
        self.epsilon_decay = 0.7 # originalmente 0.99 (max con 0.7 en 4y)
        self.learning_rate = 0.0001 #originalmente 0.001 (max 0.0001 6m142@4y)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # Best so-far 4 dense: 64-4(3output) 135k@550
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('Forex-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 128 # originalmente 32 con max 148k@545
    # muestra si hay soporte de GPU
    #from tensorflow.python.client import device_lib

    #print(device_lib.list_local_devices())
    #start loging with pastalog
    log_a = Log('http://localhost:8120', 'DQN_5123264_lr0001m128kbs128ed07_eq')
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [19, state_size])
        time=0
        points=0.0
        done = False
        while not done:
            # env.render()
            #load data in the observation buffer(action=0 for the first 1440 observations)
            if time>state_size:
                action = agent.act(state)
            else:
                action=0
                #TODO  :  REWARD COMO FUNCION DE NUM ORDENES AL DIA (4? Gauss?)
            next_state, reward, done, balance, tick_count, _ = env.step(action)
            reward = reward if not done else 0
            next_state = np.reshape(next_state, [19, state_size])
            if time>state_size:
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            time=time+1
            points += reward
            #print("e:{}/{},t:{},p:{},e:{:.2}-".format(e, EPISODES, time, points,agent.epsilon))
            if done:
                agent.update_target_model()
                print("episode Done: {}/{} ,reward: {} e: {:.2},".format(e, EPISODES, points,agent.epsilon))
                #logs the reward
                log_a.post('Reward', value=points, step=e)
                # logs the reward
                log_a.post('Balance', value=balance, step=e)
                # logs the tick Count
                log_a.post('TickCount', value=tick_count,step=e)
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
