# -*- coding: utf-8 
# 
#  For importing new environment in ubuntu run, export PYTHONPATH=${PYTHONPATH}:/home/[your username]/gym-forex/
import random
import gym
import gym_forex
import numpy as np
from collections import deque
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from gym.envs.registration import register
import sys
#from pastalog import Log

# Allows to run multiple simultaneous GPU precesses
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
K.set_session(sess)

EPISODES = 5000
NUMVECTORS = 19
VECTORSIZE = 48
REPLAYFACTOR = 1
BATCHSIZE = 2
MEMORYSIZE= 15000 #porque hay 1400 ticks y quiero recordar last 10, 
# TODO: usar prioritized replay?

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORYSIZE)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.num_vectors=NUMVECTORS # number of features
        self.vector_size=VECTORSIZE # number of ticks
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # for observation[19][48], 19 vectors of 128-dimensional vectors,input_shape = (19, 48)
        # first set of CONV => RELU => POOL
        model.add(Conv1D(512, 5, input_shape=(self.num_vectors,self.vector_size)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        # second set of CONV => RELU => POOL
        model.add(Conv1D(32, 5))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        # set of FC => RELU layers
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64)) # valor óptimo:64 @400k
        model.add(Activation('relu'))

        # Softmax activation since we neet to chose only one of athe available actions
        model.add(Dense(self.action_size))
        model.add(Activation('softmax'))
        # multi-GPU support
        #model = to_multi_gpu(model)
        # use SGD optimizer
        opt = SGD(lr=self.learning_rate)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
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
        return np.argmax(act_values[0][0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                #print("action=",action)
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
# First argument is the training dataset
    ts_f = sys.argv[1]
    # TODO ADICIONAR VALIDATION SET?
    register(
            id = 'ForexTrainingSet-v1',
            entry_point = 'gym_forex.envs:ForexEnv3',
            kwargs = {
            'dataset': ts_f, 'volume':0.2, 'sl':500, 'tp':500, 
            'obsticks':VECTORSIZE, 'capital':10000, 'leverage':100
        }
    )
    # Make environments
    env = gym.make('ForexTrainingSet-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    print("state_size = ", state_size,", action_space = ", action_size, 
        ", replay_factor = ", REPLAYFACTOR, ", batch_size=", BATCHSIZE)
    done = False
    batch_size = BATCHSIZE # originalmente 32 (con 128 max 700k)
    best_performance = -1000000.0
    last_best_episode = 0 
    # muestra si hay soporte de GPU
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())
    #log_a = Log('http://localhost:8120', '1h4yvs48DCN_536_m128kbs128lr00001ed9tp1ksl2k') #OJO, capital inicial=300
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [agent.num_vectors,state_size])
        state = np.expand_dims(state, axis=0)
        time=0
        points=0.0
        done = False
        progress = 0.0
        #print("Starting Episode = ",e, " Replaying", flush=True)
        while not done:
            # env.render()
            #load data in the observation buffer(action=0 for the first 1440 observations)
            if time>state_size:
                action = agent.act(state)
            else:
                action=0
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else 0
            next_state = np.reshape(next_state, [agent.num_vectors,state_size])
            next_state = np.expand_dims(next_state, axis=0)
            if time>state_size:
                if (action>2):
                    print("Action Error = ", action)
                agent.remember(state, action, reward, next_state, done)
                points += reward
            state = next_state
            time=time+1
            #print("e:{}/{},t:{},p:{},e:{:.2}-".format(e, EPISODES, time, points,agent.epsilon))
            if done:
                agent.update_target_model()
                #print("Done: Episodes{}/{} Balance={:.2}, reward: {:.7}, points: {} epsilon:{:.2}  ,".format(e, EPISODES, points,agent.epsilon))
                print("Done:Ep{}/{} Bal={}, r:{} , best:{}, last:{}".format(e, EPISODES, info["balance"],points, best_performance ,last_best_episode))
                #logs the reward
                #log_a.post('Reward', value=points, step=e)
                # logs the reward
                #log_a.post('Balance', value=balance, step=e)
                # logs the tick Count
                #log_a.post('TickCount', value=tick_count,step=e)
                break
            if (len(agent.memory) > batch_size) and (time > state_size) and (time%REPLAYFACTOR==0) and (not done):
                agent.replay(batch_size)
                progress = info["tick_count"]*100/1450
                sys.stdout.write("Episode: %d Progress: %d%%   \r" % (e, progress) )
                sys.stdout.flush()
                #print(".", end="",flush=True)
        #TODO: Adicionar validation set score cada vez que se encuentre un óptimo
        #TODO: Detener por no avanzar en ultimos n episodes 
        #TODO: Detener por tiempo además de max episodes
        if best_performance < points:
            best_performance = points
            last_best_episode = e
            print("***********************************")
            print("New Best Performer: Ep{}/{} Balance={}, reward: {}".format(e, EPISODES, info["balance"],points))
            print("***********************************")
            
            agent.save("forexv3-ddqn.h5")
#TODO: DESPUES DE QUE ESTÉ FUNCIONANDO CONVERTIR EN FUNCIÓN QUE RETORNA MEJOR y su performance.
        