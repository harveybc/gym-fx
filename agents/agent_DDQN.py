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
from keras.callbacks import TensorBoard, LearningRateScheduler, ReduceLROnPlateau
#from pastalog import Log

# Allows to run multiple simultaneous GPU precesses
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
K.set_session(sess)

EPISODES = 1400     # number of episodes per evaluation
NUMVECTORS = 19     # number of features
VECTORSIZE = 48     # window size (ticks per feature)
REPLAYFACTOR = 20   # number of ticks to skip between replays
BATCHSIZE = 1       # number of samples per replay
MEMORYSIZE= 128000  # porque hay 1400 ticks y quiero recordar last 50
REMEMBERTHRESHOLD=1 # frames to skip from remember if no action or change of balance is made
STOPLOSS = 50000    # stop loss for all orders
TAKEPROFIT = 50000  # take profit for all orders
CAPITAL = 10000     # starting capital
REPMAXPROFIT = 1    # number of times an action/state is recorded for replay
MOVINGAVERAGE = 20  # number of past ticks to use as performance score average
GAMMA = 0.95        # discount rate used in replay
EPSILON = 1.0       # initial exploration rate (does random action until minimum)
EPSILON_MIN = 0.005 # minimum exploration rate
EPSILON_DECAY = 0.93   # exploration rate decay factor  
LEARNING_RATE = 0.0001 # learning rate for the selected optimizer (sgd with momentum)

# TODO: usar prioritized replay?

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORYSIZE)
        self.points_log = deque(maxlen=MOVINGAVERAGE)
        self.gamma = GAMMA      # discount rate used in replay
        self.epsilon = EPSILON  # exploration rate
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.num_vectors = NUMVECTORS # number of features
        self.vector_size = VECTORSIZE # number of ticks
        
        self.model = self._build_model()
        self.model_max = self.model
        self.target_model = self._build_model()
        self.update_target_model()


    # convert an ann to a dcn model
    def ann2dcn(self, nets_ann, num_vectors, vector_size):
        # esta funci?n debe retornar un arreglo de modelos 
        # Deep Conv Neural Net for Deep-Q learning Model
        models = []
        # for each net generate a dnn model
        for net in nets_ann:
            # creates a new model
            model = Sequential()
            # node counter
            c_node = 0
            # initialize values from input node
            node, act_func, agg_func, bias, response, links = net.node_evals[net.input_nodes[0]]
            # repeat until next_node != output (add layers)
            while true:
                # add the layer depending on the conection to the next node:
                # act_funct of the next node    = core layer
                # bias of the next neuron (0.1,1) = kernel size
                kernel_size = round(bias * max_kernel_size)
                # response of the next node(0.1,1)= pool_size, stride
                pool_size = round(response * max_pool_size)
                # link.w to the next node(0,1)  = number of filters
                filters = links[0].w
                ##########################################################
                # Encoding:
                # agg_funct = min -> adds dropout, else adds conv1D
                # agg_funct = sum -> add pooling layer
                # agg_funct = product -> does nothing(add it as option in NEAT config)
                # act:funct = relu -> adds relu layer
                # act_funct = sigmoid -> adds hard-sigmoid layer
                ##########################################################
                # si agg_funct = min: adiciona capa dropout # TODO: add min aggregation_options in config_20_dqn
                if agg_funct == 'min':
                    # if its the first node:
                    if c_node==0:
                        model.add(Dropout(0.1, input_shape=(num_vectors, vector_size)))
                    else:
                        model.add(Dropout(0.1))
                # sino es dropout adiciona una capa Conv1D (note in config that sum and product are set so a conv1d layer has double the chances than dropout (min agg_funct))
                else:
                    if c_node==0:
                        model.add(Conv1D(filters, kernel_size, input_shape=(num_vectors, vector_size)))
                    else:
                        model.add(Conv1D(filters, kernel_size))
                # act_funct = sigmoid: relu 
                if act_funct == 'relu': 
                        model.add(Activation('relu'))
                # act_funct = tanh: hard_sigmoid
                if act_funct == 'sigmoid': 
                    model.add(Activation('hard_sigmoid'))
                # agg_funct = sum:pooling, product:no-pooling
                if agg_funct == 'sum':
                    model.add(MaxPooling1D(pool_size=pool_size, strides=pool_size))
                # TODO: DROPOUT SOLO SE DEBE USAR EN TRAINING, NO EN EVAL
                # stop condition for while: until next node = output
                if links[0].i==net.output_nodes[0]:
                    break
                # read values from next node
                node, act_func, agg_func, bias, response, links = net.node_evals[links[0].i]    
                # increment node counter
                c_node += 1
            # adds a dense layer with the parameters of the output node with response attribute
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(response*max_dense)) # valor ?ptimo:64 @400k
            model.add(Activation('relu'))
            # TODO: Probar con Hard sigmoid(pq controls requieren  -1,1) y relu(0,1)-> mod. controls para prueba
            model.add(Dense(self.action_size))
            model.add(Activation('hard_sigmoid'))
            # multi-GPU support
            #model = to_multi_gpu(model)
            # use SGD optimizer
            opt = SGD(lr=self.learning_rate)
            model.compile(loss="mean_squared_error", optimizer=opt,
                          metrics=["accuracy"])
            # append model to models
            models.append(model)
        return models


    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # for observation[19][48], 19 vectors of 128-dimensional vectors,input_shape = (19, 48)
        # first set of CONV => RELU => POOL
        model.add(Conv1D(512, 5,input_shape=(self.num_vectors,self.vector_size)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        # second set of CONV => RELU => POOL
        model.add(Conv1D(32, 5))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        # second set of CONV => RELU => POOL
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64)) # valor óptimo:64 @400k
        model.add(Activation('relu'))
        # output layer
        model.add(Dense(self.action_size))
        model.add(Activation('softmax'))
        # multi-GPU support
        #model = to_multi_gpu(model)
        #self.reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=1e-4)
        # use SGD optimizer
        #opt = Adam(lr=self.learning_rate)
        opt = SGD(lr=self.learning_rate, momentum=0.9)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        #model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
        return model
 
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def average_points(self):
        accum=0.0
        if len(self.points_log) == 0:
            return -100;
        for p in self.points_log:
            accum=accum + p
        return (accum/len(self.points_log))

    # copies the weights from the  current model to the max model
    def update_model_max(self):
        # copy weights from model to target_model
        self.model_max.set_weights(self.model.get_weights())

    # copies the weights from the  model with max perf to the current model
    def restore_max(self):
        # copy weights from model_max to target_model
        self.model.set_weights(self.model_max.get_weights())


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
            #self.model.fit(state, target, epochs=1, verbose=0, callbacks=[self.reduce_lr])
            
            # TODO: DELAYED REWARD
            # if action opens an order save the observation in tmpvar DONT DO FIT
            # if action closes an order, half reward to open and close obs, DO FIT
            
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

#REPLAYFACTOR = 20      # number of ticks to skip between replays
#BATCHSIZE = 1          # number of samples per replay
#MEMORYSIZE= 128000     # porque hay 1400 ticks y quiero recordar last 50
#REMEMBERTHRESHOLD = 1  # frames to skip from remember if no action or change of balance is made
#STOPLOSS = 50000
#TAKEPROFIT = 50000
#CAPITAL = 10000        # starting capital
#REPMAXPROFIT = 1    # number of times an action/state is recorded for replay
#MOVINGAVERAGE = 20  # number of past ticks to use as performance score average
#GAMMA = 0.95        # discount rate used in replay
#EPSILON = 1.0       # initial exploration rate (does random action until minimum)
#EPSILON_MIN = 0.005 # minimum exploration rate
#EPSILON_DECAY = 0.93   # exploration rate decay factor  
#LEARNING_RATE = 0.0001 # learning rate for the selected optimizer (sgd with momentum)

# Parameters to be obtained from the genomes as input 
#EPISODES = 1400
#VECTORSIZE = 48        # window size (ticks per feature)
#NUMVECTORS = 19        # number of features 

    #TODO: add parameters
    # TODO: define how the global dqn params are encoded (not layers/connections)
    def evaluate(self, dcn_model, ts_f):
        ts_f = sys.argv[1]
        # TODO ADICIONAR VALIDATION SET?
        register(
                id = 'ForexTrainingSet-v1',
                entry_point = 'gym_forex.envs:ForexEnv4',
                kwargs = {
                'dataset': ts_f, 'volume':0.2, 'sl':STOPLOSS, 'tp':TAKEPROFIT, 
                'obsticks':VECTORSIZE, 'capital':CAPITAL, 'leverage':100
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
        best_performance = -100.0
        last_best_episode = 0 
        # muestra si hay soporte de GPU
        #from tensorflow.python.client import device_lib
        #print(device_lib.list_local_devices())
        #log_a = Log('http://localhost:8120', '1h4yvs48DCN_536_m128kbs128lr00001ed9tp1ksl2k') #OJO, capital inicial=300
        max_variation = 0.0
        num_repetitions = 0
        points_max = -100.0
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [agent.num_vectors,state_size])
            state = np.expand_dims(state, axis=0)
            time=0
            points=0.0
            done = False
            progress = 0.0
            balance_ant=CAPITAL
            #print("Starting Episode = ",e, " Replaying", flush=True)
            while not done:
                #load data in the observation buffer(action=0 for the first 1440 observations)
                if time>state_size:
                    action = agent.act(state)
                else:
                    action=0
                next_state, reward, done, info = env.step(action)            
                next_state = np.reshape(next_state, [agent.num_vectors,state_size])
                next_state = np.expand_dims(next_state, axis=0)
                if time>state_size:
                    # if action  = 0 have a REMEMBERTHRESOLD prob of remembering
                    if (action>0):
                        # update max_profit 
                        variation = abs(info["balance"]-balance_ant)
                        if variation > max_variation:
                            max_variation = variation
                        # remember additional times if profit was large
                        if max_variation > 0.0:
                            num_repetitions = 1+round((variation/max_variation)* REPMAXPROFIT)
                        else: 
                            num_repetitions = int(1)
                        for repetition in range(int(num_repetitions)):
                            # remember action/state for replay
                            agent.remember(state, action, reward, next_state, done)
                    # also save if balance varies, eg. if TP or SL
                    elif (balance_ant - info["balance"])!=0.0:
                        agent.remember(state, action, reward, next_state, done)
                    else:
                        if e % REMEMBERTHRESHOLD == 0:
                            agent.remember(state, action, reward, next_state, done)
                    points += reward
                state = next_state
                time=time+1
                balance_ant = info["balance"]
                #print("e:{}/{},t:{},p:{},e:{:.2}-".format(e, EPISODES, time, points,agent.epsilon))
                if done:
                    agent.update_target_model()
                    agent.points_log.append(points)
                    avg_points = agent.average_points()
                    print("Done:Ep{}/{} Bal={}, points:{}, best:{}, last:{}, average:{}".format(e, EPISODES, info["balance"],points, best_performance ,last_best_episode, avg_points))
                    # if performance decreased, loads the last model
                    #if (points>points_max):
                    #    print("max updated")
                    #    agent.update_model_max()                                     
                    #else:
                    #    print("max restored")
                    #    agent.restore_max()
                    #points_max = points   
                    break

                if (len(agent.memory) > batch_size) and (time > state_size) and ((time)%REPLAYFACTOR==0) and (not done):
                    agent.replay(batch_size+e)
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
        