# This agent uses the forex_env_v2 that uses continuous and binary controls
from __future__ import print_function
from copy import deepcopy
import gym
import gym.wrappers
import gym_forex
import json
import matplotlib.pyplot as plt
import multiprocessing
import neat
from neat.six_util import iteritems
from neat.six_util import itervalues
import numpy as np
import os
import pickle
import random
import sys
import time
import visualize
from gym.envs.registration import register
from population_syn import PopulationSyn # extended neat population for synchronizing witn singularity p2p network
# Multi-core machine support
NUM_CORES = 1

# class for evaluating the genomes
class GenomeEvaluator(object):
    genomes_h=[]
    def __init__(self, ts_f, vs_f):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0
        self.min_reward = -15
        self.max_reward = 15
        self.episode_score = []
        self.episode_length = []
        # register the gym-forex openai gym environment
        register(
            id='ForexTrainingSet-v1',
            entry_point='gym_forex.envs:ForexEnv3',
            kwargs={'dataset': ts_f, 'volume':0.2, 'sl':500, 'tp':500,'obsticks':2, 'capital':10000, 'leverage':100}
        )
        register(
            id='ForexValidationSet-v1',
            entry_point='gym_forex.envs:ForexEnv3',
            kwargs={'dataset': vs_f,'volume':0.2, 'sl':500, 'tp':500,'obsticks':2, 'capital':10000, 'leverage':100}
        )
        # make openai gym environments
        self.env_t = gym.make('ForexTrainingSet-v1')
        self.env_v = gym.make('ForexValidationSet-v1')
        # Shows the action and observation space from the forex_env, its observation space is
        # bidimentional, so it has to be converted to an array with nn_format() for direct ANN feed. (Not if evaluating with external DQN)
        print("action space: {0!r}".format(self.env_t.action_space))
        print("observation space: {0!r}".format(self.env_t.observation_space))
        #self.env_t = gym.wrappers.Monitor(env_t, 'results', force=True)
    
    # converts a bidimentional matrix to an one-dimention array
    def nn_format(self, obs):
        output = []
        for arr in obs:
            for val in arr:
                output.append(val)
        return output    
    
    # simulates a genom in all the training dataset (all the training subsets)
    def simulate(self, nets):
        # convert nets to D   
        scores = []
        sub_scores=[]
        self.test_episodes = []
        # Evalua cada net en todos los env_t excepto el env actual 
        for genome, net in nets:
            sub_scores=[]
            observation = self.env_t.reset()
            score=0.0
            #if i==index_t:
            while 1:
                output = net.activate(self.nn_format(observation))
                action = np.argmax(output)# buy, sell or nop
                observation, reward, done, info = self.env_t.step(action)
                score += reward
                #env_t.render()
                if done:
                    break
            sub_scores.append(score)
            # calculate fitness per genome
            scores.append(sum(sub_scores) / len(sub_scores))
        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))
        return scores
    
    def evaluate_genomes(self, genomes, config):
        self.generation += 1
        t0 = time.time()
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))
        t0 = time.time()
        scores = self.simulate(nets)
        t0 = time.time()
        print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        i = 0
        self.genomes_h=[]
        for genome, net in nets:
            genome.fitness = scores[i]
            self.genomes_h.append(genome)
            i = i + 1
            
    def training_validation_score(self,gen_best,config):
        # calculate training and validation fitness
        best_scores = []
        observation = self.env_t.reset()
        score = 0.0
        step = 0
        gen_best_nn = neat.nn.FeedForwardNetwork.create(gen_best, config)
        # calculate the training set score
        while 1:
            step += 1
            output = gen_best_nn.activate(self.nn_format(observation))

            action = np.argmax(output)# buy,sell or 
            observation, reward, done, info = self.env_t.step(action)
            score += reward
            self.env_t.render()
            if done:
                break
        self.episode_score.append(score)
        self.episode_length.append(step)
        best_scores.append(score)
        avg_score = sum(best_scores) / len(best_scores)
        print("Training Set Score =", score, " avg_score=", avg_score)
        # calculate the validation set score
        best_scores = []
        observation = self.env_v.reset()
        score = 0.0
        step = 0
        gen_best_nn = neat.nn.FeedForwardNetwork.create(gen_best, config)
        while 1:
            step += 1
            output = gen_best_nn.activate(self.nn_format(observation))
            action = np.argmax(output)# buy,sell or 
            observation, reward, done, info = self.env_v.step(action)
            score += reward
            #env_v.render()
            if done:
                break
        best_scores.append(score)
        avg_score_v = sum(best_scores) / len(best_scores)
        print("Validation Set Score = ", avg_score_v)
        print("*********************************************************")
        return avg_score