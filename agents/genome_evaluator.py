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
    def __init__(self):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0
        self.min_reward = -15
        self.max_reward = 15
        self.episode_score = []
        self.episode_length = []
    
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
            observation = env_t.reset()
            score=0.0
            #if i==index_t:
            while 1:
                output = net.activate(self.nn_format(observation))
                action = np.argmax(output)# buy, sell or nop
                observation, reward, done, info = env_t.step(action)
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
        for genome, net in nets:
            genome.fitness = scores[i]
            i = i + 1
