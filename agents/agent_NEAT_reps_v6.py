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
# First argument is the training dataset
ts_f = sys.argv[1]
# Second is validation dataset 
vs_f = sys.argv[2]
# Third argument is the  url 
my_url = sys.argv[3]
# fourth is the config filename
my_config = sys.argv[4]
# Register the gym-forex environment
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
# Make environments
env_t = gym.make('ForexTrainingSet-v1')
env_v = gym.make('ForexValidationSet-v1')
# Shows the action and observation space from the forex_env, its observation space is
# bidimentional, so it has to be converted to an array with nn_format() for direct ANN feed. (Not if evaluating with external DQN)
print("action space: {0!r}".format(env_t.action_space))
print("observation space: {0!r}".format(env_t.observation_space))
env_v = gym.wrappers.Monitor(env_v, 'results', force=True)
# for cross-validation like training set
index_t = 0

# LanderGenome class
class LanderGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff
    
    def __str__(self):
        return "Reward discount: {0}\n{1}".format(self.discount,
                                                  super().__str__())

# converts a bidimentional matrix to an one-dimention array
def nn_format(obs):
    output = []
    for arr in obs:
        for val in arr:
            output.append(val)
    return output

# class for training the agent
class PooledErrorCompute(object):
    def __init__(self):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0
        self.min_reward = -15
        self.max_reward = 15
        self.episode_score = []
        self.episode_length = []
    
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
                output = net.activate(nn_format(observation))
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

def run():
    # load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, my_config)
    config = neat.Config(LanderGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # uses the extended NEAT population PopulationSyn that synchronizes with singularity
    pop = PopulationSyn(config)
    # add reporters
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # save a checkpoint every 100 generations or 900 seconds.
    rep = neat.Checkpointer(100, 900)
    pop.add_reporter(rep)
    # class for trainign the agent
    ec = PooledErrorCompute()
    # initializes genomes fitness and gen_best just for the first time
    for g in itervalues(pop.population):
        g.fitness = -10000000.0
    gen_best = g
    # initializations
    avg_score_v = -10000000.0
    avg_score_v_ant = avg_score_v
    avg_score = avg_score_v
    iteration_counter = 0
    best_fitness=-2000.0;
    pop_size=len(pop.population)
    # sets the nuber of continuous iterations 
    num_iterations = round(200/len(pop.population))+1
    # repeat NEAT iterations until solved or keyboard interrupt
    while 1:
        try:
            # if it is not the  first iteration calculate training and validation scores
            if iteration_counter >0:
                avg_score=TrainingValidationScore(gen_best)
            # if it is not the first iteration
            if iteration_counter >= 0:
                # synchronizes with singularity migrating maximum 3 specimens
                pop.synSingularity(3, my_url, stats, gen_best)
                # perform pending evaluations on the singularity network, max 2
                evaluatePending(2)
                #increment iteration counter
                iteration_counter = iteration_counter + 1
            # execute num_iterations consecutive iterations of the NEAT algorithm
            gen_best = pop.run(ec.evaluate_genomes, num_iterations)
            # verify the training score is enough to stop the NEAT algorithm: TODO change to validation score when generalization is ok 
            if avg_score < 2000000000:
                solved = False
            if solved:
                print("Solved.")
                # save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)
                break
        except KeyboardInterrupt:
            print("User break.")
            break
    env.close()

if __name__ == '__main__':
    run()