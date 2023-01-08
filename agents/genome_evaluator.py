# library for ann genome evaluation
from __future__ import print_function
import gym
import gym.wrappers
import multiprocessing
import neat
import numpy as np
import time
from gym.envs.registration import register
import requests

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
            entry_point='gym_forex.envs:ForexEnv4',
            kwargs={'dataset': ts_f, 'volume':0.2, 'sl':500, 'tp':500,'obsticks':2, 'capital':10000, 'leverage':100}
        )
        register(
            id='ForexValidationSet-v1',
            entry_point='gym_forex.envs:ForexEnv4',
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

    # log the iteration data in data_logger
    def data_log(self, validation_score, avg_score_v, training_score, avg_score, info):
        #TODO: replace config_id (number at thñe end of url) if required, same with username and pass
        url = 'http://127.0.0.1:60500/gym-fx/0'
        data = {'validation_score': validation_score, 'avg_score_v': avg_score_v, 'training_score': training_score, 'avg_score': avg_score, \
            "action":info["action"],"balance":info["balance"], "tick_count":info["tick_count"], "num_closes":info["num_closes"], \
            "equity":info["equity"], "reward":info["reward"], "order_status":info["order_status"], "margin":info["margin"], \
            "initial_capital":info["initial_capital"]}
        try:
            response = requests.post(url, json=data, timeout=3, auth=('test', 'pass')) 
            
        except requests.exceptions.Timeout:
            print("Warning: data-logger requaest timeout (t>3s)")
        except Exception as e:
            print("Warning: unable to connect to data-logger : " + str(e))
        else:
            print("Info: Data logged successfully")
      
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
        print("Training Set Score=", score, " avg_score=", avg_score, " num_closes= ", info["num_closes"], 
            " balance=", info["balance"])

        # calculate the validation set score
        best_scores = []
        observation = self.env_v.reset()
        v_score = 0.0
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
        best_scores.append(v_score)
        avg_score_v = sum(best_scores) / len(best_scores)
        print("Validation Set Score = ", v_score, " avg_score=", avg_score_v, " num_closes= ", info["num_closes"], 
            " balance=", info["balance"])

        self.data_log(validation_score=v_score, avg_score_v=avg_score_v, training_score=score, avg_score=avg_score, info=info)

        print("*********************************************************")
        return avg_score_v