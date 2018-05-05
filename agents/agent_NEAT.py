# Modified version of the Lander example included with neat-python for forex_env
from __future__ import print_function
# Imports
import gym
import gym.wrappers
import gym_forex
import json
import matplotlib.pyplot as plt
import multiprocessing
import neat
import numpy as np
import os
import pickle
import random
import time
import requests
import visualize
import sys
from copy import deepcopy
from neat.six_util import iteritems, itervalues
# Multi-core machine support
NUM_CORES = 1
# Make with the Name of the environment defined in gym_forex/__init__.py
env = gym.make('Forex-v0')
# Shows the action and observation space from the forex_env, its observation space is
# bidimentional, so it has to be converted to an array with nn_format() for direct ANN feed. (Not if evaluating with external DQN)
print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))
env = gym.wrappers.Monitor(env, 'results', force=True)
# First argument is the dataset, second is the  url
my_url=sys.argv[2]

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
    output=[]
    for arr in obs:
        for val in arr:
            output.append(val)
    return output

def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for score, data in episodes:
        # Compute normalized discounted reward.
        dr = np.convolve(data[:,-1], discount_function)[m:]
        dr = 2 * (dr - min_reward) / (max_reward - min_reward) - 1.0
        dr = np.clip(dr, -1.0, 1.0)

        for row, dr in zip(data, dr):
            observation = row[:38]
            action = int(row[3])
            #print("observation: {0!r}".format(observation))
            #print("f_observation: {0!r}".format(nn_format(observation)))
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error


class PooledErrorCompute(object):
    def __init__(self):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -2000
        self.max_reward = 2000

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        scores = []
        for genome, net in nets:
            observation = env.reset()
            step = 0
            data = []
            while 1:
                step += 1
                if step < 100 and random.random() < 0.1:
                    action = env.action_space.sample()
                else:
                    output = net.activate(nn_format(observation))
                    #print("output: {0!r}".format(output))
                    action = np.argmax(output)
                #print("observation: {0!r}".format(self.nn_format(observation)))
                #print("action: {0!r}".format(action))
                observation, reward, done, info = env.step(action)
                data.append(np.hstack((nn_format(observation), action, reward)))
                if done:
                    break

            data = np.array(data)
            score = np.sum(data[:,-1])
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes.append((score, data))

        print("Score range [{:.3f}, {:.3f}]".format(min(scores), max(scores)))

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = []
        for gid, g in genomes:
            nets.append((g, neat.nn.FeedForwardNetwork.create(g, config)))

        print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison.
        if 1 == self.generation % 10:
            self.test_episodes = self.test_episodes[-300:]
            self.simulate(nets)
            print("simulation run time {0}".format(time.time() - t0))
            t0 = time.time()

        # Selecciona aleatoriamente entre los test_episodes al pop.config.pop_size

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.

        print("Evaluating {0} test episodes".format(len(self.test_episodes)))
        if self.pool is None:
            for genome, net in nets:
                reward_error = compute_fitness(genome, net, self.test_episodes, self.min_reward, self.max_reward)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            jobs = []
            for genome, net in nets:
                jobs.append(self.pool.apply_async(compute_fitness,
                    (genome, net, self.test_episodes, self.min_reward, self.max_reward)))

            for job, (genome_id, genome) in zip(jobs, genomes):
                reward_error = job.get(timeout=None)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)

        print("final fitness compute time {0}\n".format(time.time() - t0))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(LanderGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop2 = neat.Population(config)
    stats2 = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop2.add_reporter(stats2)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    rep = neat.Checkpointer(25, 900)
    pop.add_reporter(rep)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute()
    temp = 0
    while 1:
        try:
            if temp > 0:
                # TODO: FUNCION DE SINCRONIZACION CON SINGULARITY
                # Lee en pop2 el último checkpoint desde syn
                # Hace request de getLastParam(process_hash,use_current) a syn TODO: HACER PROCESS CONFIGURABLE Y POR HASH no por id
                res = requests.get(
                    my_url+"/processes/1?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
                cont = res.json()
                print('\ncurrent_block_performance =', cont['result'][0]['current_block_performance'])
                print('\nlast_optimum_id =', cont['result'][0]['last_optimum_id'])
                last_optimum_id = cont['result'][0]['last_optimum_id']
                # Si el perf reportado pop2_champion_fitness > pop1_champion_fitness
                best_fitness = gen_best.fitness
                print('\nbest_fitness =', best_fitness)
                if cont['result'][0]['current_block_performance'] > best_fitness:
                    # hace request GetParameter(id)
                    res_p = requests.get(
                        my_url+"/parameters/" + str(
                            last_optimum_id) + "?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
                    cont_param = res_p.json()
                    # descarga el checkpoint del link de la respuesta si cont.parameter_link
                    print('\ncont_param =', cont_param)
                    if cont_param['result'][0]['parameter_link'] is not None:
                        genom_data = requests.get(cont_param['result'][0]['parameter_link']).content
                        with open('remote_genom', 'wb') as handler:
                            handler.write(genom_data)
                            handler.close()
                        # carga genom descargado en nueva población pop2
                        with open('remote_genom', 'rb') as f:
                            remote_genom = pickle.load(f)
                        # OP.MIGRATION: Reemplaza el peor de la especie pop1 más cercana por el nuevo chmpion de pop2 como http://neo.lcc.uma.es/Articles/WRH98.pdf
                        closer = None
                        min_dist = None
                        #for g in itervalues(pop.population):
                        #   dist = g.fitness
                        #   if closer is None or min_dist is None or dist < min_dist:
                        #       closer = g
                        #       min_dist = dist
                        # se selecciona el que tenga menos distancia al pop2.champion en los representantes de  pop1
                        closer = None
                        min_dist = None
                        # descarga el checkpoint del link de la respuesta si cont.parameter_link
                        for g in itervalues(pop.population):
                            dist = g.distance(remote_genom, config.genome_config)
                            if closer is None or min_dist is None:
                                closer = deepcopy(g)
                                min_dist = dist
                            if dist < min_dist:
                                closer = deepcopy(g)
                                min_dist = dist


                        # reemplazar el champ de pop2 en pop1
                        tmp_genom = deepcopy(remote_genom)
                        # Hack: overwrites original genome key with the replacing one
                        tmp_genom.key = closer.key
                        pop.population[closer.key] = deepcopy(tmp_genom)
                        # actualiza gen_best y best_genome al remoto
                        pop.best_genome=deepcopy(tmp_genom)
                        gen_best = deepcopy(tmp_genom)
                        #ejecuta speciate
                        pop.species.speciate(config, pop.population, pop.generation)
                        print("\ndone")
                # Si el perf reportado es menor pero no igual al de pop1
                if cont['result'][0]['current_block_performance'] < best_fitness:
                    # Guarda checkpoint del mejor genoma y lo copia a ubicación para servir vía syn.
                    # rep.save_checkpoint(config,pop,neat.DefaultSpeciesSet,rep.current_generation)
                    filename = '{0}{1}'.format("best-genome-", rep.current_generation)
                    with open(filename, 'wb') as f:
                        pickle.dump(gen_best, f)
                    # Hace request de CreateParam a syn
                    form_data = {"process_hash": "ph", "app_hash": "ah",
                                 "parameter_link": my_url+"/genoms/" + filename,
                                 "parameter_text": pop.best_genome.key, "parameter_blob": "", "validation_hash": "",
                                 "hash": "h", "performance": best_fitness, "redir": "1", "username": "harveybc",
                                 "pass_hash": "$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q"}
                    # TODO: COLOCAR DIRECCION CONFIGURABLE
                    res = requests.post(
                        my_url+"/parameters?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph",
                        data=form_data)
                    res_json = res.json()
                # TODO FIN: FUNCION DE SINCRONIZACION CON SINGULARITY
            temp = temp + 1
            gen_best = pop.run(ec.evaluate_genomes, 5)

            #print(gen_best)
            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")
            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig("scores.svg")
            plt.close()

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the best genomes seen so far as an ensemble-ish control system.
            best_genomes = stats.best_unique_genomes(3)
            best_networks = []
            #for g in best_genomes:
                #best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))
            ngb = neat.nn.FeedForwardNetwork.create(gen_best, config)
            solved = True
            best_scores = []
            for k in range(10):
                observation = env.reset()
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all five networks to
                    # determine the best action given the current state.
                    votes = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(nn_format(observation))
                        votes[np.argmax(output)] += 1
                    #output = ngb.activate(nn_format(observation))
                    #votes[np.argmax(output)] += 1
                    best_action = np.argmax(votes)
                    observation, reward, done, info = env.step(best_action)
                    score += reward
                    env.render()
                    if done:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print(k, score, avg_score)
                if avg_score < 20000:
                    solved = False
                    break


            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name+'.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name+"-net.gv")
                    visualize.draw_net(config, g, view=False, filename=name+"-net-enabled.gv",
                                       show_disabled=False)
                    visualize.draw_net(config, g, view=False, filename=name+"-net-enabled-pruned.gv",
                                       show_disabled=False, prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()


if __name__ == '__main__':
    run()