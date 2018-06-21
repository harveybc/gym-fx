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
import requests
import sys
import time
import visualize
from gym.envs.registration import register
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
        # convert nets to DCN
        
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

        #print("network creation time {0}".format(time.time() - t0))
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison.
        #if 1 == self.generation % 10:
        #self.test_episodes = self.test_episodes[-300:]
        scores = self.simulate(nets)
        #print("simulation run time {0}".format(time.time() - t0))
        t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.

        print("Evaluating {0} test episodes".format(len(self.test_episodes)))

        i = 0
        for genome, net in nets:
            genome.fitness = scores[i]
            i = i + 1
        #print("final fitness compute time {0}\n".format(time.time() - t0))


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, my_config)
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
    # Training set index 
    avg_score_v = -10000000.0
    avg_score_v_ant = avg_score_v
    avg_score = avg_score_v

    # asigna un gen_best para poder cargar los demás desde syn
    for g in itervalues(pop.population):
        gen_best = g
        g.fitness = -10000000.0

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute()
    temp = 0
    best_fitness=-2000.0;

    pop_size=len(pop.population)
    # sets the nuber of continuous iterations 
    num_iterations = round(200/len(pop.population))+1
    while 1:
        try:
            if temp >0:
                # Calcula training y validation fitness
                solved = True
                best_scores = []
                observation = env_t.reset()
                score = 0.0
                step = 0
                gen_best_nn = neat.nn.FeedForwardNetwork.create(gen_best, config)

                while 1:
                    step += 1
                    output = gen_best_nn.activate(nn_format(observation))

                    action = np.argmax(output)# buy,sell or 
                    observation, reward, done, info = env_t.step(action)
                    score += reward
                    env_t.render()
                    if done:
                        break
                ec.episode_score.append(score)
                ec.episode_length.append(step)
                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                print("Training Set Score =", score, " avg_score=", avg_score)

                # Calculate the real-validation set score
                solved = True
                best_scores = []
                observation = env_v.reset()
                score = 0.0
                step = 0
                gen_best_nn = neat.nn.FeedForwardNetwork.create(gen_best, config)
                while 1:
                    step += 1
                    output = gen_best_nn.activate(nn_format(observation))
                    action = np.argmax(output)# buy,sell or 
                    observation, reward, done, info = env_v.step(action)
                    score += reward
                    #env_v.render()
                    if done:
                        break
                best_scores.append(score)
                avg_score_v = sum(best_scores) / len(best_scores)
                print("Validation Set Score = ", avg_score_v)
                print("*********************************************************")
                # Calcula el best_fitness (PARA SYNC)como el promedio del score de training y el promedio del fitness de los reps. 
                best_genomes = stats.best_unique_genomes(3)
                reps_local = []
                reps = [gen_best]
                accum = 0.0
                countr = 0
                for g in best_genomes:
                    if g.fitness is not None:
                        accum = accum + g.fitness
                        countr = countr + 1
                if countr > 0:    
                    best_fitness = (3*avg_score+(accum/countr))/4
                else:
                    best_fitness = (avg_score)
                #FIN de calculo de real validation        
                
            if temp >= 0:
                # TODO: FUNCION DE SINCRONIZACION CON SINGULARITY
                # Lee en pop2 el último checkpoint desde syn
                # Hace request de getLastParam(process_hash,use_current) a syn TODO: HACER PROCESS CONFIGURABLE Y POR HASH no por id
                res = requests.get(
                                   my_url + "/processes/1?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
                cont = res.json()
                print('\ncurrent_block_performance =', cont['result'][0]['current_block_performance'])
                print('\nlast_optimum_id =', cont['result'][0]['last_optimum_id'])
                last_optimum_id = cont['result'][0]['last_optimum_id']

                
                # Si el perf reportado pop2_champion_fitness > pop1_champion_fitness de validation training        
                print("\nPerformance = ", best_fitness)
                print("*********************************************************")
                if cont['result'][0]['current_block_performance'] > best_fitness:
                    # hace request GetParameter(id)
                    res_p = requests.get(
                                         my_url + "/parameters/" + str(
                                         last_optimum_id) + "?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
                    cont_param = res_p.json()
                    # descarga el checkpoint del link de la respuesta si cont.parameter_link
                    print('Parameter Downloaded')
                    print('\nmigrations =')
                    if cont_param['result'][0]['parameter_link'] is not None:
                        genom_data = requests.get(cont_param['result'][0]['parameter_link']).content
                        with open('remote_reps', 'wb') as handler:
                            handler.write(genom_data)
                            handler.close()
                        # carga genom descargado en nueva población pop2
                        with open('remote_reps', 'rb') as f:
                            remote_reps = pickle.load(f)
                        # OP.MIGRATION: Reemplaza el peor de la especie pop1 más cercana por el nuevo chmpion de pop2 como http://neo.lcc.uma.es/Articles/WRH98.pdf
                        # para cada elemento de remote_reps, busca el closer, si remote fitness > local, lo reemplaza
                        for i in range(len(remote_reps)):
                            closer = None
                            min_dist = None
                            # initialize for less fit search
                            less_fit = None
                            less_fitness = 10000
                            for g in itervalues(pop.population):
                                if g not in remote_reps:
                                    dist = g.distance(remote_reps[i], config.genome_config)
                                    if dist is None:
                                       dist = 100000000 
                                else:
                                    dist = 100000000
                                # do not count already migrated remote_reps
                                if closer is None or min_dist is None:
                                    closer = deepcopy(g)
                                    min_dist = dist
                                if dist < min_dist:
                                    closer = deepcopy(g)
                                    min_dist = dist
                                if g.fitness is None:
                                    g.fitness = -10
                                if g.fitness < less_fitness:
                                    less_fitness = g.fitness
                                    less_fit = deepcopy(g)
                            # For the best genom in position 0
                            if i == 0 and remote_reps[0].fitness>gen_best.fitness:
                                if closer is None:
                                    # busca el pop con el menor fitness
                                    closer = less_fit
                                tmp_genom = deepcopy(remote_reps[i])
                            # Hack: overwrites original genome key with the replacing one
                                tmp_genom.key = closer.key
                                pop.population[closer.key] = deepcopy(tmp_genom)
                                print("gen_best=", closer.key)
                                pop.best_genome = deepcopy(tmp_genom)
                                #gen_best = deepcopy(tmp_genom)
                            else:
                                # si el remote fitness>local, reemplazar el remote de pop2 en pop1
                                if closer is None:
                                    # busca el pop con el menor fitness
                                    closer = less_fit
                                if closer is not None:
                                    if closer not in remote_reps:
                                        if closer.fitness is None:
                                            closer.fitness = less_fitness
                                        if closer.fitness is not None and remote_reps[i].fitness is not None:
                                            if remote_reps[i].fitness > closer.fitness:
                                                tmp_genom = deepcopy(remote_reps[i])
                                                # Hack: overwrites original genome key with the replacing one
                                                tmp_genom.key = closer.key
                                                pop.population[closer.key] = deepcopy(tmp_genom)
                                                print("Replaced=", closer.key)
                                                # actualiza gen_best y best_genome al remoto
                                                pop.best_genome = deepcopy(tmp_genom)
                                                #gen_best = deepcopy(tmp_genom)
                                        if closer.fitness is None:
                                            tmp_genom = deepcopy(remote_reps[i])
                                            # Hack: overwrites original genome key with the replacing one
                                            tmp_genom.key = len(pop.population) + 1
                                            pop.population[tmp_genom.key] = tmp_genom
                                            print("Created Por closer.fitness=NONE : ", tmp_genom.key)
                                            # actualiza gen_best y best_genome al remoto
                                            pop.best_genome = deepcopy(tmp_genom)
                                            #gen_best = deepcopy(tmp_genom)
                                    else:
                                        #si closer está en remote_reps es porque no hay ningun otro cercano así que lo adiciona
                                        tmp_genom = deepcopy(remote_reps[i])
                                        # Hack: overwrites original genome key with the replacing one
                                        tmp_genom.key = len(pop.population) + 1
                                        pop.population[tmp_genom.key] = tmp_genom
                                        print("Created por Closer in rempte_reps=", tmp_genom.key)
                                        # actualiza gen_best y best_genome al remoto
                                        pop.best_genome = deepcopy(tmp_genom)
                                        #gen_best = deepcopy(tmp_genom)

                        #ejecuta speciate
                        pop.species.speciate(config, pop.population, pop.generation)
                        print("\nSpeciation after migration done")
                # Si el perf reportado es menor pero no igual al de pop1
                if cont['result'][0]['current_block_performance'] < best_fitness:
                    # Obtiene remote_reps
                    # hace request GetParameter(id)
                    remote_reps = None
                    res_p = requests.get(
                                         my_url + "/parameters/" + str(
                                         last_optimum_id) + "?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
                    cont_param = res_p.json()
                    # descarga el checkpoint del link de la respuesta si cont.parameter_link
                    print('\nNEW OPTIMUM - cont_param =', cont_param)
                    #print('\nmigrations =')
                    if cont_param['result'][0]['parameter_link'] is not None:
                        genom_data = requests.get(cont_param['result'][0]['parameter_link']).content
                        with open('remote_reps', 'wb') as handler:
                            handler.write(genom_data)
                            handler.close()
                        # carga genom descargado en nueva población pop2
                        with open('remote_reps', 'rb') as f:
                            remote_reps = pickle.load(f)
                #Guarda los mejores reps
                    reps_local = []
                    reps = [gen_best]
                    # Para los mejores genes
                    best_genomes = stats.best_unique_genomes(3)
                    for g in best_genomes:
                        #print("\ns=",s)
                        if g not in reps_local:
                            reps_local.append(g)
                            reps_local[len(reps_local)-1] = deepcopy(g)
                    # TODO: Conservar los mejores reps, solo reemplazarlos por los mas cercanos
                    if remote_reps is None:
                        for l in reps_local:
                            reps.append(l)
                            reps[len(reps)-1] = deepcopy(l)
                    else:
                        # para cada reps_local l
                        for l in reps_local:
                            # busca el closer a l en reps_remote
                            for i in range(len(remote_reps)):
                                closer = None
                                min_dist = None
                                for g in reps_local:
                                    if g not in remote_reps:
                                        dist = g.distance(remote_reps[i], config.genome_config)
                                    else:
                                        dist = 100000000
                                    # do not count already migrated remote_reps
                                    if closer is None or min_dist is None:
                                        closer = deepcopy(g)
                                        min_dist = dist
                                    if dist < min_dist:
                                        closer = deepcopy(g)
                                        min_dist = dist
                #           si closer is in reps
                            if closer in reps:
                #               adiciona l a reps si ya no estaba en reps
                                if l not in reps:
                                    reps.append(l)
                                    reps[len(reps) - 1] = deepcopy(l)
                #           sino
                            else:
                #               si l tiene más fitness que closer,
                                if closer.fitness is not None and l.fitness is not None:
                                    if l.fitness>closer.fitness:
                #                       adiciona l a reps si ya no estaba en reps
                                        if l not in reps:
                                            reps.append(l)
                                            reps[len(reps) - 1] = deepcopy(l)
                #               sino
                                    else:
                #                      adiciona closer a reps si ya no estaba en reps
                                        if l not in reps:
                                            reps.append(closer)
                                            reps[len(reps) - 1] = deepcopy(closer)
                                            # Guarda checkpoint de los representatives de cada especie y lo copia a ubicación para servir vía syn.
                                            # rep.save_checkpoint(config,pop,neat.DefaultSpeciesSet,rep.current_generation)
                    print("\nreps=",reps)
                    filename = '{0}{1}'.format("reps-", rep.current_generation)
                    with open(filename, 'wb') as f:
                        pickle.dump(reps, f)
                    #
                    # Hace request de CreateParam a syn
                    form_data = {"process_hash": "ph", "app_hash": "ah",
                        "parameter_link": my_url + "/genoms/" + filename,
                        "parameter_text": pop.best_genome.key, "parameter_blob": "", "validation_hash": "",
                        "hash": "h", "performance": best_fitness, "redir": "1", "username": "harveybc",
                        "pass_hash": "$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q"}
                    # TODO: COLOCAR DIRECCION CONFIGURABLE
                    res = requests.post(
                                        my_url + "/parameters?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph",
                                        data=form_data)
                    res_json = res.json()
                # TODO FIN: FUNCION DE SINCRONIZACION CON SINGULARITY
            temp = temp + 1
            
            # EVALUATE THE GENOMES WITH THE SUBSET TRAINING DATASET
            gen_best = pop.run(ec.evaluate_genomes, num_iterations)
            # TODO:
            # VERIFY IF THERE ARE PENDING EVALUATIONS
            # EVALUATE NUM_EVALUATIONS PENDING EVALUATIONS
                
            #print(gen_best)
            #visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")
            #plt.plot(ec.episode_score, 'g-', label='score')
            #plt.plot(ec.episode_length, 'b-', label='length')
            #plt.grid()
            #plt.legend(loc='best')
            #plt.savefig("scores.svg")
            #plt.close()

            #mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            #print("Average mean fitness over last 5 generations: {0}".format(mfs))

            #mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            #print("Average min fitness over last 3 generations: {0}".format(mfs))
            if avg_score < 2000000000:
                solved = False

            if solved:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                    visualize.draw_net(config, g, view=False, filename=name + "-net-enabled.gv",
                                       show_disabled=False)
                    visualize.draw_net(config, g, view=False, filename=name + "-net-enabled-pruned.gv",
                                       show_disabled=False, prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    env.close()


if __name__ == '__main__':
    run()