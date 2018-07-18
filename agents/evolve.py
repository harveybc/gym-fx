# Evolve a control/reward estimation network for the OpenAI Gym
# LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
# Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg

from __future__ import print_function

import gym
import gym.wrappers

import matplotlib.pyplot as plt

import multiprocessing
import neat
import numpy as np
import os
import pickle
import random
import time

import visualize

NUM_CORES = 8

env = gym.make('LunarLander-v2')

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))

env = gym.wrappers.Monitor(env, 'results', force=True)


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
            observation = row[:8]
            action = int(row[8])
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))

    return reward_error


class PooledErrorCompute(object):
    def __init__(self):
        self.pool = None if NUM_CORES < 2 else multiprocessing.Pool(NUM_CORES)
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200

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
                if step < 200 and random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    output = net.activate(observation)
                    action = np.argmax(output)

                observation, reward, done, info = env.step(action)
                data.append(np.hstack((observation, action, reward)))

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
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute()
    while 1:
        try:
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
            for g in best_genomes:
                best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))

            solved = True
            best_scores = []
            for k in range(100):
                observation = env.reset()
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all five networks to
                    # determine the best action given the current state.
                    votes = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(observation)
                        votes[np.argmax(output)] += 1

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
                if avg_score < 200:
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