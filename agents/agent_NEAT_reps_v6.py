# This agent uses the forex_env_v2 that uses continuous and binary controls
from __future__ import print_function
from copy import deepcopy
from gym.envs.registration import register
from population_syn import PopulationSyn # extended neat population for synchronizing witn singularity p2p network
from genome_evaluator import GenomeEvaluator
import gym
import sys
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

# AgentGenome class
class AgentGenome(neat.DefaultGenome):
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
        return "Reward discount: {0}\n{1}".format(self.discount, super().__str__())
    
def run():
    # load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, my_config)
    config = neat.Config(AgentGenome, neat.DefaultReproduction,
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
    # class for evaluating the population
    ec = GenomeEvaluator()
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
                avg_score=pop.training_validation_score(gen_best)
            # if it is not the first iteration
            if iteration_counter >= 0:
                # synchronizes with singularity migrating maximum 3 specimens
                pop.syn_singularity(3, my_url, stats, gen_best,avg_score)
                # perform pending evaluations on the singularity network, max 2
                pop.evaluate_pending(2)
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