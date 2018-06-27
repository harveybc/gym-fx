import requests
from neat import Population
from copy import deepcopy 
import pickle
from neat.six_util import iteritems
from neat.six_util import itervalues
# PopulationSyn class for synchronizing optimization states with the singularity p2p optimzation network

# PopulationSyn extends Population
class PopulationSyn(Population):
    #def getBestGenomes(genomes_h, number)
    
    # replaces in remote the genomes that has less_fit_key
    def replaceGenomes(self, genomes, less_fit_key, remote):
        genomes_h = []
        for g in genomes:
            if g.key == less_fit_key:
                #remote.key = less_fit_key
                g = remote
            genomes_h.append(g)
        return genomes_h
            
    # calculateFitness(best_genomes)
    def calculateFitness(self, best_genomes):
        countr=0
        accum=0
        best=None
        max_fitness=-1000
        #search for max fitness
        for n, g in enumerate(best_genomes):
            accum=accum+g.fitness
            countr = countr + 1
            #print(' fit',n,'=',g.fitness)
            if (g.fitness > max_fitness):
                max_fitness = g.fitness
                best = g
        if countr > 0:    
            best_fitness = ((len(best_genomes)-1)*g.fitness+(accum/countr))/len(best_genomes)
        else:
            best_fitness = -100000
        return best_fitness
    
    # searchLessFit()
    def searchLessFit(self, genomes_h):
        less_fit = None
        min_fitness = 10000
        #print('\ngenomes_h = ',genomes_h)
        for g in genomes_h:
            #print('\ng = ',g)
            #print('\ng[1].key = ',g[1].key)
            #print('\ng[1].fitness=',g[1].fitness)
            if g.fitness < min_fitness:
                min_fitness = g.fitness
                less_fit = g
        return less_fit

    # synSingularity method for synchronizing NEAT optimization states with singularity 
    # args: num_replacements = number of specimens to be migrated to/from singularity
    #       my_url = url of the singularity API
    #       stats = neat.StatisticsReporter
    # returns: best_genoms selected between the remote and local
    def syn_singularity(self, num_replacements, my_url, stats, avg_score, current_generation, config, genomes_h):
        # downloads process from singualrity to find last optimum
        print('num_rep=', num_replacements,'my_url=',  my_url,'stats=',  stats,
            'avg_score=',  avg_score, 'current_generation=',  current_generation)
        res = requests.get(my_url + "/processes/1?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
        cont = res.json()
        last_optimum_id = cont['result'][0]['last_optimum_id']
        # calcualte local_perf as the weitgthed average of the best performers
        best_genomes = stats.best_unique_genomes(num_replacements)
        local_perf = self.calculateFitness(best_genomes)
        # remote performance from results of request
        remote_perf = cont['result'][0]['current_block_performance']
        # print results of request
        print('\nremote_performance =', cont['result'][0]['current_block_performance'], '\nlocal_performance =', local_perf, '\nlast_optimum_id =', cont['result'][0]['last_optimum_id'])
        # if remote_performance is not equal to local_performance, download remote_reps
        parameter_downloaded = 0
        if (local_perf != remote_perf):
            # hace request GetParameter(id)
            res_p = requests.get(my_url + "/parameters/" + str(last_optimum_id) + "?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
            cont_param = res_p.json()
            print('\ncont_param =', cont_param)
            # descarga el checkpoint del link de la respuesta si cont.parameter_link
            if cont_param['result'][0]['parameter_link'] is not None:
                genom_data = requests.get(cont_param['result'][0]['parameter_link']).content
                with open('remote_reps', 'wb') as handler:
                    handler.write(genom_data)
                    handler.close()
                # carga genom descargado en nueva población pop2
                with open('remote_reps', 'rb') as f:
                    remote_reps = pickle.load(f)
                print('\nPARAMETERS DOWNLOADED: remote_reps=', remote_reps)
                parameter_downloaded = 1
                    
        # if local_perf < remote_perf
        if (local_perf < remote_perf) and (parameter_downloaded):
            # for each remote_reps as remote
            print('\nremote_fitness = ', remote_perf, 'local_fitness = ', local_perf)
            #print('\ngenomes_h = ',genomes_h)
            genomes = genomes_h
            for remote in remote_reps:
                # search the less_fit in pop
                less_fit = self.searchLessFit(genomes)
                # replaces less_fit with remote
                #print("less_fit = ", less_fit)
                #less_fit_key = less_fit.key
                less_fit_key = remote.key
                print('\nREPLACED = ', less_fit_key, 'fitness=', less_fit.fitness, 
                    'new_fitness', remote.fitness)
                #replaces lessfit in population by remote with the same key as less fit
                remote.key=less_fit_key
                self.population[less_fit_key] = remote
                genomes=self.replaceGenomes(genomes, less_fit_key, remote)
                
        # if local_perf > remote_perf
        if (local_perf >remote_perf):
            # upload best_genomes
            print('***********************************************************')
            print("\nNEW OPTIMUM") 
            for g in best_genomes:
                print("\nbest_genomes[i] = ",g.key,"  fitness = ",g.fitness) 
            filename = '{0}{1}'.format("reps-", current_generation)
            with open(filename, 'wb') as f:
                pickle.dump(best_genomes, f)        
            # Hace request de CreateParam a syn
            form_data = {"process_hash": "ph", "app_hash": "ah",
                "parameter_link": my_url + "/genoms/" + filename,
                "parameter_text": 0, "parameter_blob": "", "validation_hash": "",
                "hash": "h", "performance": local_perf, "redir": "1", "username": "harveybc",
                "pass_hash": "$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q"}
            res = requests.post(
                                my_url + "/parameters?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph",
                                data=form_data)
            res_json = res.json()
        return 0
    
    def evaluate_pending(self,max_pending):
        # TODO:
        # VERIFY IF THERE ARE PENDING EVALUATIONS IN SINGULARITY
        # EVALUATE NUM_EVALUATIONS PENDING EVALUATIONS
        return 0
    