import requests
from neat import Population
from copy import deepcopy 
import pickle
from neat.six_util import iteritems
from neat.six_util import itervalues
# PopulationSyn class for synchronizing optimization states with the singularity p2p optimzation network

# PopulationSyn extends Population
class PopulationSyn(Population):
    # calculateFitness(best_genomes)
    def calculateFitness(self, best_genomes):
        countr=0
        accum=0
        best=None
        max_fitness=-100000
        for n, g in enumerate(best_genomes):
            accum=accum+g.fitness
            countr = countr + 1
            if (g.fitness > max_fitness):
                max_fitness = g.fitness
                best=g
        if countr > 0:    
            best_fitness = ((len(best_genomes)-1)*g.fitness+(accum/countr))/len(best_genomes)
        else:
            best_fitness = 0
        return best_fitness
    
    # searchLessFit()
    def searchLessFit(self):
        less_fit = None
        min_fitness = 100000000
        for g in itervalues(self.population):
            if g.fitness < min_fitness:
                min_fitness = g.fitness
                less_fit = deepcopy(g)
        return less_fit                

    # synSingularity method for synchronizing NEAT optimization states with singularity 
    # args: num_replacements = number of specimens to be migrated to/from singularity
    #       my_url = url of the singularity API
    #       stats = neat.StatisticsReporter
    # returns: best_genoms selected between the remote and local
    def syn_singularity(self, num_replacements, my_url, stats, gen_best, avg_score, rep, config):
        # downloads process from singualrity to find last optimum
        res = requests.get(my_url + "/processes/1?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
        cont = res.json()
        # print results of request
        print('\nremote_performance =', cont['result'][0]['current_block_performance'])
        last_optimum_id = cont['result'][0]['last_optimum_id']
        # calcualte local_perf as the weitgthed average of the best performers
        best_genomes = stats.best_unique_genomes(num_replacements)
        local_perf = self.calculateFitness(best_genomes)
        # remote performance from results of request
        remote_perf = cont['result'][0]['current_block_performance']
        print('\nremote_performance =', cont['result'][0]['current_block_performance'], '\nlocal_performance =', local_perf, '\nlast_optimum_id =', cont['result'][0]['last_optimum_id'])
        # if remote_performance is not equal to local_performance, download remote_reps
        if (local_perf != remote_perf):
            # hace request GetParameter(id)
            res_p = requests.get(my_url + "/parameters/" + str(last_optimum_id) + "?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
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
        # if local_perf < remote_perf
        if (local_perf < remote_perf):
            # for each remote_reps as remote
            for remote in remote_reps:
                # search the less_fit in pop
                less_fit = self.searchLessFit(self)
                # replaces less_fit with remote
                less_fit_key = less_fit.key
                print(less_fit_key)
                self.population[less_fit_key] = less_fit
        # if local_perf > remote_perf
        if (local_perf >remote_perf):
            # upload local_reps
            print("\nreps=",reps)
            filename = '{0}{1}'.format("reps-", rep.current_generation)
            with open(filename, 'wb') as f:
                pickle.dump(reps, f)        
            # Hace request de CreateParam a syn
            form_data = {"process_hash": "ph", "app_hash": "ah",
                "parameter_link": my_url + "/genoms/" + filename,
                "parameter_text": 0, "parameter_blob": "", "validation_hash": "",
                "hash": "h", "performance": local_perf, "redir": "1", "username": "harveybc",
                "pass_hash": "$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q"}
            # TODO: COLOCAR DIRECCION CONFIGURABLE
            res = requests.post(
                                my_url + "/parameters?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph",
                                data=form_data)
            res_json = res.json()
        return 0
    
    def syn_singularity2(self, num_replacements, my_url, stats, gen_best, avg_score, rep, config):
        # requests the last optimization state TODO: HACER PROCESS CONFIGURABLE Y POR HASH no por id for multi-process
        res = requests.get(my_url + "/processes/1?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph")
        cont = res.json()
        # print results of request
        print('\ncurrent_block_performance =', cont['result'][0]['current_block_performance'])
        print('\nlast_optimum_id =', cont['result'][0]['last_optimum_id'])
        last_optimum_id = cont['result'][0]['last_optimum_id']
        # calcualte the best fitness as the weitgthed average of the best performers
        best_genomes = stats.best_unique_genomes(num_replacements)
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

        print("\nPerformance = ", best_fitness)
        print("*********************************************************")
        # Si el perf reportado pop2_champion_fitness > pop1_champion_fitness de validation training        
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
                    for g in itervalues(self.population):
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
                        self.population[closer.key] = deepcopy(tmp_genom)
                        print("gen_best=", closer.key)
                        self.best_genome = deepcopy(tmp_genom)
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
                                        self.population[closer.key] = deepcopy(tmp_genom)
                                        print("Replaced=", closer.key)
                                        # actualiza gen_best y best_genome al remoto
                                        self.best_genome = deepcopy(tmp_genom)
                                        #gen_best = deepcopy(tmp_genom)
                                if closer.fitness is None:
                                    tmp_genom = deepcopy(remote_reps[i])
                                    # Hack: overwrites original genome key with the replacing one
                                    tmp_genom.key = len(self.population) + 1
                                    self.population[tmp_genom.key] = tmp_genom
                                    print("Created Por closer.fitness=NONE : ", tmp_genom.key)
                                    # actualiza gen_best y best_genome al remoto
                                    self.best_genome = deepcopy(tmp_genom)
                                    #gen_best = deepcopy(tmp_genom)
                            else:
                                #si closer está en remote_reps es porque no hay ningun otro cercano así que lo adiciona
                                tmp_genom = deepcopy(remote_reps[i])
                                # Hack: overwrites original genome key with the replacing one
                                tmp_genom.key = len(self.population) + 1
                                self.population[tmp_genom.key] = tmp_genom
                                print("Created por Closer in rempte_reps=", tmp_genom.key)
                                # actualiza gen_best y best_genome al remoto
                                self.best_genome = deepcopy(tmp_genom)
                                #gen_best = deepcopy(tmp_genom)

                #ejecuta speciate
                self.species.speciate(config, self.population, self.generation)
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
            best_genomes = stats.best_unique_genomes(num_replacements)
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
                "parameter_text": self.best_genome.key, "parameter_blob": "", "validation_hash": "",
                "hash": "h", "performance": best_fitness, "redir": "1", "username": "harveybc",
                "pass_hash": "$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q"}
            # TODO: COLOCAR DIRECCION CONFIGURABLE
            res = requests.post(
                                my_url + "/parameters?username=harveybc&pass_hash=$2a$04$ntNHmofQoMoajG89mTEM2uSR66jKXBgRQJnCgqfNN38aq9UkN4Y6q&process_hash=ph",
                                data=form_data)
            res_json = res.json()
        # TODO FIN: FUNCION DE SINCRONIZACION CON SINGULARITY
        return 0
    
    def evaluate_pending(self,max_pending):
        # TODO:
        # VERIFY IF THERE ARE PENDING EVALUATIONS IN SINGULARITY
        # EVALUATE NUM_EVALUATIONS PENDING EVALUATIONS
        return 0
    