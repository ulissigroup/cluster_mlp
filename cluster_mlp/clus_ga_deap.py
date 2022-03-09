import random
from deap import base
from deap import creator
from deap import tools
from cluster_mlp.fillPool import fillPool
from cluster_mlp.mutations import (
    homotop,
    rattle_mut,
    rotate_mut,
    twist,
    tunnel,
    partialInversion,
    mate,
    skin,
    changeCore,
)
import copy
import ase.db
from ase.calculators.singlepoint import SinglePointCalculator as sp
from cluster_mlp.utils import write_to_db, checkBonded, checkSimilar, checkOverlap
from ase.io.trajectory import TrajectoryWriter
from ase.io  import read, write, Trajectory
import ase
import dask.bag as db
import tempfile
from ase.optimize import BFGS
import sys
import os
import subprocess


def minimize(clus, calculator, optimizer,vasp_inter):
    """
    Cluster relaxation using an ase optimizer
    Refer https://wiki.fysik.dtu.dk/ase/ase/optimize.html for a list of possible optimizers
    Recommended optimizer with VASP is GPMin
    """
    if vasp_inter == True:
        with calculator as calc:
            clus.calc = calculator
            with tempfile.TemporaryDirectory() as tmp_dir:
                clus.get_calculator().set(directory=tmp_dir)
            dyn = optimizer(clus, logfile=None)
            dyn.run(fmax=0.05, steps=2000)
            energy = clus.get_potential_energy()
    else:
        clus.calc = calculator
        with tempfile.TemporaryDirectory() as tmp_dir:
            clus.get_calculator().set(directory=tmp_dir)
        dyn = optimizer(clus, logfile=None)
        dyn.run(fmax=0.05, steps=1000)
        energy = clus.get_potential_energy()
    clus.set_calculator(sp(atoms=clus, energy=energy))
    return clus


def minimize_vasp(clus, calc):
    """
    Cluster relaxation function for using the inbuilt VASP optimizer
    All files related to the vasp run (INCAR,OUTCAR etc)
    are stored in a temporary directory
    """
    clus.calc = calc
    with tempfile.TemporaryDirectory() as tmp_dir:
        clus.get_calculator().set(directory=tmp_dir)
    energy = clus.get_potential_energy()
    clus.set_calculator(sp(atoms=clus, energy=energy))
    return clus


def minimize_al(
    clus, calc, eleNames, al_learner_params, train_config, dataset_parent, optimizer, al_method
):
    """
    Cluster relaxation function that employs using active learning
    For more information refer: https://github.com/ulissigroup/al_mlp
    Support provided for both online and offline methods
    """

    # Import al run functions
    from cluster_mlp.online_al_new import run_onlineal
    #from cluster_mlp.offline_al import run_offlineal Temporary issues with pytorch/cuda version conflicts

    with open("al_relaxation.out", "a+") as fh:
        fh.write(" Cluster geometry before Al relaxation \n")
        for atom in clus:
            fh.write(
                "{} {:12.8f} {:12.8f} {:12.8f} \n".format(
                    atom.symbol, atom.x, atom.y, atom.z
                )
            )

    clus.calc = calc
    if al_method == "online":
        relaxed_cluster, parent_calls, parent_dataset = run_onlineal(
            clus, calc, eleNames, al_learner_params, train_config, dataset_parent, optimizer
        )
    elif al_method == "offline":
        relaxed_cluster, parent_calls = run_offlineal(
            clus, calc, eleNames, al_learner_params, train_config, optimizer
        )
    else:
        sys.exit("Incorrect values for al_method, please use only offline or online")

    with open("al_relaxation.out", "a+") as fh:
        fh.write(" cluster geometry after Al relaxation \n")
        for atom in relaxed_cluster:
            fh.write(
                "{} {:12.8f} {:12.8f} {:12.8f} \n".format(
                    atom.symbol, atom.x, atom.y, atom.z
                )
            )
        fh.write(" \n")

    return relaxed_cluster, parent_calls


def fitness_func(individual):
    """
    Obtain the stored energy values in a deap individual object containing a single cluster
    """
    clus = individual[0]
    energy = clus.get_potential_energy()
    return (energy,)


def cluster_GA(
    nPool,
    eleNames,
    eleNums,
    eleRadii,
    generations,
    calc,
    filename,
    log_file,
    CXPB=0.5,
    singleTypeCluster=False,
    use_dask=False,
    use_vasp=False,
    al_method=None,
    al_learner_params=None,
    train_config=None,
    optimizer=BFGS,
    use_vasp_inter = False,
    restart = False,
    gen_num=None,
):
    """
    DEAP Implementation of the GIGA Geneting Algorithm for nanoclusters

    nPool : Total number of clusters present in the initial pool
    eleNames : List of element symbols present in the cluster
    eleNums : List of the number of atoms of each element present in the cluster
    eleRadii : List of radii of each element present in the cluster
    generations : Total number of generations to run the genetic algorithm
    calc : The calculator used to perform relaxations (must be an ase calculator object)
    filename : Name of the file to be used to generate ase traj and db files
    log_file : Name of the log file
    CXPB : probability of a crossover operation in a given generation
    singleTypeCluster : Default = False, set to True if only 1 element is present in cluster
    use_Dask : Default = False, set to True if using dask (Refer examples on using dask)
    use_vasp : Default = False, set to True if using inbuilt vasp optimizer to run GA code (not supported with active learning)
    al_method : Default = None, accepts values 'online' or 'offline'
    al_learner_params : Default = None, refer examples or https://github.com/ulissigroup/al_mlp for sample set up
    trainer_config : Default = None, refer examples or https://github.com/ulissigroup/al_mlp for sample set up
    optimizer : Default = BFGS, ase optimizer to be used
    use_vasp_inter : Default = False, whether to use vasp interactive mode or not
    """

    def calculate(atoms):
        """
        Support function to assign the type of minimization to e performed (pure vasp, using ase optimizer or using active learning)
        """
        if al_method is not None:

            atoms_min, parent_calls  = minimize_al(
                atoms,
                calc,
                eleNames,
                al_learner_params,
                train_config,
                dataset_parent,
                optimizer,
                al_method,
            )
        else:
            parent_calls = 0
            if use_vasp == True:
                atoms_min = minimize_vasp(atoms, calc)
            else:
                atoms_min = minimize(atoms, calc, optimizer,use_vasp_inter)
        return atoms_min, parent_calls

    if al_method is not None:
        al_method = al_method.lower()

    # Creating DEAP types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Registration of the evolutionary tools in the toolbox
    toolbox = base.Toolbox()
    toolbox.register("poolfill", fillPool, eleNames, eleNums, eleRadii, calc)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.poolfill, 1
    )
    toolbox.register("evaluate", fitness_func)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Registering mutations and crossover operators
    toolbox.register("mate", mate)
    toolbox.register("mutate_homotop", homotop)
    toolbox.register("mutate_rattle", rattle_mut)
    toolbox.register("mutate_rotate", rotate_mut)
    toolbox.register("mutate_twist", twist)
    toolbox.register("mutate_tunnel", tunnel)
    toolbox.register("mutate_partialinv", partialInversion)
    toolbox.register("mutate_skin", skin)
    toolbox.register("mutate_changecore", changeCore)

    # Registering selection operator
    toolbox.register("select", tools.selTournament)

    #Initialize the parent dataset
    dataset_parent = []
    # Creating a list of cluster atom objects from population
    if not restart:
        population = toolbox.population(n=nPool)
        pop_list = []
        for individual in population:
            pop_list.append(individual[0])
        write('init_pop_before_relax.traj', pop_list)


        if use_dask == True:
            # distribute and run the calculations (requires dask and needs to be set up correctly)
            clus_bag = db.from_sequence(pop_list, partition_size=1)
            clus_bag_computed = clus_bag.map(calculate)
            lst_clus_min = clus_bag_computed.compute()

        else:
            lst_clus_min  = list(map(calculate, pop_list))

        for i, p in enumerate(population):
            p[0] = lst_clus_min[i][0]
    
        init_pop_list_after_relax = []
        for individual in population:
            init_pop_list_after_relax.append(individual[0])
        write('init_pop_after_relax.traj', init_pop_list_after_relax)
        with open(log_file, "a+") as fh:
            fh.write(f'Total clusters in the intital pool after relaxationi: {len(population)}'"\n")
        
        
        #parent_calls list if online learner
        total_parent_calls = []
        parent_calls_initial_pool = []
        for i in range(len(lst_clus_min)):
            parent_calls_initial_pool.append(lst_clus_min[i][1])
        total_parent_calls.extend(parent_calls_initial_pool)

        with open(log_file, "a+") as fh:
            fh.write(f'parent calls after initial pool relaxation: {parent_calls_initial_pool}' '\n')
            fh.write(f'Total parent calls after initial pool relaxation: {sum(parent_calls_initial_pool)}' '\n')
    

        # Fitnesses (or Energy) values of the initial random population
        fitnesses = list(map(toolbox.evaluate, population))

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        #Removing bad geometries     
        population_filter = []
        for i, p in enumerate(population):
            if checkBonded(p[0]) == True:
                if checkOverlap(p[0]) == False:
                    population_filter.append(p)      
        population  = copy.deepcopy(population_filter)

   
        population = tools.selWorst(population,  len(population))   
 
        init_pop_list_after_filter = []
        for individual in population:
            init_pop_list_after_filter.append(individual[0])
        write('init_pop_after_filter.traj', init_pop_list_after_filter)
        with open(log_file, "a+") as fh:
            fh.write(f'Total clusters in the intital pool after filtering: {len(population)}'"\n")
    
        fitnesses_init_pool = list(map(toolbox.evaluate, population))
        with open(log_file, "a+") as fh:
            fh.write("Energies (fitnesses) of the initial pool" "\n")
            for value in fitnesses_init_pool:
                fh.write("{} \n".format(value[0]))


        # Evolution of the Genetic Algorithm
        with open(log_file, "a+") as fh:
            fh.write("\n")
            fh.write("Starting Evolution" "\n")

        g = 0  # Generation counter

        init_pop_db = ase.db.connect("init_pop_{}.db".format(filename))
        for cl in population:
            write_to_db(init_pop_db, cl[0])

        bi = []

    else:
        population = toolbox.population(n=nPool)
        restart_gen = 'best_n_clus_after_gen'+str(gen_num)+'.traj'
        restart_traj = Trajectory(restart_gen)
        for i, p in enumerate(population):
            p[0] = restart_traj[i]
        # Fitnesses (or Energy) values of the restart population from the gen_num
        fitnesses = list(map(toolbox.evaluate, population))

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        fitnesses_restart_pool = list(map(toolbox.evaluate, population))

        # Restarting the Evolution of the Genetic Algorithm from Restart Trajectory
        with open(log_file, "a+") as fh:
            fh.write("\n")
            fh.write("Restarting  Evolution" "\n")
            fh.write("Energies (fitnesses) of the Restarted pool" "\n")
            for value in fitnesses_restart_pool:
                fh.write("{} \n".format(value[0]))
            fh.write("\n")

        g = gen_num  # Generation counter -Restart gen number
        #parent_calls list if online learner
        total_parent_calls = []
        bi = []
        
        old_final_pop_db = "./final_pop_{}.db".format(filename)
        copy_final_pop_db = "final_pop_{}_{}.db".format(filename,gen_num)
        if os.path.exists(old_final_pop_db):
            subprocess.call( ['mv', old_final_pop_db, 'old_'+copy_final_pop_db ] )

    ##### Evolution of Generations ######

    while g < generations:
        mutType = None
        muttype_list = []
        g = g + 1
        with open(log_file, "a+") as fh:
            fh.write("{} {} \n".format("Generation", g))

        cm_pop = []
        if random.random() < CXPB:  # Crossover Operation
            mutType = "crossover"
            with open(log_file, "a+") as fh:
                fh.write("{} {} \n".format("mutType", mutType))

            # Crossover operation step.
            # The child clusters will be checked for bonding and similarity
            # between other child clusters.
            loop_count = 0
            while (
                loop_count != 200
            ):  # Perform 200 possible crossovers or until unique crossovers match pool size
                clusters = toolbox.select(population, 2, 1)
                muttype_list.append(mutType)
                parent1 = copy.deepcopy(clusters[0])
                parent2 = copy.deepcopy(clusters[1])
                fit1 = clusters[0].fitness.values
                (f1,) = fit1
                fit2 = clusters[1].fitness.values
                (f2,) = fit2
                child_clus = toolbox.mate(parent1[0], parent2[0], f1, f2)
                parent1[0] = child_clus

                diff_list = []
                if checkBonded(parent1[0]) == True:
                    if loop_count == 0:
                        cm_pop.append(parent1)
                    else:
                        for c, cluster in enumerate(cm_pop):
                            diff = checkSimilar(cluster[0], parent1[0])
                            diff_list.append(diff)

                        if all(diff_list) == True:
                            cm_pop.append(parent1)
                loop_count = loop_count + 1

                if len(cm_pop) == nPool:
                    break

        else:  # Mutation Operation
            mutType = "mutations"
            with open(log_file, "a+") as fh:
                fh.write("{} {} \n".format("mutType", mutType))

                # Mutation opeation step
                # Each cluster in the population will undergo a randomly chosen mutation step
                # Mutated new clusters will be checked for bonding and similarity with other new clusters
            for m, mut in enumerate(population):
                mutant = copy.deepcopy(mut)
                if singleTypeCluster:
                    mutType = random.choice(
                        [
                            "rattle",
                            "rotate",
                            "twist",
                            "partialinv",
                            "tunnel",
                            "skin",
                            "changecore",
                        ]
                    )
                else:
                    mutType = random.choice(
                        [
                            "rattle",
                            "rotate",
                            "homotop",
                            "twist",
                            "partialinv",
                            "tunnel",
                            "skin",
                            "changecore",
                        ]
                    )

                muttype_list.append(mutType)

                if mutType == "homotop":
                    mutant[0] = toolbox.mutate_homotop(mutant[0])
                if mutType == "rattle":
                    mutant[0] = toolbox.mutate_rattle(mutant[0])
                if mutType == "rotate":
                    mutant[0] = toolbox.mutate_rotate(mutant[0])
                if mutType == "twist":
                    mutant[0] = toolbox.mutate_twist(mutant[0])
                if mutType == "tunnel":
                    mutant[0] = toolbox.mutate_tunnel(mutant[0])
                if mutType == "partialinv":
                    mutant[0] = toolbox.mutate_partialinv(mutant[0])
                if mutType == "skin":
                    mutant[0] = toolbox.mutate_skin(mutant[0])
                if mutType == "changecore":
                    mutant[0] = toolbox.mutate_changecore(mutant[0])

                diff_list = []
                if checkBonded(mutant[0]) == True:
                    for c, cluster in enumerate(cm_pop):
                        diff = checkSimilar(cluster[0], mutant[0])
                        diff_list.append(diff)

                    if all(diff_list) == True:
                        cm_pop.append(mutant)

            with open(log_file, "a+") as fh:
                fh.write("{} {} \n".format("mutType_list", muttype_list))

        mut_new_lst = []
        for mut in cm_pop:
            mut_new_lst.append(mut[0])
        write('mut_before_relax_gen'+str(g)+'.traj', mut_new_lst)

        # DASK Parallel relaxation of the crossover child/mutated clusters
        if use_dask == True:
            mut_bag = db.from_sequence(mut_new_lst, partition_size=1)
            mut_bag_computed = mut_bag.map(calculate)
            mut_new_lst_min = mut_bag_computed.compute()

        else:
            mut_new_lst_min = list(map(calculate, mut_new_lst))

        for i, mm in enumerate(cm_pop):
            mm[0] = mut_new_lst_min[i][0]
        
        
        mut_list_after_relax = []
        for individual in cm_pop:
            mut_list_after_relax.append(individual[0])
        write('mut_after_relax_gen'+str(g)+'.traj', mut_list_after_relax)
        with open(log_file, "a+") as fh:
            fh.write(f'Total clusters relaxed in  Generation {g}: {len(cm_pop)}'"\n")

        #parent calls list if online learner
        parent_calls_mut_list = []
        for i in range(len(mut_new_lst_min)):
            parent_calls_mut_list.append(mut_new_lst_min[i][1])
        
        total_parent_calls.extend(parent_calls_mut_list)

        with open(log_file, "a+") as fh:
            fh.write(f'Parent calls list after relaxtions in this generation: {parent_calls_mut_list} ' '\n')
            fh.write(f'Total Parent calls  specific to this  generation: {sum(parent_calls_mut_list)} ' '\n')
            fh.write(f'Total Parent calls  up  to this  generation: {sum(total_parent_calls)} ' '\n')
        
         

        fitnesses_mut = list(map(toolbox.evaluate, cm_pop))

        for ind, fit in zip(cm_pop, fitnesses_mut):
            ind.fitness.values = fit

        new_population = copy.deepcopy(population)
        # Relaxed clusters will be checked for bonded and similarity with the other
        # clusters in the population. If dissimilar, they will be added to the new population.
        for cm1, cmut1 in enumerate(cm_pop):
            new_diff_list = []
            if checkBonded(cmut1[0]) == True:
                if checkOverlap(cmut1[0]) == False:
                    for c2, cluster1 in enumerate(population):
                        diff = checkSimilar(cluster1[0], cmut1[0])
                        new_diff_list.append(diff)
                    if all(new_diff_list) == True:
                        new_population.append(cmut1)
                    else:
                        pass
        
        mut_list_after_filter = []
        for individual in new_population:
            mut_list_after_filter.append(individual[0])
        write('mut_after_filter_gen'+str(g)+'.traj', mut_list_after_filter)
        with open(log_file, "a+") as fh:
            fh.write(f'Total clusters flitered out in  Generation {g}: {len(cm_pop) + len(population) - len(new_population)}'"\n")
            fh.write(f'Total clusters in the pool after filtering in Generation {g}: {len(new_population)}'"\n")



        fitnesses_pool = list(map(toolbox.evaluate, new_population))

        with open(log_file, "a+") as fh:
            fh.write(
                "Energies (fitnesses) of the present pool before best 10 are selected"
                "\n"
            )
            for value in fitnesses_pool:
                fh.write("{} \n".format(value[0]))

        # Selecting the lowest energy npool clusters from the new_population
        len_new_pop = len(new_population)
        if len_new_pop > nPool: 
            best_n_clus = tools.selWorst(new_population, nPool)
        else:
            best_n_clus = new_population

        best_n_clus_list = []
        for individual in best_n_clus:
            best_n_clus_list.append(individual[0])
        write('best_n_clus_after_gen'+str(g)+'.traj', best_n_clus_list)
        

        population = best_n_clus

        best_clus = tools.selWorst(population, 1)[0]
        with open(log_file, "a+") as fh:
            fh.write(
                "{} {} \n".format(
                    "Lowest energy for this generation is", best_clus.fitness.values[0]
                )
            )
            fh.write("\n Best cluster in this generation: \n")
            for atom in best_clus[0]:
                fh.write(
                    "{} {:12.8f} {:12.8f} {:12.8f} \n".format(
                        atom.symbol, atom.x, atom.y, atom.z
                    )
                )
            fh.write("\n")

        bi.append(best_clus[0])
        if g == 1:
            writer = TrajectoryWriter(
                filename + "_best.traj", mode="w", atoms=best_clus[0]
            )
            writer.write()
        else:
            writer = TrajectoryWriter(
                filename + "_best.traj", mode="a", atoms=best_clus[0]
            )
            writer.write()

    final_pop_db = ase.db.connect("final_pop_{}.db".format(filename))
    for clus in population:
        write_to_db(final_pop_db, clus[0])

    with open(log_file, "a+") as fh:
        fh.write("Global Minimum after {} Generations \n".format(g))
        for atom in best_clus[0]:
            fh.write(
                "{} {:12.8f} {:12.8f} {:12.8f} \n".format(
                    atom.symbol, atom.x, atom.y, atom.z
                )
            )
    # Return the list of best clusters in every generations and the overall best cluster
    return bi, best_clus[0]
