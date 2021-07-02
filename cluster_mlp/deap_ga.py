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
from cluster_mlp.utils import write_to_db, checkBonded, checkSimilar
from ase.io.trajectory import TrajectoryWriter
import ase
import dask.bag as db
import tempfile
from ase.optimize import BFGS
import sys


def minimize(clus, calc, optimizer):
    """
    Cluster relaxation using an ase optimizer
    Refer https://wiki.fysik.dtu.dk/ase/ase/optimize.html for a list of possible optimizers
    Recommended optimizer with VASP is GPMin
    """
    clus.calc = calc
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
    clus, calc, eleNames, al_learner_params, train_config, optimizer, al_method
):
    """
    Cluster relaxation function that employs using active learning
    For more information refer: https://github.com/ulissigroup/al_mlp
    Support provided for both online and offline methods
    """

    # Import al run functions
    from cluster_mlp.online_al import run_onlineal
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
        relaxed_cluster, parent_calls = run_onlineal(
            clus, calc, eleNames, al_learner_params, train_config, optimizer
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

    return relaxed_cluster


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
    """

    def calculate(atoms):
        """
        Support function to assign the type of minimization to e performed (pure vasp, using ase optimizer or using active learning)
        """
        if al_method is not None:

            atoms_min = minimize_al(
                atoms,
                calc,
                eleNames,
                al_learner_params,
                train_config,
                optimizer,
                al_method,
            )
        else:
            if use_vasp == True:
                atoms_min = minimize_vasp(atoms, calc)
            else:
                atoms_min = minimize(atoms, calc, optimizer)
        return atoms_min

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

    population = toolbox.population(n=nPool)

    # Creating a list of cluster atom objects from population
    pop_list = []
    for individual in population:
        pop_list.append(individual[0])

    if use_dask == True:
        # distribute and run the calculations (requires dask and needs to be set up correctly)
        clus_bag = db.from_sequence(pop_list, partition_size=1)
        clus_bag_computed = clus_bag.map(calculate)
        lst_clus_min = clus_bag_computed.compute()

    else:
        lst_clus_min = list(map(calculate, pop_list))

    for i, p in enumerate(population):
        p[0] = lst_clus_min[i]

    # Fitnesses (or Energy) values of the initial random population
    fitnesses = list(map(toolbox.evaluate, population))

    with open(log_file, "a+") as fh:
        fh.write("Energies (fitnesses) of the initial pool" "\n")
        for value in fitnesses:
            fh.write("{} \n".format(value[0]))

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Evolution of the Genetic Algorithm
    with open(log_file, "a+") as fh:
        fh.write("\n")
        fh.write("Starting Evolution" "\n")

    g = 0  # Generation counter

    init_pop_db = ase.db.connect("init_pop_{}.db".format(filename))
    for cl in population:
        write_to_db(init_pop_db, cl[0])

    bi = []

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
                toolbox.mate(parent1[0], parent2[0], f1, f2)

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

        # DASK Parallel relaxation of the crossover child/mutated clusters
        if use_dask == True:
            mut_bag = db.from_sequence(mut_new_lst, partition_size=1)
            mut_bag_computed = mut_bag.map(calculate)
            mut_new_lst_min = mut_bag_computed.compute()

        else:
            mut_new_lst_min = list(map(calculate, mut_new_lst))

        for i, mm in enumerate(cm_pop):
            mm[0] = mut_new_lst_min[i]

        fitnesses_mut = list(map(toolbox.evaluate, cm_pop))

        for ind, fit in zip(cm_pop, fitnesses_mut):
            ind.fitness.values = fit

        new_population = copy.deepcopy(population)
        # Relaxed clusters will be checked for bonded and similarity with the other
        # clusters in the population. If dissimilar, they will be added to the new population.
        for cm1, cmut1 in enumerate(cm_pop):
            new_diff_list = []
            if checkBonded(cmut1[0]) == True:
                for c2, cluster1 in enumerate(population):
                    diff = checkSimilar(cluster1[0], cmut1[0])
                    new_diff_list.append(diff)
                if all(new_diff_list) == True:
                    new_population.append(cmut1)
                else:
                    pass

        fitnesses_pool = list(map(toolbox.evaluate, new_population))

        with open(log_file, "a+") as fh:
            fh.write(
                "Energies (fitnesses) of the present pool before best 10 are selected"
                "\n"
            )
            for value in fitnesses_pool:
                fh.write("{} \n".format(value[0]))

        # Selecting the lowest energy npool clusters from the new_population
        best_n_clus = tools.selWorst(new_population, nPool)
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
