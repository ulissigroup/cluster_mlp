import random
from ase.optimize import BFGS
from deap import base
from deap import creator
from deap import tools
from fillPool import fillPool
from mutations import homotop,rattle_mut,rotate_mut,twist,tunnel,partialInversion,mate
import copy
import ase.db
from utils import write_to_db,checkBonded,checkSimilar

def fitness_func1(individual):
	atoms = individual[0]
	energy = atoms.get_potential_energy()
	return energy,

def fitness_func2(individual,calc):
	atoms = individual[0]
	atoms.set_calculator(calc)
	dyn = BFGS(atoms,logfile = None)
	dyn.run(fmax = 0.05)
	energy = atoms.get_potential_energy()
	return energy,

def cluster_GA(nPool,eleNames,eleNums,eleRadii,generations,calc,filename,CXPB = 0.5, MUTPB = 0.2,singleTypeCluster = False):
	best_db = ase.db.connect("{}.db".format(filename))
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list,fitness=creator.FitnessMax)

	toolbox = base.Toolbox()

	toolbox.register("poolfill", fillPool,eleNames,eleNums,eleRadii,calc)

	toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.poolfill,1)


	toolbox.register("evaluate1", fitness_func1)
	toolbox.register("evaluate2", fitness_func2,calc = calc)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	#REGISTERING MUTATIONS AND CROSSOVER
	toolbox.register("mate", mate)
	toolbox.register("mutate_homotop", homotop)
	toolbox.register("mutate_rattle", rattle_mut)
	toolbox.register("mutate_rotate", rotate_mut)
	toolbox.register("mutate_twist", twist)
	toolbox.register("mutate_tunnel", tunnel)
	toolbox.register("mutate_partialinv",partialInversion)

	toolbox.register("select", tools.selRoulette)


	population = toolbox.population(n=nPool)

	fitnesses = list(map(toolbox.evaluate1, population))
	for ind, fit in zip(population, fitnesses):
	        ind.fitness.values = fit
	g = 0
	fits = [ind.fitness.values[0] for ind in population]

	init_pop_db = ase.db.connect("init_pop_{}.db".format(filename))
	for cl in population:
		write_to_db(init_pop_db,cl[0])

	bi = []
	while g < generations:
			mutType = None
			g = g + 1
			print('Generation',g)
			print('Starting Evolution')

			if random.random() < CXPB:
				clusters = toolbox.select(population,2)
				index1 = population.index(clusters[0])
				index2 = population.index(clusters[1])
				mutType = 'crossover'
				parent1 = copy.deepcopy(clusters[0])
				parent2 = copy.deepcopy(clusters[1])
				fit1 = clusters[0].fitness.values
				f1, = fit1
				fit2 = clusters[1].fitness.values
				f2, = fit2
				toolbox.mate(parent1[0],parent2[0],f1,f2)
				new_fitness = fitness_func2(parent1,calc)

				diff_list = []
				if checkBonded(parent1[0]) == True:
					for c,cluster in enumerate(population):
						diff = checkSimilar(cluster[0],parent1[0])
						diff_list.append(diff)

					if all(diff_list) == True:
						highest_energy_ind = tools.selBest(population,1)[0]
						hei_index = population.index(highest_energy_ind)
						hei_fitness = highest_energy_ind.fitness.values
						if new_fitness < hei_fitness:
							del highest_energy_ind.fitness.values
							population.pop(hei_index)
							highest_energy_ind = parent1
							highest_energy_ind.fitness.values = new_fitness
							population.append(highest_energy_ind)

			if random.random() < MUTPB and mutType != 'crossover':
				mut = toolbox.select(population,1)
				index3 = population.index(mut[0])
				mutant = copy.deepcopy(mut[0])
				if singleTypeCluster:
					mutType = random.choice(['rattle','rotate','twist','partialinv'])
				else:
					mutType = random.choice(['rattle','homotop','rotate','twist','partialinv'])

				if mutType == 'homotop':
					toolbox.mutate_homotop(mutant[0])
				if mutType == 'rattle':
					toolbox.mutate_rattle(mutant[0])
				if mutType == 'rotate':
					toolbox.mutate_rotate(mutant[0])
				if mutType == 'twist':
					toolbox.mutate_twist(mutant[0])
				if mutType == 'tunnel':
					toolbox.mutate_tunnel(mutant[0])
				if mutType == 'partialinv':
					toolbox.mutate_partialinv(mutant[0])

				new_fitness = fitness_func2(mutant,calc)

				diff_list = []
				if checkBonded(mutant[0]) == True:
					for c,cluster in enumerate(population):
						diff = checkSimilar(cluster[0],mutant[0])
						diff_list.append(diff)

					if all(diff_list) == True:
						highest_energy_ind = tools.selBest(population,1)[0]
						hei_index = population.index(highest_energy_ind)
						hei_fitness = highest_energy_ind.fitness.values
						if new_fitness < hei_fitness:
							del highest_energy_ind.fitness.values
							population.pop(hei_index)
							highest_energy_ind = mutant
							highest_energy_ind.fitness.values = new_fitness
							population.append(highest_energy_ind)
			print(mutType)
			best_ind = tools.selWorst(population,1)[0]
			print('Lowest energy individual is',best_ind)
			print('Lowest energy is',best_ind.fitness.values)
			bi.append(best_ind[0])
			write_to_db(best_db,best_ind[0])

	final_pop_db = ase.db.connect("final_pop_{}.db".format(filename))
	for clus in population:
		write_to_db(final_pop_db,clus[0])

	return bi,best_ind[0]