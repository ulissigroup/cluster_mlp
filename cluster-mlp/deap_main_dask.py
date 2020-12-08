import random
from ase.optimize import BFGS
from deap import base
from deap import creator
from deap import tools
from fillPool import fillPool
from mutations import homotop,rattle_mut,rotate_mut,twist,tunnel,partialInversion,mate,skin,changeCore
import copy
import ase.db
from ase.calculators.singlepoint import SinglePointCalculator as sp
from utils import write_to_db,checkBonded,checkSimilar
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import dask
import ase
import dask.bag as db
import tempfile

def minimize(clus,calc):
	clus.calc = copy.deepcopy(calc)
	with tempfile.TemporaryDirectory() as tmp_dir:
		clus.get_calculator().set(directory=tmp_dir)
	#dyn = BFGS(clus,logfile = None)
	#dyn.run(fmax = 0.05,steps = 1000)
	energy = clus.get_potential_energy()
	clus.set_calculator(sp(atoms=clus, energy=energy))
	return clus

def fitness_func1(individual):
	clus = individual[0]
	print(clus)
	energy = clus.get_potential_energy()
	return energy,


def  cluster_GA(nPool,eleNames,eleNums,eleRadii,generations,calc,filename,CXPB = 0.5,singleTypeCluster = False):
	best_db = ase.db.connect("{}.db".format(filename))
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list,fitness=creator.FitnessMax)

	toolbox = base.Toolbox()

	toolbox.register("poolfill", fillPool,eleNames,eleNums,eleRadii,calc)

	toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.poolfill,1)


	toolbox.register("evaluate1", fitness_func1)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	#REGISTERING MUTATIONS AND CROSSOVER
	toolbox.register("mate", mate)
	toolbox.register("mutate_homotop", homotop)
	toolbox.register("mutate_rattle", rattle_mut)
	toolbox.register("mutate_rotate", rotate_mut)
	toolbox.register("mutate_twist", twist)
	toolbox.register("mutate_tunnel", tunnel)
	toolbox.register("mutate_partialinv",partialInversion)
	toolbox.register("mutate_skin",skin)
	toolbox.register("mutate_changecore",changeCore)

	toolbox.register("select", tools.selTournament)


	population = toolbox.population(n=nPool)

	#Creating a list of cluster atom objects from pouplation
	pop_list = []
	for individual in population:
		pop_list.append(individual[0])


	#Dask Parallelization
	def calculate(atoms):
		atoms_min = minimize(atoms,calc)
		return atoms_min

	#distribute and run the calculations
	clus_bag = db.from_sequence(pop_list)
	clus_bag_computed = clus_bag.map(calculate)
	lst_clus_min = clus_bag_computed.compute()

	for clus in lst_clus_min:
		print(clus)
		print(clus.get_positions())
	
	for i,p in enumerate(population):
		p[0] = lst_clus_min[i]
	
	print(population,'pop')
	
	fitnesses = list(map(toolbox.evaluate1, population)) #USE DASK TO PARALLELIZE
	print('Random Initial population after Minimization')
	print('fitnesses:', fitnesses)
	for i in range(len(population)):
		print(population[i][0])
		print(population[i][0].get_positions())
	
	for ind, fit in zip(population, fitnesses):
		ind.fitness.values = fit
	
	g = 0
	init_pop_db = ase.db.connect("init_pop_{}.db".format(filename))
	for cl in population:
		write_to_db(init_pop_db,cl[0])

	bi = []
	while g < generations:
			mutType = None
			muttype_list = []
			g = g + 1
			print('Generation',g)
			print('Starting Evolution')
			cm_pop = []
			if random.random() < CXPB:
				mutType = 'crossover'
				print('mutType', mutType)
				loop_count = 0
				while  loop_count != 200:
					clusters = toolbox.select(population,2,1)
					muttype_list.append(mutType)
					parent1 = copy.deepcopy(clusters[0])
					parent2 = copy.deepcopy(clusters[1])
					fit1 = clusters[0].fitness.values
					f1, = fit1
					fit2 = clusters[1].fitness.values
					f2, = fit2
					toolbox.mate(parent1[0],parent2[0],f1,f2)

					diff_list = []
					if checkBonded(parent1[0]) == True:
						if loop_count == 0:
							cm_pop.append(parent1)
						else:
							for c,cluster in enumerate(cm_pop):
								diff = checkSimilar(cluster[0],parent1[0])
								diff_list.append(diff)

							if all(diff_list) == True:
								cm_pop.append(parent1)
					loop_count = loop_count+1
					if len(cm_pop) == 10:
						break

			else:
				mutType = 'mutations'
				print('mutType', mutType)
				for m,mut in enumerate(population):
					mutant = copy.deepcopy(mut)
					if singleTypeCluster:
						mutType = random.choice(['rattle','rotate','twist','partialinv','tunnel','skin','changecore'])
					else:
						mutType = random.choice(['rattle','rotate','homotop','twist','partialinv','tunnel','skin','changecore'])

					muttype_list.append(mutType)

					if mutType == 'homotop':
						mutant[0] = toolbox.mutate_homotop(mutant[0])
					if mutType == 'rattle':
						mutant[0] = toolbox.mutate_rattle(mutant[0])
					if mutType == 'rotate':
						mutant[0] = toolbox.mutate_rotate(mutant[0])
					if mutType == 'twist':
						mutant[0] = toolbox.mutate_twist(mutant[0])
					if mutType == 'tunnel':
						mutant[0] = toolbox.mutate_tunnel(mutant[0])
					if mutType == 'partialinv':
						mutant[0] = toolbox.mutate_partialinv(mutant[0])
					if mutType == 'skin':
						mutant[0] = toolbox.mutate_skin(mutant[0])
					if mutType == 'changecore':
						mutant[0] = toolbox.mutate_changecore(mutant[0])
					
					diff_list = []
					if checkBonded(mutant[0]) == True:
						for c,cluster in enumerate(cm_pop):
							diff = checkSimilar(cluster[0],mutant[0])
							diff_list.append(diff)

						if all(diff_list) == True:
							cm_pop.append(mutant)
			
				print('Mutations were:',muttype_list)
				
			print('len(muttype_list:',len(muttype_list))
			print('len(cm_pop)', len(cm_pop))

			mut_new_lst = []
			for mut in cm_pop:
				mut_new_lst.append(mut[0])

			mut_bag = db.from_sequence(mut_new_lst)
			mut_bag_computed = mut_bag.map(calculate)
			mut_new_lst_min = mut_bag_computed.compute()
			for o,mm in enumerate(cm_pop):
				mm[0] = mut_new_lst_min[o]

			fitnesses_mut = list(map(toolbox.evaluate1, cm_pop)) #USE DASK TO PARALLELIZE
			print('Minimazation Results') 
			print('fitnesses_mut', fitnesses_mut) 
			print('cm_pop')
			for i in range(len(cm_pop)):
				print(cm_pop[i][0])
				print(cm_pop[i][0].get_positions())

			for ind, fit in zip(cm_pop, fitnesses_mut):
				ind.fitness.values = fit

			new_population = copy.deepcopy(population)

			for cm1,cmut1 in enumerate(cm_pop):
				new_diff_list = []
				if checkBonded(cmut1[0]) == True:
					for c2,cluster1 in enumerate(population):
						diff = checkSimilar(cluster1[0],cmut1[0])
						new_diff_list.append(diff)
					if all(new_diff_list) == True:
						new_population.append(cmut1)
					else:
						pass
			print('new_population:', len(new_population))
			best_n_clus = tools.selWorst(new_population,nPool)
			population = best_n_clus

			best_clus = tools.selWorst(population,1)[0]
			print('Lowest energy individual is',best_clus)
			print(best_clus[0].get_positions())
			print('Lowest energy is',best_clus.fitness.values)
			bi.append(best_clus[0])
			write_to_db(best_db,best_clus[0])
			print('\n')
	final_pop_db = ase.db.connect("final_pop_{}.db".format(filename))
	for clus in population:
		write_to_db(final_pop_db,clus[0])

	return bi,best_clus[0]
