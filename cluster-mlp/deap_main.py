import random
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.visualize import view
from deap import base
from deap import creator
from deap import tools
from ase import Atoms
from fillPool import fillPool
from mutations import homotop,rattle_mut,rotate_mut,twist,tunnel,partialInversion,mate

eleNames = ['Cu', 'Al']
eleNums = [3, 5]
nPool = 100
eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
calc = EMT()

def fitness_func1(individual):
	atoms = individual[0]
	energy = atoms.get_potential_energy()
	return -energy,

def fitness_func2(individual,calc=calc):
	atoms = individual[0]
	atoms.set_calculator(calc)
	dyn = BFGS(atoms,logfile = None)
	dyn.run(fmax = 0.05)
	energy = atoms.get_potential_energy()
	return -energy,

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

toolbox.register("select", tools.selTournament)

def main():
	CXPB, MUTPB = 0.5, 0.2
	pop = toolbox.population(n=nPool)

	fitnesses = list(map(toolbox.evaluate1, pop))
	for ind, fit in zip(pop, fitnesses):
	        ind.fitness.values = fit
	g = 0
	fits = [ind.fitness.values[0] for ind in pop]
	while g < 1000:
			g = g + 1
			print('Generation',g)
			print('Starting Evolution')
			offspring = toolbox.select(pop,2,10)
			if random.random() < CXPB:
				parent1, = offspring[0]
				parent2, = offspring[1]
				fit1, = offspring[0].fitness.values
				fit2, = offspring[1].fitness.values
				toolbox.mate(parent1,parent2,fit1,fit2)
				del offspring[0].fitness.values
				del offspring[1].fitness.values
			for mutant in offspring:
				if random.random() < MUTPB:
					mutType = random.choice(['homotop','rattle','rotate','twist','tunnel','partialinv'])
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
				#CHECK IF REQUIRED
				new_fitness = fitness_func2(mutant)
				if new_fitness < mutant.fitness.values:
					del mutant.fitness.values
					mutant.fitness.values = new_fitness
				#ENDS
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluate2,invalid_ind)
			for ind,fit in zip(invalid_ind,fitnesses):
				ind.fitness.values = fit

			print('End of evolution')
			best_ind = tools.selWorst(pop,1)[0]
			print('Best individual is',best_ind)
			print('Best individual fitness is',best_ind.fitness.values)

if __name__ == '__main__':
	main()