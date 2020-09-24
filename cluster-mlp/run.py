from deap_main import cluster_GA
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.calculators.emt import EMT
from ase.visualize import view

eleNames = ['Cu', 'Al']
eleNums = [3, 5]
nPool = 100
generations = 1000
eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
calc = EMT()

final_cluster = cluster_GA(nPool,eleNames,eleNums,eleRadii,generations,calc)
view(final_cluster)