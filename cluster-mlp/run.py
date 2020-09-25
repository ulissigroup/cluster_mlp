from deap_main import cluster_GA
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.calculators.emt import EMT
from ase.visualize import view

eleNames = ['Cu', 'Al']
eleNums = [3, 5]
nPool = 100
generations = 100
CXPB = 0.5
MUTPB = 0.2
eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
calc = EMT()

bi,final_cluster = cluster_GA(nPool,eleNames,eleNums,eleRadii,generations,calc,CXPB,MUTPB)
view(final_cluster)
view(bi[0])
