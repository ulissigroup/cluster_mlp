from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from fillPool import fillPool
from utils import ase2list

eleNames = ['Cu', 'Ni']
eleNums = [3, 5]
nPool = 25
mutRate = 0.001
mutTypes = ['Homotop','Core','Skin','Rattle','Twist','Tunnel','Invert','Rotate']
mutProb = [1, 1, 1, 1, 1, 1, 1, 1]
tripleProb = 0.5
ngen = 100
diverCheck = 3
vac = 9.0
surfGA = False

eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]


ini_pool = fillPool(nPool,eleNames,eleNums,eleRadii)

for item in ini_pool:
	print(item.get_positions())

for clus in ini_pool:
	clus_list = ase2list(clus)
	for item in clus_list:
		print(item[0], item[1], item[2], item[3])
	print('\n')

#StartCalc = poolGA(eleNames,eleNums,eleRadii,nPool,mutRate,mutTypes,mutProb,tripleProb, ngen, subString,diverCheck,vac,surfGA)
