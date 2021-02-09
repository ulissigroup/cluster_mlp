from utils import addAtoms
from ase import Atoms
from utils import fixOverlap

def fillPool(eleNames,eleNums,eleRadii,calc):
	'''
	Fill Initial pool with random geometries
	'''
	ele_initial = [eleNames[0], eleNames[-1]]
	d = (eleRadii[0] + eleRadii[-1])/2
	clusm = Atoms(ele_initial, [(-d, 0.0, 0.0), (d, 0.0, 0.0)])
	clus = addAtoms(clusm,eleNames,eleNums,eleRadii )
	return  clus

