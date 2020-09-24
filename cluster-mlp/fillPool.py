from utils import addAtoms
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator as sp

def fillPool(eleNames,eleNums,eleRadii,calc):
	'''
	Fill Initial pool with random geometries
	'''
	#ini_pool = []
	ele_initial = [eleNames[0], eleNames[-1]]
	d = (eleRadii[0] + eleRadii[-1])/2
	clusm = Atoms(ele_initial, [(-d, 0.0, 0.0), (d, 0.0, 0.0)])
	clus = addAtoms(clusm,eleNames,eleNums,eleRadii )
	clus.calc = calc
	dyn = BFGS(clus,logfile = None)
	dyn.run(fmax = 0.05,steps = 100)
	energy = clus.get_potential_energy()
	clus.set_calculator(sp(atoms=clus, energy=energy))
            #vaspIN(self.calcNum,clus,self.vac,self.surfGA,self.clusHeight)
            #nmut = 0
            #ibond = 0
            #relax()
            #print(clus)
		#ini_pool.append(clus)
            #print(ini_pool)
	return  clus

