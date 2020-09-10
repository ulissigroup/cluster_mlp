import numpy as np
import ase
from ase import Atoms
import random as ran
from ase.constraints import FixAtoms
from ase.data import atomic_numbers,vdw_radii
from xtb.ase.calculator import XTB
from ase.calculators.emt import EMT
from ase.visualize import view
from ase.optimize import BFGS

def get_data(cluster):
		eleNames = list(set(cluster.get_chemical_symbols()))
		eleNums = []
		for i,name in enumerate(eleNames):
			ele_count = 0
			for a in cluster.get_chemical_symbols():
				if a == eleNames[i]:
					ele_count += 1
			eleNums.append(ele_count)
		natoms = len(cluster)
		stride = natoms + 2
		eleRadii = []
		atomic_numbers_list = []
		for i in range(len(eleNames)):
			atomic_num = atomic_numbers[eleNames[i]]
			atomic_numbers_list.append(atomic_num)
			eleR = vdw_radii[atomic_num]
			eleRadii.append(eleR)

		return eleNames,eleNums,natoms,stride,eleRadii

'''
def fixOverlap(clus):
	   natoms = len(clus)
	   #com = clus.get_center_of_mass()
	   #clus.center(about = com)
	   for i in range(natoms):
		   for j in range(i):
			   r1 = np.array(clus[j].position)
			   r2 = np.array(clus[i].position)
			   rij = r2 - r1
			   distance = np.sqrt(np.dot(rij, rij))
			   dmin = vdw_radii[clus[i].number] + vdw_radii[clus[j].number]
			   if distance < 0.9 * dmin:
				   a = np.dot(r2, r2)
				   b = np.dot(r1, r2)
				   c = np.dot(r1, r1) - dmin**2
				   alpha = 1.000001 * (b + np.sqrt(b * b - a * c)) / a
				   clus[i].x *= alpha
				   clus[i].y *= alpha
				   clus[i].z *= alpha
				   #print(distance,dmin)
	   return clus
'''

def fixOverlap(clus):
	clus.set_calculator(XTB(method="GFN0-xTB"))
	dyn = BFGS(clus,fmax = 0.05,steps = 25,logfile = None)
	dyn.relax()
	return clus

def add_atoms(clusm,atcenter):
		'''
		Add atom(s) to a smaller cluster given in poolm.dat
		'''
		eleList = []
		rlist = []
		for atom in clusm:
			ele = atom.symbol
			x = atom.x
			y = atom.y
			z = atom.z
			w = np.sqrt(x*x + y*y + z*z)
			rlist.append(w)
			eleList.append(ele)

		r = max(rlist) + 0.5
		print(rlist)
		eleNames,eleNums,natoms,stride,eleRadii = get_data(clusm)

		for i in range(len(eleNames)):
				ele = eleNames[i]
				n = 0
				for elem in eleList:
					if ele == elem:
						n += 1
				while n < eleNums[i]:
					if atcenter:
						added_atom = Atoms(ele,positions = [[0.0,0.0,0.0]])
						atcenter = False
					else:
						a = ran.uniform(0,2*np.pi)
						b = ran.uniform(0,np.pi)
						x = r*np.cos(a)*np.sin(b)
						y = r*np.sin(a)*np.sin(b)
						z = r*np.cos(b)
						added_atom = Atoms(ele,positions = [[x,y,z]])
						#print(x,y,z)
						print(added_atom)

					clusm += added_atom
					n += 1
		print(clusm.positions)
		view(clusm)
		clus = fixOverlap(clusm)
		return clus

def homotop(clus): #DONE
		'''
		Choose pair of different elements to swap
		'''
		eleNames,eleNums,natoms,stride,eleRadii = get_data(clus)
		eles = ran.sample(eleNames,2)
		ele1_index = []
		ele2_index = []
		for i in clus:
			if i.symbol == eles[0]:
				ele1_index.append(i.index)
			if i.symbol == eles[1]:
				ele2_index.append(i.index)

		ele1_position = ran.choice(ele1_index)
		ele2_position = ran.choice(ele2_index)
		clus.positions[[ele1_position,ele2_position]] = clus.positions[[ele2_position,ele1_position]]
		return clus


def rattle_mut(clus): #DONE
		'''
		Choose one atom and move 1/4 of the cluster atoms
		nearest to that atom a distance between 0 and
		+/- its radius
		'''
		#CHANGED TO FIX A THIRD OF ATOMS AND RATTLE REST
		indices = ran.sample(range(len(clus)), int(len(clus)/3))
		const = FixAtoms(indices = indices)
		clus.set_constraint(const)
		clus.rattle(stdev=0.1)
		del clus.constraints
		view(clus)
		clus = fixOverlap(clus)
		return clus


def twist(clus):
		'''
		Twist the cluster
		'''
		#ROTATE ALONG X FIXING Y AND Z
		clus.rotate('y','z',center = 'COP')
		print(clus.positions)
		view(clus)
		clus = fixOverlap(clus)
		return clus


def tunnel(clus):
		'''
		Tunnel one of the atoms farthest from the center to
		the other side of the cluster
		'''
		center = clus.get_center_of_mass()
		distances = []
		for atom in clus:
			distance = np.sqrt(np.dot(center - atom.position))
			distances.append(distance)

		distances = np.array(distances)
		max_index = np.argmax(distances)
		x = clus[max_index].x
		y = clus[max_index].y
		z = clus[max_index].z
		clus.positions[max_index] = (-x,-y,-z)

		clus = fixOverlap(clus)
		return clus


def rotate_mut(clus):
		'''
		Two modes for rotate mutation

		1 - If complete, rotate entire cluster
		2 - If incomplete, rotate 25% of cluster atoms
		'''
		#MODIFIED TO ALWAYS ROTATE FULL CLUSTER
		angle = ran.randint(1,180)
		axis = ran.choice(['x','y','z'])
		clus.rotate(angle,axis,center = 'COP')
		clus = fixOverlap(clus)

		return clus

def partialInversion(clus):
		'''
		Choose a fragment with 30% of the cluster atoms
		nearest to a randomly chosen atom and invert the
		structure with respect to its geometrical center
		'''
		selected_atom = ran.choice(clus)
		distances = clus.get_distances(selected_atom.index,np.arange(0,len(clus)))
		ncount = int(0.3*len(clus))
		idx = []
		idx.append(np.argsort(distances)[-ncount:])
		for i in idx:
			x = clus[idx].x
			y = clus[idx].y
			z = clus[idx].z
			clus.positions[idx] = (-x,-y,-z)

		clus = fixOverlap(clus)
		return clus

def mate(clus1,clus2,fit1,fit2,surfGA = False):
		"""
		1. Select a pair of clusters from pool
		   using roulette-wheel selection.
		2. If gas-phase, rotate randomly the clusters.
		3. Weighted cut of the clusters in a plane
		   perpendicular to the surface.
		4. Join parts and repare overlaps.
		"""
		#ROULETTE WHEEL THROUGH DEAP
		compositionWrong = True
		parent1 = clus1.copy()
		parent2 = clus2.copy()
		while compositionWrong:
			if surfGA == False:
				clus1 = rotate_mut(clus1)
				clus2 = rotate_mut(clus2)

			child = []
			eleNames,eleNums,natoms,_,_ = get_data(clus1)
			cut = int(natoms*(fit1/(fit1+fit2)))

			if cut == 0:
				cut = 1
			elif cut == natoms:
				cut = natoms - 1

			for i in range(cut):
				child.append(clus1[i])

			for j in range(cut,len(clus2)):
				child.append(clus2[j])

			CheckEle = []
			for ele in eleNames:
				eleCount = 0
				for atom in child:
					if atom.symbol == ele:
						eleCount += 1
				CheckEle.append(eleCount)

			if CheckEle == eleNums:
				compositionWrong = False

		final_child = Atoms(child[0].symbol,positions = [child[0].position])
		for i in range(1,len(child)):
			c = Atoms(child[i].symbol,positions = [child[i].position])
			final_child += c

		fixOverlap(final_child)
		return final_child, parent1,parent2

#TESTING
'''
testclus1 = Atoms('Ni3Au2',positions = [[7.11429776,8.18434723,8.39830404],
						   [5.95722660,7.50727631,6.42107119],
						   [8.27136892,7.50727631,6.42107119],
						   [8.27919396,6.16668782,8.39830404],
						   [5.94940156,6.16668782,8.39830404]], cell = [10.0, 10.0, 10.0],pbc = True)
testclus2 = Atoms('Ni3Au2',positions = [[6.11429776,4.18434723,8.39830404],
						   [4.95722660,7.50727631,6.42107119],
						   [7.27136892,4.50727631,6.42107119],
						   [9.27919396,6.16668782,6.39830404],
						   [6.94940156,6.16668782,8.39830404]], cell = [10.0, 10.0, 10.0],pbc = True)
view(testclus1)
view(testclus2)
testclus1.set_calculator(EMT())
testclus2.set_calculator(EMT())
a = testclus1.get_potential_energy()
b = testclus2.get_potential_energy()
print(a,b)
new_clus,parent1,parent2 = mate(testclus1,testclus2,a,b,False)
print(new_clus.positions)
view(new_clus)'''