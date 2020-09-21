import numpy as np
import ase
from ase import Atoms
import random as ran
from ase.constraints import FixAtoms
from ase.data import atomic_numbers,covalent_radii

def get_data(cluster):
		'''
		Support function to get required data from an ase atoms object
		'''
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
			eleR = covalent_radii[atomic_num]
			eleRadii.append(eleR)

		return eleNames,eleNums,natoms,stride,eleRadii


def fixOverlap(clus_to_fix):
	   '''
	   Support function to fix any overlaps that may arise due to the mutations by radially moving the atoms that have overlap
	   '''
	   natoms = len(clus_to_fix)
	   com = clus_to_fix.get_center_of_mass()
	   clus_to_fix.center(about = com)
	   for i in range(natoms):
		   for j in range(i):
			   r1 = np.array(clus_to_fix[j].position)
			   r2 = np.array(clus_to_fix[i].position)
			   rij = r2 - r1
			   distance = np.sqrt(np.dot(rij, rij))
			   dmin = covalent_radii[clus_to_fix[i].number] + covalent_radii[clus_to_fix[j].number]
			   if distance < 0.9 * dmin:
				   a = np.dot(r2, r2)
				   b = np.dot(r1, r2)
				   c = np.dot(r1, r1) - dmin**2
				   alpha = 1.000001 * (b + np.sqrt(b * b - a * c)) / a
				   clus_to_fix[i].x *= alpha
				   clus_to_fix[i].y *= alpha
				   clus_to_fix[i].z *= alpha
	   return clus_to_fix

def add_atoms(clusm,atcenter):
		'''
		Add atom(s) to a smaller cluster
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
					clusm += added_atom
					n += 1
		clus = fixOverlap(clusm)
		return clus

def homotop(parent):
		'''
		Choose pair of different elements to swap
		'''
		clus = parent.copy()
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


def rattle_mut(parent): #DONE
		'''
		Fix a third of the atoms, then rattle the rest with a std deviation of 0.01
		'''
		clus = parent.copy()
		indices = ran.sample(range(len(clus)), int(len(clus)/3))
		const = FixAtoms(indices = indices)
		clus.set_constraint(const)
		clus.rattle(stdev=0.1)
		del clus.constraints
		clus = fixOverlap(clus)
		return clus


def twist(parent):
		'''
		Twist the cluster
		'''
		clus = parent.copy()
		clus.rotate('y','z',center = 'COP')
		clus = fixOverlap(clus)
		return clus


def tunnel(parent):
		'''
		Tunnel one of the atoms farthest from the center to
		the other side of the cluster
		'''
		clus = parent.copy()
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


def rotate_mut(parent):
		'''
		Rotate the cluster by a randomly selected angle over a randomly selected axis
		'''
		clus = parent.copy()
		angle = ran.randint(1,180)
		axis = ran.choice(['x','y','z'])
		clus.rotate(angle,axis,center = 'COP')
		clus = fixOverlap(clus)

		return clus

def partialInversion(parent):
		'''
		Choose a fragment with 30% of the cluster atoms
		nearest to a randomly chosen atom and invert the
		structure with respect to its geometrical center
		'''
		clus = parent.copy()
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

def mate(parent1,parent2,fit1,fit2,surfGA = False):
		"""
		Randomly selected clusters from roulette wheel selection are passed:

		1. If gas-phase, rotate randomly the clusters.
		2. Weighted cut of the clusters in a plane
		   perpendicular to the surface.
		3. Join parts and repare overlaps.
		"""

		compositionWrong = True
		clus1 = parent1.copy()
		clus2 = parent2.copy()
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