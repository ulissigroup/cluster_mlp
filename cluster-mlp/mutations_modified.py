import numpy as np
import ase
from ase import Atoms
import random as ran
import database as db
import prepare as pre
import checkPool as chk
from ase.constraints import FixAtoms
from ase.data import atomic_numbers,vdw_radii

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

	eleRadii = eleRadii

	return eleNames,eleNums,natoms,stride,eleRadii

class mutations:
	def __init__(self,eleNames,eleNums,natoms,stride,eleRadii,mutType):
		self.eleNames = eleNames
		self.eleNums = eleNums
		self.natoms = sum(eleNums)
		self.stride = self.natoms + 2
		self.eleRadii = eleRadii
		self.nPool = nPool
		self.mutType = mutType
		self.surfGA = surfGA

		ave = 0.0
		for i in range(len(eleNums)):
			for j in range(eleNums[i]):
				ave += eleRadii[i] / self.natoms
		self.radave = ave


	def add_atoms(self,clusm,atcenter):
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

		for i in range(len(self.eleNames)):
				ele = self.eleNames[i]
				n = 0
				for elem in eleList:
					if ele == elem:
						n += 1

				while n < self.eleNums[i]:
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
		clus = pre.fixOverlap(clusm,self.eleNames,self.eleRadii)
		return clus
	def mutate(self): #BASE DONE, NOT YET FULLY FIXED
			pair,fitpair,parents = self.choosePair()
			clus = pair[0]
			parent = parents[0]
			if self.mutType == 'All':
				mType = ran.choice(['Rattle','Twist','Tunnel','Rotate'])
			else:
				mType = self.mutType

			if len(self.eleNames) == 1:
				mono = True
			else:
				mono = False

			if not mono and mType == 'Homotop':
				mutant = self.homotop(clus)

			elif mType == 'Rattle':
				mutant = self.rattle_mut(clus)

			elif mType == 'Twist':
				mutant = self.twist(clus)

			elif mType == 'Tunnel':
				mutant = self.tunnel(clus)

			elif mType == 'Rotate':
				mutant = self.rotate_mut(clus,self.surfGA)

			else:
				mutant = clus

			return mutant, parent, mType

	def homotop(self,clus): #DONE
			'''
			Choose pair of different elements to swap
			'''
			eles = ran.sample(self.eleNames,2)
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


	def rattle_mut(self,clus): #DONE
			'''
			Choose one atom and move 1/4 of the cluster atoms
			nearest to that atom a distance between 0 and
			+/- its radius
			'''
			#CHANGED TO FIX A THIRD OF ATOMS AND RATTLE REST
			indices = ran.sample(range(len(clus)), int(len(clus)/3))
			const = FixAtoms(indices = indices)
			clus.set_constraint(const)
			clus.rattle()
			del clus.constraints

			pre.fixOverlap(clus,self.eleNames,self.eleRadii)

			return clus


	def twist(self,clus):
			'''
			Twist the cluster
			'''
			#ROTATE ALONG X FIXING Y AND Z
			clus.rotate('y','z',center = 'COP')

			pre.fixOverlap(clus,self.eleNames,self.eleRadii)
			return clus


	def tunnel(self,clus):
			'''
			Tunnel one of the atoms farthest from the center to
			the other side of the cluster
			'''
			#FIND THE DISTANCES OF ALL ATOMS TO EACH OTHER,FIND 2 ATOMS WITH THE GREATEST DISTANCE
			#AND TUNNEL ONE OF THEM TO THE OPPOSITE SIDE
			distances = clus.get_all_distances()
			max_indices = np.unravel_index(distances.argmax(),distances.shape)
			atom_index = ran.choice(max_indices)
			x = clus[atom_index].x
			y = clus[atom_index].y
			z = clus[atom_index].z
			clus.positions[atom_index] = (-x,-y,-z)

			#OR

			for i in range(len(clus)):
				flag = np.isclose(clus.get_center_of_mass(),clus[i].position,atol = 1e-1,rtol = 1e-1) #TRICKY NOT SURE IF IT WILL WORK
				check = (flag[0] and flag[1]) or flag[2]
				if check == True:
					center = clus[i].position
					break
			distances = clus.get_distances(center,indices = np.array([np.arange(0,len(clus))]))
			max_index = np.argmax(distances)
			x = clus[max_index].x
			y = clus[max_index].y
			z = clus[max_index].z
			clus.positions[max_index] = (-x,-y,-z)

			pre.fixOverlap(clus,self.eleNames,self.eleRadii)
			return clus


	def rotate_mut(self,clus):
			'''
			Two modes for rotate mutation

			1 - If complete, rotate entire cluster
			2 - If incomplete, rotate 25% of cluster atoms
			'''
			#MODIFIED TO ALWAYS ROTATE FULL CLUSTER
			angle = ran.randint(1,180)
			axis = ran.choice(['x','y','z'])
			clus.rotate(angle,axis,center = 'COP')
			pre.fixOverlap(clus,self.eleNames,self.eleRadii)

			return clus


