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

		return eleNames,eleNums,natoms,stride,eleRadii

def fixOverlap(clus):
		natoms = len(clus)
		com = clus.get_center_of_mass()
		clus.center(about = com)
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
		clus.rattle()
		del clus.constraints

		clus = fixOverlap(clus)

		return clus


def twist(clus):
		'''
		Twist the cluster
		'''
		#ROTATE ALONG X FIXING Y AND Z
		clus.rotate('y','z',center = 'COP')

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