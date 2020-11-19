import numpy as np
from ase import Atoms
import random as ran
from ase.constraints import FixAtoms
from utils import CoM,get_data,fixOverlap,addAtoms,sortR0

def homotop(parent):
	'''
	Choose pair of different elements to swap
	'''
	clus = parent
	CoM(clus)
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

def rattle_mut(parent):
	'''
	Fix a third of the atoms, then rattle the rest with a std deviation of 0.01
	'''
	clus = parent
	CoM(clus)
	indices = ran.sample(range(len(clus)), int(len(clus)/3))
	const = FixAtoms(indices = indices)
	clus.set_constraint(const)
	clus.rattle(stdev=0.1)
	del clus.constraints
	print(clus.positions)
	clus = fixOverlap(clus)
	print(clus.positions)
	return clus


def twist(parent):
	'''
	Twist the cluster
	'''
	clus = parent
	CoM(clus)
	clus.rotate('y','z',center = 'COM')
	clus = fixOverlap(clus)
	return clus

def rotate_mut(parent):
	'''
	Rotate the cluster by a randomly selected angle over a randomly selected axis
	'''
	clus = parent
	CoM(clus)
	angle = ran.randint(1,180)
	axis = ran.choice(['x','y','z'])
	clus.rotate(angle,axis,center = 'COM')
	clus = fixOverlap(clus)

	return clus


def partialInversion(parent):
                '''
                Choose a fragment with 30% of the cluster atoms
                nearest to a randomly chosen atom and invert the
                structure with respect to its geometrical center
                '''
                clus = parent
                natoms = len(clus)
                CoM(clus)
                nInvert = int(round(0.3*natoms))
                mAtom = ran.randrange(natoms)
                R0 = clus.get_positions()[mAtom]
                clus = sortR0(clus,R0)

                fc = np.array([0.0,0.0,0.0])

                for i in range(nInvert):
                        r = clus.get_positions()[i]
                        fc += r/nInvert
                new_pos = []
                for i in range(nInvert):
                        ele,x,y,z = clus.get_chemical_symbols()[i],  clus.get_positions()[i][0], clus.get_positions()[i][1], clus.get_positions()[i][2]
                        r = np.array([x,y,z])
                        ri = 2*fc - r
                        clus[i].x = ri[0]
                        clus[i].y = ri[1]
                        clus[i].z = ri[2]
                        new_coord = (ri[0], ri[1], ri[2])

                clus = fixOverlap(clus)
                return clus

def tunnel(parent):
                '''
                Tunnel one of the atoms farthest from the center to
                the other side of the cluster
                '''
                clus = parent
                natoms = len(clus)
                CoM(clus)
                w = []
                for atom in clus:
                        ele = atom.symbol
                        x,y,z = atom.position
                        r = np.sqrt(x*x + y*y + z*z)
                        w.append([r,ele,x,y,z])
                w.sort()
                for i in range(natoms):
                        clus[i].symbol = w[i][1]
                        clus[i].x = w[i][2]
                        clus[i].y = w[i][3]
                        clus[i].z = w[i][4]

                nat = int(round(0.75*natoms))
                atomNum = ran.randrange(nat,natoms)
                x,y,z = clus[atomNum].x, clus[atomNum].y, clus[atomNum].z
                clus[atomNum].x, clus[atomNum].y, clus[atomNum].z = -x,-y,-z

                clus = fixOverlap(clus)
                return clus

def skin(parent):
                '''
                Keep 80% of the cluster atoms and relocate the remaining
                '''
                clus = parent
                eleNames,eleNums,natoms,stride,eleRadii = get_data(clus)
                CoM(clus)
                nfix = int(round(0.8*natoms))
                R0 = [0.0,0.0,0.0]
                clus = sortR0(clus,R0)
                core_pos = []
                core_ele = []
                for i in range(nfix):
                        x,y,z = clus[i].position
                        core_pos.append((x,y,z))
                        ele = clus[i].symbol
                        core_ele.append(ele)
                core = Atoms(core_ele, core_pos)
                clus = addAtoms(core,eleNames,eleNums,eleRadii)

                return clus

def changeCore(parent):
                '''
                Modify the core
                '''
                clus = parent
                eleNames,eleNums,natoms,stride,eleRadii = get_data(clus)
                CoM(clus)
                inout = ran.choice([1,2]) #inout = 1 muttpe: + core; inout = 2 muttype: - core
                if inout == 1:
                        nout = int(0.2*natoms)
                        if nout < 1:
                                nout = 1
                        icenter = ran.randrange(nout) + 1
                        R0 = [0.0,0.0,0.0]
                        clus = sortR0(clus,R0)
                        clus[-icenter].position  = [0.1, 0.0, 0.0]
                        clus = fixOverlap(clus)

                elif inout == 2:
                        ncore = int(0.1*natoms)
                        if ncore < 1:
                                ncore = 1
                        iout = ran.randrange(ncore)
                        R0 = [0.0,0.0,0.0]
                        clus = sortR0(clus,R0)
                        del clus[iout]
                        clus = addAtoms(clus,eleNames,eleNums,eleRadii)

                return clus



def mate(parent1,parent2,fit1,fit2,surfGA = False):
	"""
	Randomly selected clusters from tournament selection are passed:
	1. If gas-phase, rotate randomly the clusters.
	2. Weighted cut of the clusters in a plane
	   perpendicular to the surface.
	3. Join parts and repare overlaps.
	"""

	compositionWrong = True
	clus1 = parent1
	clus2 = parent2
	while compositionWrong:
		if surfGA == False:
			angle = ran.randint(1,180)
			axis = ran.choice(['x','y','z'])
			clus1.rotate(angle,axis,center = 'COM')
			clus1 = fixOverlap(clus1)
			clus2.rotate(angle,axis,center = 'COM')
			clus2 = fixOverlap(clus2)

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

	final_child = fixOverlap(final_child)
	parent1 = final_child
	parent2 = parent2
	return [parent1,parent2]