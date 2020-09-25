import numpy as np
import random as ran
from ase import Atoms
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii

def CoM(clus):
		'''
		Support function to set the origin of the cluster at the centre of the mass
		'''
		elems = clus.get_chemical_symbols()
		(cx, cy, cz) = clus.get_center_of_mass()
		new_xyz = []
		for i,a in enumerate(clus):
			x,y,z = a.position
			x -= cx
			y -= cy
			z -= cz
			new_xyz.append((x,y,z))
		clus.set_positions(new_xyz)
		return clus

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
	   #com = clus_to_fix.get_center_of_mass()
	   #clus_to_fix.center(about = com)
	   CoM(clus_to_fix)
	   for i in range(natoms):
		   for j in range(i):
			   r1 = np.array(clus_to_fix[j].position)
			   r2 = np.array(clus_to_fix[i].position)
			   rij = r2 - r1
			   distance = np.sqrt(np.dot(rij, rij))
			   dmin = covalent_radii[clus_to_fix[i].number] + covalent_radii[clus_to_fix[j].number]
			   if distance < 0.8 * dmin:
				   a = np.dot(r2, r2)
				   b = np.dot(r1, r2)
				   c = np.dot(r1, r1) - dmin**2
				   alpha = 1.000001 * (b + np.sqrt(b * b - a * c)) / a
				   clus_to_fix[i].x *= alpha
				   clus_to_fix[i].y *= alpha
				   clus_to_fix[i].z *= alpha
	   clus_to_fix.center(vacuum=10)

	   return clus_to_fix

def addAtoms(clusm,eleNames,eleNums,eleRadii):
        '''
        Add atom(s) to  a smaller clusters in the initial pool
        '''

        eleList = clusm.get_chemical_symbols()
        coord_xyz = []
        for i in range(len(clusm)):
                (x,y,z) = (clusm.get_positions()[i][0], clusm.get_positions()[i][1], clusm.get_positions()[i][2])
                coord_xyz.append((x,y,z))

        #eleNames, eleNums,natoms,stride,eleRadii = get_data(clusm)

        for  i in range(len(eleNames)):
                ele  = eleNames[i]
                n = 0

                for elem in eleList:
                        if ele == elem:
                                n +=1
                while n < eleNums[i]:
                        CoM(clusm)
                        rlist = []

                        for atom in range(len(clusm)):
                                x,y,z = clusm.get_positions()[atom][0], clusm.get_positions()[atom][1], clusm.get_positions()[atom][2]
                                w = np.sqrt(x*x + y*y + z*z)
                                rlist.append(w)

                        rlist.sort()
                        r = rlist[-1]
                        a = ran.uniform(0,2*np.pi)
                        b = ran.uniform(0,np.pi)
                        x = r*np.cos(a)*np.sin(b)
                        y = r*np.sin(a)*np.sin(b)
                        z = r*np.cos(b)
                        atom  = (x, y, z)
                        coord_xyz.append(atom)
                        eleList.append(ele)
                        clusm = Atoms(eleList, coord_xyz)
                        fixOverlap(clusm)
                        n += 1
        #print(clusm)
        return clusm