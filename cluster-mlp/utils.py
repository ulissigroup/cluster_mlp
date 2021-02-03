import numpy as np
import random as ran
from ase import Atoms
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.build import sort
def write_to_db(database,image):
	image.get_potential_energy()
	database.write(image,relaxed = True)

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
		   dmin = (covalent_radii[clus_to_fix[i].number] + covalent_radii[clus_to_fix[j].number])*0.7
		   if distance < 0.8 * dmin:
			   a = np.dot(r2, r2)
			   b = np.dot(r1, r2)
			   c = np.dot(r1, r1) - dmin**2
			   alpha = 1.000001 * (b + np.sqrt(b * b - a * c)) / a
			   clus_to_fix[i].x *= alpha
			   clus_to_fix[i].y *= alpha
			   clus_to_fix[i].z *= alpha
   clus_to_fix.center(vacuum=9)
   clus_to_fix_sorted = sort(clus_to_fix)
   clus_to_fix_sorted.pbc = (True, True, True)
   return clus_to_fix_sorted

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
                        clusm = fixOverlap(clusm)
                        n += 1
        #print(clusm)
        return clusm

def checkBonded(clus):
	'''
	Check if every atom of the cluster is bonded to other
	'''
	natoms = len(clus)
	ele_list = clus.get_chemical_symbols()
	radList = [covalent_radii[atomic_numbers[ele]] for ele in ele_list]
	bonded = True

	for i in range(natoms):
		checkList = []
		for j in range(natoms):
			if i != j:
				x = clus.get_positions()[j][0] - clus.get_positions()[i][0]
				y = clus.get_positions()[j][1] - clus.get_positions()[i][1]
				z = clus.get_positions()[j][2] - clus.get_positions()[i][2]
				dij = np.sqrt(x**2 +y**2 +z**2)
				dmin = radList[i] + radList[j]
				if dij > 1.3*dmin:
					checkList.append(False)
				else:
					checkList.append(True)
		if True not in checkList:
			bonded = False
	return bonded


def checkSimilar(clus1,clus2):

	'''Check whether two clusters are similar or not by comparing their moments of inertia'''
	Inertia1=clus1.get_moments_of_inertia()
	Inertia2=clus2.get_moments_of_inertia()
	#print(Inertia1, Inertia2, 'diff: ', Inertia1-Inertia2)

	tol = 0.01
	if Inertia1[0]*(1-tol) <= Inertia2[0] <= Inertia1[0]*(1+tol) and Inertia1[1]*(1-tol) <= Inertia2[1] <= Inertia1[1]*(1+tol) and Inertia1[2]*(1-tol) <= Inertia2[2] <= Inertia1[2]*(1+tol):
		differ = False
	else:
		differ = True

	return differ

def sortR0(clus,R0):
                '''
                Sort the atom list according to their distance to R0
                '''
                w = []
                natoms = len(clus)
                for atom in clus:
                        ele = atom.symbol
                        x, y, z = atom.position
                        dx = x - R0[0]
                        dy = y - R0[1]
                        dz = z - R0[2]
                        dr = np.sqrt(dx**2 +dy**2 +dz**2)
                        w.append([dr,ele,x,y,z])

                w.sort()
                ele = [w[i][1] for i in range(natoms)]
                coord_xyz = [ (w[i][2], w[i][3], w[i][4]) for i in range(natoms)]
                clus = Atoms(ele,coord_xyz)
                return clus
