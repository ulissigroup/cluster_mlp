import numpy as np
from utils import CoM
from mutations import fixOverlap, get_data
import ase
import mutations
from ase import Atoms
import random as ran

def fillPool(nPool,eleNames,eleNums,eleRadii):
	'''
	Fill Inition pool with random geometries
	'''
	ini_pool = []
	nc = 0
	while nc  < nPool:
		ele_initial = [eleNames[0], eleNames[-1]]
		d = (eleRadii[0] + eleRadii[-1])/2
		clusm = Atoms(ele_initial, [(-d, 0.0, 0.0), (d, 0.0, 0.0)])
		clus = addAtoms_GIGA(clusm,eleNames,eleNums,eleRadii )
            #vaspIN(self.calcNum,clus,self.vac,self.surfGA,self.clusHeight)
            #nmut = 0
            #ibond = 0
            #relax()
            #print(clus)
		ini_pool.append(clus)
            #print(ini_pool)
		nc += 1
	return  ini_pool


def addAtoms_GIGA(clusm,eleNames,eleNums,eleRadii):
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
