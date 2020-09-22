

def CoM(clus):
	'''
	Support function to set the origin of the cluster at the centre of the mass
	'''
        elems = clus.get_chemical_symbols()
        (cx, cy, cz) = clus.get_center_of_mass()

        new_xyz = []
        for i in range(len(clus)):
                (x,y,z) = (clus.get_positions()[i][0], clus.get_positions()[i][1], clus.get_positions()[i][2])
                x -= cx
                y -= cy
                z -= cz
                new_xyz.append((x,y,z))
        clus.set_positions(new_xyz)

        return clus


def addAtoms_GIGA(clusm):
        '''
        Add atom(s) to  a smaller clusters in the initial pool
        '''
       
        eleList = clusm.get_chemical_symbols()
        coord_xyz = []
        for i in range(len(clusm)):
                (x,y,z) = (clusm.get_positions()[i][0], clusm.get_positions()[i][1], clusm.get_positions()[i][2])
                coord_xyz.append((x,y,z))
        
        eleNames, eleNums,natoms,stride,eleRadii = get_data(clusm) 
        
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
