

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


