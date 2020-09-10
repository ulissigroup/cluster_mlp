def fillPool(nPool):
	'''
	Fill Inition pool with random geometries
        ini_pool = []
	'''
        nc = 0
        while nc  < nPool:
            ele_initial = [eleNames[0], eleNames[-1]]
            d = (eleRadii[0] + eleRadii[-1])/2
            clusm = Atoms(ele_initial, [(-d, 0.0, 0.0), (d, 0.0, 0.0)])
            clus = addAtoms_GIGA(clusm )
            #vaspIN(self.calcNum,clus,self.vac,self.surfGA,self.clusHeight)
            #nmut = 0
            #ibond = 0
            #relax()
            #print(clus)
            ini_pool.append(clus)
            #print(ini_pool)
            nc += 1
        return  ini_pool
