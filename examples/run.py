from deap_main_dask import cluster_GA
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp2, Vasp
from ase.visualize import view
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import functools

if __name__ == '__main__':
	eleNames = ['Cu', 'Al']
	eleNums = [4,3]
	nPool = 10
	generations = 40
	CXPB = 0.7
	eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
	filename = 'clus_Cu4Al3' #For saving the best cluster at every generation
	log_file = 'clus_Cu4Al3.log' 
	singleTypeCluster = False

	#calc = EMT()
	calc = Vasp2(kpar=1,
             ncore=4,
             encut=400,
             xc='PBE',
             #gga='PS',
             kpts=(1,1,1),
             gamma = True,# Gamma-centered
             ismear=1,
             sigma=0.2,
             ibrion=2,
             nsw=1000,
             #lorbit=11,
             potim=0.2,
             isif=0,
             #ediffg=-0.02,
             #ediff=1e-6,
             lcharg=False,
             lwave=False,
             lreal=False,
             ispin=2,
             isym=0)

	# Run between 0 and 4 1-core/1-gpu workers on the kube cluster
	cluster = KubeCluster.from_yaml('worker-cpu-spec.yml')
	client = Client(cluster)
	#cluster.adapt(minimum=0, maximum=10)
	cluster.scale(10)


	files_list = ['deap_main_dask.py', 'fillPool.py', 'mutations.py', 'utils.py']
	for i in range(len(files_list)):
		fname = files_list[i]
		with open(fname, 'rb') as f:
  			data = f.read()

		def _worker_upload(dask_worker, *, data, fname):
  			dask_worker.loop.add_callback(
    		callback=dask_worker.upload_file,
    		comm=None,  # not used
    		filename=fname,
    		data=data,
    		load=True)

		client.register_worker_callbacks(
  			setup=functools.partial(
    			_worker_upload, data=data, fname=fname,
  			)
		)


	bi,final_cluster = cluster_GA(nPool,eleNames,eleNums,eleRadii,generations,calc,filename,log_file,CXPB, singleTypeCluster)
	#view(final_cluster)
	#view(bi[0])

