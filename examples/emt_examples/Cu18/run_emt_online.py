from cluster_mlp.clus_ga_deap import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.vasp import Vasp
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch
from ase.optimize import BFGS
from ase.calculators.emt import EMT
#from ase.calculators.vasp import Vasp

if __name__ == "__main__":
    use_dask = False #Launch Dask
    eleNames = [ "Cu"] #element list in the cluster
    eleNums = [18] #element composition
    nPool = 10 #number of clusters in the initial pool (population)
    generations = 3 # number of generations
    CXPB = 0.5 #cross-over probability; 1-CXPB is the mutation probability 
    use_vasp = False # use vasp for VASP DFT calculations
    use_vasp_inter = False # vasp_interative, not recommended
    al_method = "Online" # active learning (AL-GA), if you want DFT-GA, use al_method = None
    optimizer = BFGS
    restart = False # if you want to restart from a generation use True, otherwise False
    gen_num = 16 #if restart=True, give the generation number to restart
    calc = EMT()# ASE calculator, we have tested EMT() and VASP.
    
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    comp_list = [ eleNames[i]+str(eleNums[i])  for i in range(len(eleNames))]
    filename='clus_'+''.join(comp_list)# For saving the best cluster at every generation
    log_file = filename+".log"
    if len(eleNames) == 1:
        singleTypeCluster = True
    else:
        singleTypeCluster = False
    
    if use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(nPool)

    learner_params = {
        "filename": "relax_example",
        "file_dir": "./",
        "stat_uncertain_tol": 0.08,
        "dyn_uncertain_tol": 0.1,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "reverify_with_parent": False,
        "suppress_warnings": True
    }
    train_config  = {
        "sigma": 4.5,
        "power": 2,
        "cutoff_function": "quadratic",
        "cutoff": 5.0,
        "radial_basis": "chebyshev",
        "cutoff_hyps": [],
        "sigma_e": 0.009,
        "sigma_f": 0.005,
        "sigma_s": 0.0006,
        "hpo_max_iterations": 50,
        "freeze_hyps": 0,
    }


    bi, final_cluster = cluster_GA(
        nPool,
        eleNames,
        eleNums,
        eleRadii,
        generations,
        calc,
        filename,
        log_file,
        CXPB,
        singleTypeCluster,
        use_dask,
        use_vasp,
        al_method,
        learner_params,
        train_config,
        optimizer,
        use_vasp_inter,
        restart,
        gen_num,
    )
