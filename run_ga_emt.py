from cluster_mlp.clus_ga_deap_emt import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch
from ase.optimize import BFGS
#from ase.optimize import GPMin
from vasp_interactive import VaspInteractive

if __name__ == "__main__":
    use_dask = False
    eleNames = ["Pt"]
    eleNums = [10]
    nPool = 10
    generations = 50
    CXPB = 0.5
    use_vasp = False
    use_vasp_inter = False
    al_method = "Online"
    optimizer = BFGS
    restart = False
    gen_num = 16
    calc = EMT()

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
