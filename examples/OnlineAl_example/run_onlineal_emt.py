from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client
from ase.optimize import BFGS

if __name__ == "__main__":
    cluster_use_dask = False
    eleNames = ["Cu"]
    eleNums = [10]
    nPool = 5
    generations = 10
    CXPB = 0.5
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Ni10"  # For saving the best cluster at every generation
    log_file = "clus_Ni10.log"
    singleTypeCluster = True
    calc = EMT()
    use_vasp = True
    al_method = "online"

    if cluster_use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)

    learner_params = {
        "max_iterations": 3,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "stat_uncertain_tol": 0.1,
        "dyn_uncertain_tol": 1.2,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 10,
        "use_dask": False,
    }

    config = {
        "sigma": 1.0,
        "power": 2,
        "cutoff_function": "quadratic",
        "cutoff": 3.0,
        "radial_basis": "chebyshev",
        "cutoff_hyps": [],
        "sigma_e": 1.0,
        "sigma_f": 0.1,
        "sigma_s": 0.0,
        "max_iterations": 50,
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
        cluster_use_dask,
        use_vasp,
        al_method,
        learner_params,
        config,
        optimizer=BFGS,  # Set ase optimizer
    )
