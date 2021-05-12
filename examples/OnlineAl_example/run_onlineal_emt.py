from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch
from ase.optimize import BFGS

if __name__ == "__main__":
    use_dask = True
    eleNames = ["Ni"]
    eleNums = [10]
    nPool = 10
    generations = 20
    CXPB = 0.5
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Ni10"  # For saving the best cluster at every generation
    log_file = "clus_Ni10.log"
    singleTypeCluster = True
    calc = EMT()
    use_vasp = True
    al_method = "online"
    if use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)
    learner_params = {
        "max_iterations": 40,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "uncertain_tol": 0.5,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 3,
        "use_dask": False,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 20},
        "optim": {
            "device": "cpu",
            "force_coefficient": 4.0,
            "lr": 1e-3,
            "batch_size": 10,
            "epochs": 200,
            "optimizer": torch.optim.LBFGS,
            "optimizer_args": {"optimizer__line_search_fn": "strong_wolfe"},
        },
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
        config,
        optimizer=BFGS,  # Set ase optimizer
    )
