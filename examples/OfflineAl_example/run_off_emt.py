from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch
from ase.optimize import BFGS

"""
Example code to run 5 generations of the Genetic algorithm for a Cu5 cluster using the
offline active learning framework and the ASE pure python EMT calculator
For more info on parameters : https://github.com/ulissigroup/al_mlp
"""

if __name__ == "__main__":
    use_dask = False
    eleNames = ["Cu"]
    eleNums = [5]
    nPool = 5
    generations = 5
    CXPB = 0.5
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Cu5"  # For saving the best cluster at every generation
    log_file = "clus_Cu5.log"
    singleTypeCluster = True
    calc = EMT()
    use_vasp = False
    al_method = "offline"
    if use_dask == True:
        # Set up the dask run using the worker-spec file based on the computing cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)  # Since 10 clusters in the pool

    learner_params = {
        "max_iterations": 2,
        "force_tolerance": 0.01,
        "samples_to_retrain": 2,
        "filename": "relax_example",
        "file_dir": "./",
        "max_evA": 0.07,  # eV/AA
        "use_dask": False,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 3, "num_nodes": 20},
        "optim": {
            "device": "cpu",
            "force_coefficient": 30.0,
            "lr": 0.01,
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
