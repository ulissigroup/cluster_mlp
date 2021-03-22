from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch


if __name__ == "__main__":
    use_dask = False
    eleNames = ["Cu"]
    eleNums = [5]
    nPool = 5
    generations = 2
    CXPB = 0.7
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Cu5"  # For saving the best cluster at every generation
    log_file = "clus_Cu5.log"
    singleTypeCluster = True
    calc = EMT()
    use_vasp = False
    al_method = "offline"
    if use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)

    learner_params = {
        "max_iterations": 10,
        "force_tolerance": 0.01,
        "samples_to_retrain": 2,
        "filename": "relax_example",
        "file_dir": "./",
        "max_evA": 0.07,  # eV/AA
        "use_dask": True,
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
    )
