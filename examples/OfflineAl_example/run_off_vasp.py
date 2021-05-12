from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.vasp import Vasp2
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch
from ase.optimize import BFGS

if __name__ == "__main__":
    use_dask = True
    eleNames = ["Cu"]
    eleNums = [10]
    nPool = 10
    generations = 20
    CXPB = 0.5
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Cu10"  # For saving the best cluster at every generation
    log_file = "clus_Cu10.log"
    singleTypeCluster = True
    # calc = EMT()
    calc = Vasp2(
        kpar=1,
        ncore=4,
        encut=400,
        xc="PBE",
        kpts=(1, 1, 1),
        gamma=True,  # Gamma-centered
        ismear=1,
        sigma=0.2,
        ibrion=-1,
        nsw=0,
        potim=0.2,
        isif=0,
        # ediffg=-0.02,
        # ediff=1e-6,
        lcharg=False,
        lwave=False,
        lreal=False,
        ispin=2,
        isym=0,
    )
    use_vasp = True
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
