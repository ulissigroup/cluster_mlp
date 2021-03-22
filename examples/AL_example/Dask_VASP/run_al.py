from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from ase.calculators.vasp import Vasp2
from dask_kubernetes import KubeCluster
from dask.distributed import Client
import torch


if __name__ == "__main__":
    use_dask = True
    eleNames = ["Cu"]
    eleNums = [5]
    nPool = 10
    generations = 2
    CXPB = 0.5
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Cu5"  # For saving the best cluster at every generation
    log_file = "clus_Cu5.log"
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
        ediffg=-0.02,
        ediff=1e-6,
        lcharg=False,
        lwave=False,
        lreal=False,
        ispin=2,
        isym=0,
    )
    use_vasp = True
    use_al = True
    if use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)
    learner_params = {
        "max_iterations": 10,
        "samples_to_retrain": 1,
        "filename": "relax_example",
        "file_dir": "./",
        "uncertain_tol": 0.1,
        "fmax_verify_threshold": 0.05,  # eV/AA
        "relative_variance": True,
        "n_ensembles": 10,
        "use_dask": False,
    }

    config = {
        "model": {"get_forces": True, "num_layers": 20, "num_nodes": 3},
        "optim": {
            "device": "cpu",
            "force_coefficient": 30.0,
            "lr": 1e-2,
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
        use_al,
        learner_params,
        config,
    )
