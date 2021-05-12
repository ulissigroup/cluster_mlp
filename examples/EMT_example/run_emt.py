from cluster_mlp.deap_ga import cluster_GA
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client
from ase.optimize import BFGS

if __name__ == "__main__":
    use_dask = False
    eleNames = ["Cu", "Al"]
    eleNums = [3, 5]
    nPool = 10
    generations = 20
    CXPB = 0.5
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Cu4"  # For saving the best cluster at every generation
    log_file = "clus_Cu4.log"
    singleTypeCluster = False
    calc = EMT()
    use_vasp = False

    if use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)

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
        optimizer=BFGS,  # Set ase optimizer
    )
