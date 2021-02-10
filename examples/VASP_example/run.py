from cluster_mlp.deap_ga import cluster_GA
from cluster_mlp.dask_utils import call_dask
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.emt import EMT
from dask_kubernetes import KubeCluster
from dask.distributed import Client


if __name__ == "__main__":
    use_dask = True
    eleNames = ["Cu"]
    eleNums = [4]
    nPool = 10
    generations = 50
    CXPB = 0.7
    eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in eleNames]
    filename = "clus_Cu4"  # For saving the best cluster at every generation
    log_file = "clus_Cu4.log"
    singleTypeCluster = False
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

    if use_dask == True:
        # Run between 0 and 4 1-core/1-gpu workers on the kube cluster
        cluster = KubeCluster.from_yaml("worker-cpu-spec.yml")
        client = Client(cluster)
        # cluster.adapt(minimum=0, maximum=10)
        cluster.scale(10)
        call_dask(client)

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
    )
