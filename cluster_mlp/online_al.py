import numpy as np
from al_mlp.online_learner import OnlineLearner
from amptorch.trainer import AtomsTrainer
from ase.optimize import BFGS
from al_mlp.atomistic_methods import Relaxation
import os


def run_onlineal(cluster, parent_calc, elements, al_learner_params, config, optimizer):

    Gs = {
        "default": {
            "G2": {
                "etas": np.logspace(np.log10(0.05), np.log10(5.0), num=4),
                "rs_s": [0],
            },
            "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
            "cutoff": 6,
        },
    }

    images = [cluster]

    config["dataset"] = {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_params": Gs,
        "save_fps": False,
        "scaling": {"type": "standardize"},
    }

    config["cmd"] = {
        "debug": False,
        "run_dir": "./",
        "seed": 1,
        "identifier": "cluster",
        "verbose": False,
        # "logger": True,
        "single-threaded": True,
    }

    trainer = AtomsTrainer(config)

    onlinecalc = OnlineLearner(
        al_learner_params,
        trainer,
        images,
        parent_calc,
    )
    if os.path.exists("relaxing.traj"):
        os.remove("relaxing.traj")

    optim_struc = Relaxation(cluster, optimizer, fmax=0.01, steps=100)
    optim_struc.run(onlinecalc, filename="relaxing")
    relaxed_clus = optim_struc.get_trajectory("relaxing")[-1]

    return relaxed_clus, onlinecalc.parent_calls
