import numpy as np
from al_mlp.preset_learners.offline_learner.fmax_learner import FmaxLearner
from amptorch.trainer import AtomsTrainer
from al_mlp.atomistic_methods import Relaxation
from al_mlp.base_calcs.morse import MultiMorse
from ase.io import read
from ase.calculators.emt import EMT
import os


def run_offlineal(cluster, parent_calc, elements, al_learner_params, config, optimizer):

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

    al_learner_params["atomistic_method"] = Relaxation(
        cluster, optimizer, fmax=0.01, steps=100
    )

    config["dataset"] = {
        "raw_data": images,
        "val_split": 0,
        "elements": elements,
        "fp_params": Gs,
        "save_fps": False,
        "scaling": {"type": "normalize", "range": (-1, 1)},
    }

    config["cmd"] = {
        "debug": False,
        "run_dir": "./",
        "seed": 2,
        "identifier": "cluster",
        "verbose": True,
        # "logger": True,
        "single-threaded": True,
    }

    trainer = AtomsTrainer(config)
    # base_calc = MultiMorse(images, Gs["default"]["cutoff"], combo="mean")
    base_calc = EMT()
    offlinecalc = FmaxLearner(
        al_learner_params, trainer, images, parent_calc, base_calc
    )
    if os.path.exists("queried_images.db"):
        os.remove("queried_images.db")

    offlinecalc.learn()
    al_iterations = offlinecalc.iterations - 1

    file_path = al_learner_params["file_dir"] + al_learner_params["filename"]
    final_ml_traj = read("{}_iter_{}.traj".format(file_path, al_iterations), ":")
    relaxed_clus = final_ml_traj[-1]

    return relaxed_clus, offlinecalc.parent_calls
