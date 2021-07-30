from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
from al_mlp.atomistic_methods import Relaxation, replay_trajectory
import os

# Refer examples or https://github.com/ulissigroup/al_mlp for sample parameters


def run_onlineal(cluster, parent_calc, elements, al_learner_params, config, optimizer):

    images = [cluster]

    flare_params = config

    ml_potential = FlarePPCalc(flare_params, images)

    onlinecalc = OnlineLearner(
        al_learner_params,
        images,
        ml_potential,
        parent_calc,
    )

    if os.path.exists("relaxing.traj"):
        os.remove("relaxing.traj")

    optim_struc = Relaxation(cluster, optimizer, fmax=0.01, steps=100)
    optim_struc.run(onlinecalc, filename="relaxing")
    relaxed_clus = optim_struc.get_trajectory("relaxing")[-1]

    return relaxed_clus, onlinecalc.parent_calls
