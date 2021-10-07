from al_mlp.online_learner.online_learner import OnlineLearner
from al_mlp.ml_potentials.flare_pp_calc import FlarePPCalc
from al_mlp.atomistic_methods import Relaxation, replay_trajectory, check_final_point
import os
from ase.optimize import *
from ase.calculators.vasp import Vasp
from ase.calculators.emt import EMT
from vasp_interactive import VaspInteractive
from ase.io.trajectory import Trajectory
# Refer examples or https://github.com/ulissigroup/al_mlp for sample parameters

def run_onlineal(cluster, parent_calc, elements, al_learner_params, config, optimizer):

    images = [cluster]

    flare_params = config

    ml_potential = FlarePPCalc(flare_params, images)

    if ((type(parent_calc) == Vasp) or (type(parent_calc == EMT))):
        onlinecalc = OnlineLearner(
            al_learner_params,
            images,
            ml_potential,
            parent_calc,
            )
        if os.path.exists("relaxing.traj"):
            os.remove("relaxing.traj")
        cluster.calc = onlinecalc
        dyn = optimizer(cluster, trajectory = 'relaxing.traj')
        #dyn.attach(replay_trajectory, 1, cluster.calc, dyn)
        dyn.attach(check_final_point, 1, cluster.calc, dyn)
        dyn.run(fmax=0.05, steps=1000) 

    elif type(parent_calc) == VaspInteractive:
        with parent_calc as calc:
            onlinecalc = OnlineLearner(
                al_learner_params,
                images,
                ml_potential,
                calc,
                )

            if os.path.exists("relaxing.traj"):
                os.remove("relaxing.traj")
            cluster.calc = onlinecalc
            dyn = optimizer(cluster, trajectory = 'relaxing.traj')
            dyn.attach(replay_trajectory, 1, cluster.calc, dyn)
            dyn.attach(check_final_point, 1, cluster.calc, dyn)
            dyn.run(fmax=0.05, steps=1000)

    #optim_struc = Relaxation(cluster, optimizer, fmax=0.01, steps=100)
    #optim_struc.run(onlinecalc, filename="relaxing")
    #relaxed_clus = optim_struc.get_trajectory("relaxing")[-1]
    relaxed_clus = Trajectory('relaxing.traj')[-1]

    return relaxed_clus, onlinecalc.parent_calls
