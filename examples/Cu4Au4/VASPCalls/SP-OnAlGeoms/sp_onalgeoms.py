from ase.calculators.vasp import Vasp
from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator as sp
import subprocess
import os
import shutil

calc = Vasp(
        kpar=1,
        ncore=8,
        encut=400,
        xc="PBE",
        kpts=(1, 1, 1),
        gamma=True,  # Gamma-centered
        ismear=1,
        sigma=0.2,
        ibrion=-1,
        nsw=0,
        #potim=0.2,
        isif=0,
        # ediffg=-0.02,
        # ediff=1e-6,
        lcharg=False,
        lwave=False,
        lreal=False,
        ispin=2,
        isym=0,
    )

for k in range(26):
    fname='mut_after_relax_gen'+str(k+1)+'.traj'
    trajopt = 'vasp_mut_after_relax_gen'+str(k+1)+'.traj'

    traj = Trajectory(fname)
    
    with open('results_sp_onalgeoms.log', 'a+') as fout:
        fout.write(f'Trajectory: {fname}' '\n') 
        fout.write(f'Total images in the trajectory: {len(traj)}' '\n') 
    img_list = []
    for i, img in enumerate(traj):
        print(i, img) 
        img.set_calculator(calc)
        energy = img.get_potential_energy()
        forces = img.get_forces()
        img.set_calculator(sp(atoms=img, energy=energy, forces=forces))
        img_list.append(img)

        with open('results_sp_onalgeoms.log', 'a+') as fout:
            fout.write(f'{i}th trajectory image Energy: {energy}' '\n') 
            fout.write('\n') 

    write(trajopt,img_list)
    trajout = Trajectory(trajopt)
    print(len(trajout))
