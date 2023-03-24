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
        ibrion=2,
        nsw=1000,
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

fname='mut_before_relax_gen2.traj'

trajopt = 'vasp_'+fname

traj = Trajectory(fname)
    
with open('results.log', 'a+') as fout:
    fout.write(f'Trajectory: {fname}' '\n') 
    fout.write(f'Total images in the trajectory: {len(traj)}' '\n') 
    
i = 9
img=traj[i]

img.set_calculator(calc)
energy = img.get_potential_energy()
forces = img.get_forces()
img.set_calculator(sp(atoms=img, energy=energy, forces=forces))

with open('OSZICAR', 'r') as f:
    last_line = f.readlines()[-1]
    last_line.strip()
    ionic_line_split = last_line.split()
    vasp_calls = int(ionic_line_split[0])
    print(int(ionic_line_split[0]))
    
with open('results.log', 'a+') as fout:
    fout.write(f'{i}th trajectory image Energy: {energy}' '\n') 
    fout.write(f'{i}th trajectory image VASP Calls: {vasp_calls}' '\n') 
    fout.write('\n') 

write(trajopt,img)
trajout = Trajectory(trajopt)
print(len(trajout))
