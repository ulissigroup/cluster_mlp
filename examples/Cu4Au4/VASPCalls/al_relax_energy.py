from ase.calculators.vasp import Vasp
from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator as sp
import subprocess
import os
import shutil


for k in range(26):
    fname='mut_after_relax_gen'+str(k+1)+'.traj'
    print(fname)


    traj = Trajectory(fname)
    for i, img in enumerate(traj):
        energy = img.get_potential_energy()
        forces = img.get_forces()
        print(i, energy)
        #print(forces)
    print('\n')
