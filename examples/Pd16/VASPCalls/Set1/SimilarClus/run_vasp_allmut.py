from ase.calculators.vasp import Vasp
from ase.io import read, write, Trajectory
from ase.calculators.singlepoint import SinglePointCalculator as sp
import subprocess
import os
import shutil

calc = Vasp(
        kpar=1,
        ncore=16,
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

fname='mut_before_relax_gen1.traj'
fout='mut_before_relax_gen1_relax.traj'
i = 1


traj = Trajectory(fname)
img = traj[i] 
img.set_calculator(calc)
energy = img.get_potential_energy()
forces = img.get_forces()
img.set_calculator(sp(atoms=img, energy=energy, forces=forces))

write(fout,img)
fout1 = Trajectory(fout)
print(len(fout1))

write("trajout.traj","OUTCAR", format='vasp-out')
fout2 = Trajectory("trajout.traj")
print(len(fout2))
