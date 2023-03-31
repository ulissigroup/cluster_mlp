import ase
from ase.io import read,write, Trajectory
from matplotlib import pyplot as plt
import glob
from ase.calculators.emt import EMT
import numpy as np

traj = Trajectory('mut_after_filter_gen19.traj')
sorted_traj = 'mut_after_filter_gen19_sorted.traj'
best_traj = 'best_n_clus_after_gen19.traj'
print(len(traj))

ene_list = []
for i, clus in enumerate(traj):
    ene = clus.get_potential_energy()
    ene_list.append(ene)
print(ene_list)

sort_index = np.argsort(ene_list)
print(sort_index)

img_list = []
for i in sort_index:
    img = traj[i]
    img_list.append(img)

write(sorted_traj, img_list) 

out_traj = Trajectory(sorted_traj)

print(len(out_traj))

ene_list_out = []
for i, clus in enumerate(out_traj):
    ene = clus.get_potential_energy()
    ene_list_out.append(ene)
print(ene_list_out)

sort_index_out = np.argsort(ene_list_out)
print(sort_index_out)


best_img_list = []
for i in range(10):
    best_img = traj[i]
    best_img_list.append(best_img)
write(best_traj, best_img_list) 

select_best_10 = Trajectory(best_traj)
print(len(select_best_10))

ene_list_best = []
for i, clus in enumerate(select_best_10):
    ene = clus.get_potential_energy()
    ene_list_best.append(ene)
print(ene_list_best)

sort_index_best = np.argsort(ene_list_best)
print(sort_index_best)
