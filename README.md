# cluster-mlp

## Installation with conda

```
# Ignore first 2 steps if you have mamba already installed
conda activate base
conda install -c conda-forge mamba

git clone --recursive https://github.com/ulissigroup/cluster_mlp.git
cd cluster_mlp
# This will create a conda environment cluster-ga
mamba env create --file conda_env.yml
conda activate cluster-ga
# Run quick test
python run_emt_online.py
#Example run file for vasp is provided run_vasp_online.py

```


Install the package with `pip install git+https://github.com/ulissigroup/cluster_mlp.git`

An ASE + DEAP implementation of the genetic algorithm framework presented in the following papers:
- GIGA: a versatile genetic algorithm for free and supported clusters and nanoparticles in the presence of ligands (Marc Jäger,Rolf Schäfer and  Roy L. Johnston) [https://doi.org/10.1039/C9NR02031D]
- First principles global optimization of metal clusters and nanoalloys (Marc Jäger,Rolf Schäfer and  Roy L. Johnston) [https://doi.org/10.1080/23746149.2018.1516514]
- New AuN (N = 27–30) Lowest Energy Clusters Obtained by Means of an Improved DFT–Genetic Algorithm Methodology (Jorge A. Vargas, Fernando Buendía, and Marcela R. Beltrán) [https://doi.org/10.1021/acs.jpcc.6b12848]

Required dependancies:
- Atomic Simulation Environment(ASE) [https://wiki.fysik.dtu.dk/ase/about.html]
- Distributed Evolutionary algorithms in Python(DEAP) [https://deap.readthedocs.io/en/master/index.html]
