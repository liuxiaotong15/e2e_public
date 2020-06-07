# e2e_public
For paper 'Artificial intuition for solving chemistry problems via an End-to-End approach'

# db
we download the [ASE](https://wiki.fysik.dtu.dk/ase/ase/db/db.html) format qm9 from [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack).

you can get it directly from another repo of this paper: [e2e_electron_counting](https://github.com/liuxiaotong15/qm9_electron_counting)

# requirments
tensorflow 1.14,
ASE, ...

# how to run
```
python bond_cluster.py

python main.py -m type

python main.py -m xyz

python main.py -m ucfc
```

# new clustering method

bond_cluster.py only contains 2 atoms cluster. We improve it in another repo [e2e_reaction_public](https://github.com/liuxiaotong15/e2e_reaction_public) for further more atoms clustering work.
