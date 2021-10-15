.. _Version history:

Version history
=================

v21.1
--------
- Separate function: generate / preprocess works only if each tag turns on in input.yaml.
- Bug fix: Inconsistent position from ASE.
- Memory issue fix: pair_nn.cpp for more atom type without memory leak.

v20.2
--------
- Replica ensemble: you can estimate the standard deviation of atomic energy to detect the atom out of training set.
- Bug fix: LAMMPS atom type matching error, memory allocation

v20.1
--------
- :code:`use_stress`: (default false) if you set :code:`use_stress` true in :code:`neural_network` of your :code:`input.yaml` , you can train your neural network with virial stress tensor
- LAMMPS Optimization: Using optimization in for loop and applying 'memoization', it becomes around 2 times faster!
