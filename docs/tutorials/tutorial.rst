=========
Tutorials
=========

Inputs
======

simple-nn use YAML style input file.
detailed input parameters are listed in module tab

simplified input file (input.yaml)::

    atom_types:
        - Si

    symmetry_function:
        params:
            Si: params_Si

    neural_network:
        method: Adam
        use_force: true


Run the code
============

>>> python run.py