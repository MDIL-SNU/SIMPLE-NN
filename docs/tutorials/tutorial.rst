.. include:: /share.rst

=========
Tutorials
=========

Preparing dataset
=================
simple-nn use the data from *ab initio* packages like VASP and Quantum espresso.
These calculation outputs are handled with ASE package.
Thus, one can use various *ab initio* packages that supported by ASE.

simple-nn need atom coordinates, lattice parameters, structure energy 
and atomic force(optional) to generate dataset. One can check if the output has
the information that simple-nn need with the command below:

::

    from ase import io

    atoms = io.read('some_output')
    # atom coordinates
    atoms.get_positions()
    # lattice parameters
    atoms.get_cell()
    # structure_energy
    atoms.get_potential_energy()
    # atomic force
    atoms.get_forces()


Preparing inputs files
======================

simple-nn use YAML style input file, input.yaml.
input.yaml consists of 3 part: basic parameters, 
feature-related parameters and model-related parameters.
The basic format of input.yaml is like below::

    # Basic parameters
    generate_features: true
    preprocess: true
    train_model: true
    atom_types:
      - Si
      - O

    # feature-related parameters
    symmetry_function: # class name of feature
    params:
      Si: params_Si
      O: params_O

    # model-related parameters
    neural_network: # class name of model
      method: Adam
      nodes: 30-30
      batch_size: 10
      total_epoch: 50000
      learning_rate: 0.001

Depending on the feature and model classes, additional input files may be required.
For example, for :gray:`symmetry_function` class, 
additional files named :gray:`str_list` and :gray:`params_XX` are required. 
Details of parameters and additional files are listed in 
:doc:`/simple_nn/Simple_nn` (basic parameters), 
:doc:`/features/features` (feature-related parameters) and 
:doc:`/models/models` (model-related parameters) page.

Run the code
============

After preparing all input files, one simply run the predefined script run.py to run simple-nn.
The basic format of :gray:`run.py` is described below::

    # run.py
    #
    # Usage:
    # $ python run.py

    from simple_nn import Simple_nn
    from simple_nn.features.symmetry_function import Symmetry_function
    from simple_nn.models.neural_network import Neural_network

    model = Simple_nn('input.yaml', 
                      descriptor=Symmetry_function(), 
                      model=Neural_network())
    model.run()

One can find the practical usage in :doc:`/examples/examples`



