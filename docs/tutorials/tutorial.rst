.. include:: /share.rst

=========
Tutorials
=========

Preparing dataset
=================
SIMPLE-NN uses `ASE`_ to handle output from *ab initio* programs like VASP or Quantum ESPRESSO. 
All output types supported by ASE can be used in SIMPLE-NN, 
but they need to contain essential information such as atom coordinates, lattice parameters, energy, and forces.  
You can check if the output file contains the appropriate information by using the following command:

.. _ASE: https://wiki.fysik.dtu.dk/ase/

::

    from ase import io

    atoms = io.read('some_output')
    # atom coordinates
    atoms.get_positions()
    # chemical symbols
    atoms.get_chemical_symbols()
    # lattice parameters
    atoms.get_cell()
    # structure_energy
    atoms.get_potential_energy()
    # atomic force
    atoms.get_forces()


Preparing inputs files
======================

SIMPLE-NN use YAML style input file: :gray:`input.yaml`.
:gray:`input.yaml` consists of three part: basic parameters, 
feature-related parameters, and model-related parameters.
The basic format of :gray:`input.yaml` is like below::

    # Basic parameters
    generate_features: true
    preprocess: true
    train_model: true
    atom_types:
      - Si
      - O

    # feature-related parameters
    symmetry_function: # class name of the feature
    params:
      Si: params_Si
      O: params_O

    # model-related parameters
    neural_network: # class name of the model
      method: Adam
      nodes: 30-30
      batch_size: 10
      total_epoch: 50000
      learning_rate: 0.001

Depending on the feature and model classes, additional input files may be required.
For example, for :gray:`symmetry_function` class, 
additional files named :gray:`str_list` and :gray:`params_XX` 
(name of :gray:`params_XX` can be changed) are required. 
Details of parameters and additional files are listed in 
:doc:`/simple_nn/Simple_nn` (basic parameters), 
:doc:`/features/features` (feature-related parameters) and 
:doc:`/models/models` (model-related parameters) section.

Running the code
================

To run SIMPLE-NN, you simply have to run the predefined script :gray:`run.py` after preparing all input files.
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

Examples of actual use for the entire process of generating neural network potential 
can be found in :doc:`/examples/examples` section or :gray:`SIMPLE-NN/examples/`

Outputs
=======

The default output file of SIMPLE-NN is :gray:`LOG`, which contains the execution log of SIMPLE-NN.
In addition to :gray:`LOG`, an additional output file is created for each process of SIMPLE-NN. 
After :gray:`Symmetry_function.generate` method, you can find the output files listed below:

    - :gray:`data/data##.pickle`\: (## indicates number) 
      Data file which contains descriptor vectors, a derivative of descriptor 
      vectors, and other parameters per structure.


After :gray:`Symmetry_function.preprocess` method, you can find the output files listed below:

    - :gray:`data/{training,valid}_data_####_to_####.tfrecord`\: 
      Packed Training/validation dataset which contains the information in 
      :gray:`data/data##.pickle` and additional parameters like atomic weights
      which are used during training process.
    - :gray:`pickle_{training,valid}_list`\: List of pickle files that are included in 
      :gray:`data/{training,valid}_data_####_to_####.tfrecord` file.

    - :gray:`{train,valid}_list`\: List of tfrecord files (used in network optimization process)
    - :gray:`scale_factor`\: Scale factor for symmetry function.
    - :gray:`atomic_weights`\: Data file that contains atomic weights.


After :gray:`Neural_network.train` method, you can find the output files listed below:

    - :gray:`SAVER_epochXXXX.*`, :gray:`checkpoint`\: Tensorflow save file which contains 
      the network information at the XXXXth epoch.
    - :gray:`potential_saved_epochXXXX`\: LAMMPS potential file which contains 
      the network information at the XXXXth epoch.


.. MDwithLAMMPS_

MD simulation with LAMMPS
=========================

To run MD simulation with LAMMPS, add the lines into the LAMMPS script file.
::

    pair_style nn
    pair_coeff * * /path/to/potential_saved Si O

Regarding the unit system, the NNP trained with VASP output is compatible with the LAMMPS units ‘metal’. 
For outputs from other ab initio programs, however, 
the appropriate unit should be chosen with the user’s discretion.