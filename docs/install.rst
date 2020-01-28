.. _install:

=======
Install
=======

Install SIMPLE-NN
=================
SIMPLE-NN is tested and supported on the following versions of Python:

- Python :code:`2.7`, :code:`3.4-3.6`

Requirements
------------
In SIMPLE-NN, various Python modules are used. 
Most of these modules are installed automatically during the install process of SIMPLE-NN.
However, we recommend to install Tensorflow manually for better performance.
In addition, you need to install mpi4py(optional) manually if you want to use MPI.
(MPI is supported only in generate_features and preprocess part. See :doc:`/simple_nn/Simple_nn` section.) 
Detailed information for installing Tensorflow and mpi4py can be found from the links below.

**Install Tensorflow**: https://www.tensorflow.org/install/

Tensorflow :code:`r1.6`-:code:`r1.13` is supported.

**Install mpi4py**: https://mpi4py.readthedocs.io/en/stable/install.html


Install from source
-------------------

You can download a current SIMPLE-NN source package from link below. 
Once you have a zip file, unzip it. This will create SIMPLE-NN directory.
After unzipping the file, run the command below to install SIMPLE-NN.

**Download SIMPLE-NN**: https://github.com/MDIL-SNU/SIMPLE-NN

::

    cd SIMPLE-NN
    python setup.py install
    # If root permission is not available, add --user command like below
    # python setup.py install --user

Currently, pip install is not supported but will be addressed.



Install LAMMPS implementation
=============================

To utilize LAMMPS package for SIMPLE-NN generated NN potentials, 
you must copy the source codes to LAMMPS/src directory with the following command 
and compile LAMMPS package.

::

    cp SIMPLE-NN/simple_nn/features/symmetry_function/symmetry_functions.h /path/to/lammps/src/
    cp SIMPLE-NN/simple_nn/features/symmetry_function/pair_nn.* /path/to/lammps/src/
