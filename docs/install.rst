.. include:: /share.rst
.. _install:

=======
Install
=======

Install SIMPLE-NN
=================
SIMPLE-NN is tested and supported on the following version of Python:

- Python :gray:`2.7`, :gray:`3.4-3.6`

Requirements
------------
In SIMPLE-NN, various Python modules are used. 
Most of these modules are installed automatically during the install process of SIMPLE-NN.
However, you need to install Tensorflow and mpi4py(optional) manually before installing SIMPLE-NN.
Detailed method to install Tensorflow and mpi4py is listed below. 

**Install Tensorflow**: https://www.tensorflow.org/install/

We support Tensorflow >= :gray:`r1.6`.

**Install mpi4py**::

    pip install mpi4py

Install from source
-------------------

You can download a current SIMPLE-NN source package fromlink below. 
Once you have a zip file unzip it, this will create SIMPLE-NN directory.
After unzipping the file, run the command below to install SIMPLE-NN.

::

    cd SIMPLE-NN
    python setup.py

Currently, pip install is not supported but will be addressed.



Install LAMMPS implements
=========================

To utilize LAMMPS package for SIMPLE-NN generated NN potentials, 
you must copy the source code to LAMMPS/src directory with following command 
and compile LAMMPS package.

::

    cp SIMPLE-NN/simple_nn/features/symmetry_function/{symmetry_function.*,pair_nn.*} /path/to/lammps/src/