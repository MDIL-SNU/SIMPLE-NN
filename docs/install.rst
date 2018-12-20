.. include:: /share.rst
.. _install:

=======
Install
=======

Install simple-nn
=================
simple-nn is tested and supported on the following version of Python:

- Python :gray:`2.7`, :gray:`3.4-3.6`

Download a package
------------------
simple-nn package can be found at this link:

After unzip the file, run the command below to install simple-nn::

    Python setup.py

Currently pip install is not supported but will be addressed.

Requirements
------------
In simple-nn, various Python modules are used. 
Most of these modules are installed automatically during the install process of simple-nn.
However, one need to install Tensorflow and mpi4py(optional) manually before install simple-nn.
Detailed method to install Tensorflow and mpi4py is listed below. 

**Install Tensorflow**: https://www.tensorflow.org/install/

We support Tensorflow >= :gray:`r1.6`.

**Install mpi4py**::

    pip install mpi4py
