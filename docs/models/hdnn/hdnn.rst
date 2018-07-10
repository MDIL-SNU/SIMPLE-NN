===============================
High-dimensional neural network
===============================

Introduction
============
Total energy is the sum of atomic energies.

Network architecture
====================

Inputs
======

Function related parameter
--------------------------

* train (boolean, default: true)
    If true, training process proceeds.

* test (boolean, default: false)
    If true, predicted energy and forces for test set are calculated.

* continue (boolean, default: false)
    If true, training process restart from save file (SAVER.*, checkpoints)


Network related parameter
-------------------------
* nodes (str or dict, default: 30-30)
    String value to indicate the network architecture.
    30-30 means 2 hidden layers and each hidden layer has 30 hidden nodes.
    One can use different network for different atom types.
    In this case, use like below.
    
::

    nodes:
        Si: 30-30
        O: 15-15

* regularization (dict)
    Regularization setting.

* use_force (boolean, default: false)
    If true, both energy and force are used for training.




Optimization related parameter
------------------------------

* method (str, default: Adam)
    Optimization method. Currently only support Adam optimizer.

* batch_size (int, default: 64)
    The number of samples in batch training set.

* total_epoch (int, default: 10000)
    The number of total training epoch

* learning_rate (float, default: 0.01)
    * Exponential decay available. See :ref:`exponential_dacay-label`

    Learning rate for gradient descendent based optimization algoritm.

* force_coeff (float, default: 0.3) / energy_coeff (float, default: 1.)
    * Exponential decay available. See :ref:`exponential_dacay-label`

    Scaling coefficient for force and energy loss.

* loss_scale (float, default: 1.)
    Scaling coefficient for entire loss function.


.. _exponential_dacay-label:

Exponential decay
-----------------
One need to change some parameters (e.g. learning rate) gradually during training process.
In simple_nn, we use expoenetial_decay function in Tensorflow.

::

    parameter_name:
        learning_rate: 1.
        decay_rate: 0.95
        decay_steps: 10000
        staircase: false