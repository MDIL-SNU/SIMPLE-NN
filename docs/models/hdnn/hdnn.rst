.. include:: /share.rst

===============================
High-dimensional neural network
===============================

Introduction
============
Total energy is the sum of atomic energies.

Network architecture
====================

Parameters
==========

Function related parameter
--------------------------

* :gray:`train`\: (boolean, default: true)
  If true, training process proceeds.

* :gray:`test`\: (boolean, default: false)
  If true, predicted energy and forces for test set are calculated.

* :gray:`continue`\: (boolean, default: false)
  If true, training process restart from save file (SAVER.*, checkpoints)


Network related parameter
-------------------------
* :gray:`nodes`\: (str or dictionary, default: 30-30)
  String value to indicate the network architecture.
  30-30 means 2 hidden layers and each hidden layer has 30 hidden nodes.
  One can use different network for different atom types.
  In this case, use like below.
    
::

    nodes:
        Si: 30-30
        O: 15-15

* :gray:`regularization`\: (dictionary)
  Regularization setting. Currently, L1 and L2 regularization is available. 
  ::

    regularization:
      - type:
      - params:

* :gray:`use_force`\: (boolean, default: false)
  If :gray:`true`, both energy and force are used for training.

* :gray:`double_precision`\: ()

* :gray:`stddev`\: ()


Optimization related parameter
------------------------------

* :gray:`method`\: (str, default: Adam)
  Optimization method. One can choose Adam or L-BFGS. 

* :gray:`batch_size`\: (int, default: 64)
  The number of samples in batch training set.

* :gray:`full_batch`\: ()

* :gray:`total_epoch`\: (int, default: 10000)
  The number of total training epoch.

* :gray:`learning_rate`\: (float, default: 0.01, :ref:`exponential_dacay-label`)
  Learning rate for gradient descendent based optimization algoritm.

* :gray:`force_coeff` and :gray:`energy_coeff`\: (float, default: 0.3 and 1., :ref:`exponential_dacay-label`)
  Scaling coefficient for force and energy loss.

* :gray:`loss_scale` (float, default: 1.)
    Scaling coefficient for entire loss function.


Logging & saving related parameters
-----------------------------------
* :gray:`save_interval`\: ()

* :gray:`show_interval`\: ()

* :gray:`echeck` and :gray:`fcheck`\: ()

* :gray:`break_max`\: ()

* :gray:`print_structure_rmse`\: ()


Performance related parameters
------------------------------
* :gray:`inter_op_parallelism_threads`\: ()

* :gray:`intra_op_parallelism_threads`\: ()

* :gray:`cache`\: ()


.. _exponential_dacay-label:

Exponential decay
-----------------
Some parameters in neural_network may need to decrease exponentially during optimization process. 
In the case, one can use this format instead of float value. More information can be found in 
Tensorflow homepage(link)

::

    parameter_name:
        learning_rate: 1.
        decay_rate: 0.95
        decay_steps: 10000
        staircase: false

methods
=======
.. py:function::
    __init__(self)

    Initiator of Neural_network class. 

.. py:function::
    train(self, user_optimizer=None, user_atomic_weights_function=None)

    Args:
        - :gray:`user_optimizer`\: User defined optimizer. 
          Can be set in the script run.py
        - :gray:`user_atomic_weights_function`\: User defined atomic weight function.

    Method for optimizing neural network potential.