.. include:: /share.rst

===============================
High-dimensional neural network
===============================

Introduction
============
simple-nn use High-Dimensional Neural Network(HDNN) [#f1]_ as a default machine learning model.

Parameters
==========

Function related parameter
--------------------------

* :gray:`train`\: (boolean, default: true)
  If :gray:`true`, training process proceeds.

* :gray:`test`\: (boolean, default: false)
  If :gray:`true`, predicted energy and forces for test set are calculated.

* :gray:`continue`\: (boolean, default: false)
  If :gray:`true`, training process restart from save file (SAVER.*, checkpoints). 
  If :gray:`weights`, training process restart from the LAMMPS potential file (potential_saved).

.. Note::
    :gray:`potential_saved` only contains the weights and bias of the network. 
    Thus, hyperparameter used in Adam optimizer is reset with :gray:`weights`.


Network related parameter
-------------------------
* :gray:`nodes`\: (str or dictionary, default: 30-30)
  String value to indicate the network architecture.
  30-30 means 2 hidden layers and each hidden layer has 30 hidden nodes.
  You can use different network structure for different atom types.
  For example::

    nodes:
      Si: 30-30
      O: 15-15

* :gray:`regularization`\: (dictionary)
  Regularization setting. Currently, L1 and L2 regularization is available. 
  
  ::

    regularization:
      type: l2 # l1 and l2 is available
      params:
        coeff: 1e-6

* :gray:`use_force`\: (boolean, default: false)
  If :gray:`true`, both energy and force are used for training.

* :gray:`double_precision`\: (boolean, default: true)
  Switch the double precision(:gray:`true`) and single precision(:gray:`false`).

* :gray:`stddev`\: (float, default: 0.3)
  Standard deviation for weights initialization.


Optimization related parameter
------------------------------

* :gray:`method`\: (str, default: Adam)
  Optimization method. You can choose Adam or L-BFGS. 

* :gray:`batch_size`\: (int, default: 64)
  The number of samples in the batch training set.

* :gray:`full_batch`\: (boolean, default: true)
  If :gray:`true`, full batch mode is enabled. 

.. Note::
    In the full_batch mode, :gray:`batch_size` behaves differently. 
    In full_batch mode, the entire dataset 
    must be considered in one iteration, 
    but this often causes out of memory problems. 
    Therefore, in SIMPLE-NN, a batch dataset with the size of 
    :gray:`batch_size` is processed at once, 
    and this process is repeated to perform operations on the entire data set.

* :gray:`total_epoch`\: (int, default: 10000)
  The number of total training epoch.

* :gray:`learning_rate`\: (float, default: 0.0001, :ref:`exponential_dacay-label`)
  Learning rate for gradient descendent based optimization algorithm.

* :gray:`force_coeff` and :gray:`energy_coeff`\: (float, default: 0.1 and 1., :ref:`exponential_dacay-label`)
  Scaling coefficient for force and energy loss.

* :gray:`loss_scale`\: (float, default: 1.)
  Scaling coefficient for the entire loss function.

* :gray:`optimizer`\: (dictionary) additional parameters for user-defined optimizer

Logging & saving related parameters
-----------------------------------
* :gray:`save_interval`\: (int, default: 1000)
  Interval for saving the neural network potential file.

* :gray:`show_interval`\: (int, default: 100)
  Interval for printing RMSE in LOG file.

* :gray:`echeck` and :gray:`fcheck`\: (boolean, default: true, true)
  If :gray:`true`, simple-nn check the selected type of RMSE for the validation set.
  The network is saved when current RMSE is smaller than the RMSE of the previous save point.

* :gray:`break_max`\: (int, default: 10)
  If RMSE of validation set is larger then that of previous save point, :gray:`break_count` increases.
  Optimization process terminated when :gray:`break_count` >= :gray:`break_max`.

* :gray:`print_structure_rmse`\: (boolean, default: false)
  If :gray:`true`, RMSEs for each structure type are also printed in LOG file.


Performance related parameters
------------------------------
* :gray:`inter_op_parallelism_threads` and :gray:`intra_op_parallelism_threads`\: (int, default: 0, 0)
  The number of threads for CPU. Zero means a single thread.

* :gray:`cache`\: (boolean, default: false)
  If :gray:`true`, batch dataset is temporarily saved using caches. 
  Calculation speed may increase but larger memory is needed.


.. _exponential_dacay-label:

Exponential decay
-----------------
Some parameters in neural_network may need to decrease exponentially during the optimization process. 
In those cases, you can use this format instead of float value. More information can be found in 
`Tensorflow homepage`_

.. _Tensorflow homepage: https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay

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
    train(self, user_optimizer=None, aw_modifier=None)

    Args:
        - :gray:`user_optimizer`\: User defined optimizer. 
          Can be set in the script run.py
        - :gray:`aw_modifier`\: scale function for atomic weights.

    Method for optimizing neural network potential.

.. rubric:: Reference

.. [#f1] `J. Behler, M. Parrinello, Phys. Rev. Lett. 98 (2007) 146401`_

.. _J. Behler, M. Parrinello, Phys. Rev. Lett. 98 (2007) 146401: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401