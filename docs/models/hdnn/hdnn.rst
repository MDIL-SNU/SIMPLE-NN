.. include:: /share.rst

===============================
High-dimensional neural network
===============================

Introduction
============
SIMPLE-NN use High-Dimensional Neural Network(HDNN) [#f1]_ as a default machine learning model.

Parameters
==========

Function related parameter
--------------------------

* :gray:`train`\: (boolean, default: true)
  If :gray:`true`, training process proceeds.

* :gray:`test`\: (boolean, default: false)
  If :gray:`true`, predicted energy and forces for test set are calculated.

* :gray:`continue`\: (boolean or string, default: false)
  If :gray:`true`, training process restarts from save file (SAVER.*, checkpoints). 
  If :gray:`continue: weights`, training process restarts from the LAMMPS potential file (potential_saved).

.. Note::
    :gray:`potential_saved` only contains the weights and bias of the network. 
    Thus, hyperparameter used in Adam optimizer is reset with :gray:`continue: weights`.


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
  If negative, early termination scheme is activated (See :gray:`break_max` below).

* :gray:`learning_rate`\: (float, default: 0.0001, :ref:`exponential_dacay-label`)
  Learning rate for gradient descendent based optimization algorithm.

* :gray:`force_coeff` and :gray:`energy_coeff`\: (float, default: 0.1 and 1., :ref:`exponential_dacay-label`)
  Scaling coefficient for force and energy loss.

* :gray:`loss_scale`\: (float, default: 1.)
  Scaling coefficient for the entire loss function.

* :gray:`optimizer`\: (dictionary) additional parameters for user-defined optimizer

Logging & saving related parameters
-----------------------------------
* :gray:`show_interval`\: (int, default: 100)
  Interval for printing RMSE in LOG file.

* :gray:`save_interval`\: (int, default: 1000)
  Interval for saving the neural network potential file.

* :gray:`save_criteria`\: (list, default: [])
  Criteria for saving the neural network potential file. 
  Energy error for validation set (:gray:`v_E`),
  force error for validation set (:gray:`v_F`),
  and force error for validation set for sparsely sampled training points (:gray:`v_F_XX_sparse`) 
  are possible.
  A network is saved only when all values in the criteria are smaller than previous save points.
  If not, :gray:`break_count` is increased (See :gray:`break_max` below).

.. Note::
    In SIMPLE-NN, save conditions(:gray:`save_interval` and :gray:`save_critera`) 
    are checked every multiple of :gray:`show_interval`.
    Thus, it is recommended to set :gray:`save_interval` to multiples of :gray:`show_interval`. 

.. Note::
    Every multiple of :gray:`show_interval`, SIMPLE-NN calculates energies and forces for entire validation set.
    so the process takes a lot of time in general. 
    Thus, small :gray:`show_interval` may slow down the training speed.

* :gray:`break_max`\: (int, default: 10)
  If save criteria is not satisfied in current save points, :gray:`break_count` increases.
  Optimization process is terminated when :gray:`break_count` >= :gray:`break_max`.
  This tag is only activated when total_epoch is negative.

* :gray:`print_structure_rmse`\: (boolean, default: false)
  If :gray:`true`, RMSEs for each structure type are also printed in LOG file.


Performance related parameters
------------------------------
* :gray:`inter_op_parallelism_threads` and :gray:`intra_op_parallelism_threads`\: (int, default: 0, 0)
  The number of threads for CPU. Default is 0, which results the values set to the number of logical cores. 
  The recommended values are the number of physical cores 
  for intra_op_parallelism_threads and the number of sockets for inter_op_parallelism_threads. 
  intra_op_parallelism_threads should be equal to OMP_NUM_THREADS.

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

.. Note::
    If :gray:`continue: true`, :gray:`global_step` (see the link above) of save points is also loaded. 
    Thus, you need to consider the :gray:`global_step` to calculate the values from :gray:`exponential_decay`.
    On the contrary, :gray:`global_step` is reset when :gray:`continue: weights` 

Methods
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

.. [#f1] `J. Behler, M. Parrinello, Phys. Rev. Lett. 98 (2007) 146401`_

.. _J. Behler, M. Parrinello, Phys. Rev. Lett. 98 (2007) 146401: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401