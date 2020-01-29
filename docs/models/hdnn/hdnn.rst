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

* :code:`train`\: (boolean, default: true)
  If :code:`true`, training process proceeds.

* :code:`test`\: (boolean, default: false)
  If :code:`true`, predicted energy and forces for test set are calculated.

* :code:`continue`\: (boolean or string, default: false)
  If :code:`true`, training process restarts from save file (SAVER.*, checkpoints). 
  If :code:`continue: weights`, training process restarts from the LAMMPS potential file (potential_saved).

.. Note::
    :code:`potential_saved` only contains the weights and bias of the network. 
    Thus, hyperparameter used in Adam optimizer is reset with :code:`continue: weights`.


Network related parameter
-------------------------
* :code:`nodes`\: (str or dictionary, default: 30-30)
  String value to indicate the network architecture.
  30-30 means 2 hidden layers and each hidden layer has 30 hidden nodes.
  You can use different network structure for different atom types.
  For example::

    nodes:
      Si: 30-30
      O: 15-15

* :code:`regularization`\: (dictionary)
  Regularization setting. Currently, L1 and L2 regularization is available. 
  
  ::

    regularization:
      type: l2 # l1 and l2 is available
      params:
        coeff: 1e-6

* :code:`use_force`\: (boolean, default: false)
  If :code:`true`, both energy and force are used for training.

* :code:`force_coeff`\: (float, default: 0.1)
  Ratio of force loss to energy loss in total loss

* :code:`use_stress`\: (boolean, default: false)
  If :code:`true`, both energy and stress are used for training.
  (Caution : The unit of stress RMSE written in LOG file is kbar.)

* :code:`stress_coeff`\: (float, default: 0.00001)
  Ratio of stress loss to energy loss in total loss

* :code:`stddev`\: (float, default: 0.3)
  Standard deviation for weights initialization.


PCA-related parameters
----------------------

* :code:`pca`: When set to true, PCA is applied to the input descriptor vector.

  **Default:**

  .. code:: yaml

     pca: false

* :code:`pca_whiten`: When set to true, PCA-transformed vector is whitened (the variances of principal components are normalized to unity).
  The whitening is different than original scheme.
  See :ref:`below <pca_min_whiten_level>` for more details on the difference.

  **Default:**

  .. code:: yaml

     pca_whiten: true

.. _pca_min_whiten_level:

* :code:`pca_min_whiten_level`: This option can be used to suppress whitening of principal components
  with small variances. Originally, whitening normalize variances of all principal components to unity:

  .. math::

     PC_{i,\text{whiten}}=\frac{PC_i}{\sqrt{\text{Var}_i}}

  A small constant :math:`a` is added to the variance to suppress principal components with small variances:

  .. math::

     PC_{i,\text{whiten}}=\frac{PC_i}{\sqrt{\text{Var}_i + a}}

  In practice, this can be used to reduce the generalization error.

  **Default:**

  .. code:: yaml

     pca_min_whiten_level: 1.0e-8


Optimization related parameter
------------------------------

* :code:`method`\: (str, default: Adam)
  Optimization method. You can choose Adam or L-BFGS. 

* :code:`batch_size`\: (int, default: 64)
  The number of samples in the batch training set.

* :code:`full_batch`\: (boolean, default: true)
  If :code:`true`, full batch mode is enabled. 

.. Note::
    In the full_batch mode, :code:`batch_size` behaves differently. 
    In full_batch mode, the entire dataset 
    must be considered in one iteration, 
    but this often causes out of memory problems. 
    Therefore, in SIMPLE-NN, a batch dataset with the size of 
    :code:`batch_size` is processed at once, 
    and this process is repeated to perform operations on the entire data set.

* :code:`total_iteration`\: (int, default: 10000)
  The number of total training iteration.
  If negative, early termination scheme is activated (See :code:`break_max` below).

* :code:`learning_rate`\: (float, default: 0.0001, :ref:`exponential_dacay-label`)
  Learning rate for gradient descendent based optimization algorithm.

* :code:`force_coeff` and :code:`energy_coeff`\: (float, default: 0.1 and 1., :ref:`exponential_dacay-label`)
  Scaling coefficient for force and energy loss.

* :code:`loss_scale`\: (float, default: 1.)
  Scaling coefficient for the entire loss function.

* :code:`optimizer`\: (dictionary) additional parameters for user-defined optimizer

Logging & saving related parameters
-----------------------------------
* :code:`show_interval`\: (int, default: 100)
  Interval for printing RMSE in LOG file.

* :code:`save_interval`\: (int, default: 1000)
  Interval for saving the neural network potential file.

* :code:`save_criteria`\: (list, default: [])
  Criteria for saving the neural network potential file. 
  Energy error for validation set (:code:`v_E`),
  force error for validation set (:code:`v_F`),
  and force error for validation set for sparsely sampled training points (:code:`v_F_XX_sparse`) 
  are possible.
  A network is saved only when all values in the criteria are smaller than previous save points.
  If not, :code:`break_count` is increased (See :code:`break_max` below).

.. Note::
    In SIMPLE-NN, save conditions(:code:`save_interval` and :code:`save_critera`) 
    are checked every multiple of :code:`show_interval`.
    Thus, it is recommended to set :code:`save_interval` to multiples of :code:`show_interval`. 

.. Note::
    Every multiple of :code:`show_interval`, SIMPLE-NN calculates energies and forces for entire validation set.
    so the process takes a lot of time in general. 
    Thus, small :code:`show_interval` may slow down the training speed.

* :code:`break_max`\: (int, default: 10)
  If save criteria is not satisfied in current save points, :code:`break_count` increases.
  Optimization process is terminated when :code:`break_count` >= :code:`break_max`.
  This tag is only activated when total_iteration is negative.

* :code:`print_structure_rmse`\: (boolean, default: false)
  If :code:`true`, RMSEs for each structure type are also printed in LOG file.


Performance related parameters
------------------------------
* :code:`inter_op_parallelism_threads` and :code:`intra_op_parallelism_threads`\: (int, default: 0, 0)
  The number of threads for CPU. Default is 0, which results the values set to the number of logical cores. 
  The recommended values are the number of physical cores 
  for intra_op_parallelism_threads and the number of sockets for inter_op_parallelism_threads. 
  intra_op_parallelism_threads should be equal to OMP_NUM_THREADS.

* :code:`cache`\: (boolean, default: false)
  If :code:`true`, batch dataset is temporarily saved using caches. 
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
    If :code:`continue: true`, :code:`global_step` (see the link above) of save points is also loaded. 
    Thus, you need to consider the :code:`global_step` to calculate the values from :code:`exponential_decay`.
    On the contrary, :code:`global_step` is reset when :code:`continue: weights` 

Methods
=======
.. py:function::
    __init__(self)

    Initiator of Neural_network class. 

.. py:function::
    train(self, user_optimizer=None, aw_modifier=None)

    Args:
        - :code:`user_optimizer`\: User defined optimizer. 
          Can be set in the script run.py
        - :code:`aw_modifier`\: scale function for atomic weights.

    Method for optimizing neural network potential.

.. rubric:: References

.. [#f1] `J. Behler, M. Parrinello, Phys. Rev. Lett. 98 (2007) 146401`_

.. _J. Behler, M. Parrinello, Phys. Rev. Lett. 98 (2007) 146401: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401
