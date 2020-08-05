========
Examples
========

Introduction
============

This section demonstrate SIMPLE-NN with examples. 
Example files are in :code:`SIMPLE-NN/examples/`.
In this example, snapshots from 500K MD trajectory of 
amorphous SiO\ :sub:`2`\  (60 atoms) are used as training set.  

.. Note::

    Since we set the relative path for reference file in :code:`str_list`, 
    You need to move to the directory indicated in each section below to run the examples.

.. _Generate NNP:

Generate NNP
============

To generate NNP using symmetry function and neural network, 
you need three types of input file (input.yaml, str_list, params_XX) 
as described in :doc:`/tutorials/tutorial` section.
The example files except params_Si and params_O are introduced below.
Detail of params_Si and params_O can be found in :doc:`/features/symmetry_function/symmetry_function` section.
Input files introduced in this section can be found in 
:code:`SIMPLE-NN/examples/SiO2/generate_NNP`.

::

    # input.yaml
    generate_features: true
    preprocess: true
    train_model: true
    atom_types:
      - Si
      - O

    symmetry_function:
      params:
        Si: params_Si
        O: params_O
       
    neural_network:
      method: Adam
      nodes: 30-30
      batch_size: 10
      total_iteration: 50000
      learning_rate: 0.001

::

    # str_list
    ../ab_initio_output/OUTCAR_comp ::10

With this input file, SIMPLE-NN calculate feature vectors and its derivatives (:code:`generate_features`), 
generate training/validation dataset (:code:`preprocess`) and optimize the network (:code:`train_model`).
Sample VASP OUTCAR file (the file is compressed to reduce the file size) is in :code:`SIMPLE-NN/examples/SiO2/ab_initio_output`.
In MD trajectory, snapshots are sampled in the interval of 10 MD steps.
In this example, 70 symmetry functions consist of 8 radial symmetry functions per 2-body combination 
and 18 angular symmetry functions per 3-body combination.
Thus, this model uses 70-30-30-1 network for both Si and O. 
The network is optimized by Adam optimizer with the 0.001 of learning rate and batch size is 10. 

Output files can be found in :code:`SIMPLE-NN/examples/SiO2/generate_NNP/outputs`.
In the folder, generated dataset is stored in :code:`data` folder
and execution log and energy/force RMSE are stored in :code:`LOG`. 

Potential test
==============

.. _gen_test_data:

Generate test dataset
---------------------
Generating a test dataset is same as generating a training/validation dataset.
In this example, we use same VASP OUTCAR to generate test dataset.
Input files introduced in this section can be found in 
:code:`SIMPLE-NN/examples/SiO2/generate_test_data`.

::

    # input.yaml
    generate_features: true
    preprocess: true
    train_model: false
    atom_types:
      - Si
      - O

    symmetry_function:
      params:
        Si: params_Si
        O: params_O
      valid_rate: 0.

In this case, :code:`train_model` is set to :code:`false` 
because training process is not required to generate test dataset.
In addition, valid_rate also set to 0.
:code:`str_list` is same as `Generate NNP`_ section.

.. Note::

    To prevent overwriting of the existing training/validation dataset,
    create a new folder and create a test dataset.


.. _test_mode:

Error check
-----------

To check the error for test dataset, use the setting below.
And for running test mode, you need to copy the :code:`train_list` 
file generated in :ref:`gen_test_data` section
to :code:`SIMPLE-NN/examples/SiO2/error_check` and change filename to :code:`test_list`.
Edit the path to data directory in :code:`test_list` file accordingly.
For example, it should be changed from :code:`./data/training_data_0000_to_0006.tfrecord` to :code:`../generate_test_data/data/training_data_0000_to_0006.tfrecord` in this example.
Also, copy :code:`scale_factor` and :code:`params_*` to the current directory.
These files contain information on data set, so you have to carry them with the data set.
Input files introduced in this section can be found in 
:code:`SIMPLE-NN/examples/SiO2/error_check`.

::

    # input.yaml
    generate_features: false
    preprocess: false
    train_model: true
    atom_types:
      - Si
      - O

    symmetry_function:
      params:
        Si: params_Si
        O: params_O
       
    neural_network:
      method: Adam
        nodes: 30-30
      batch_size: 10
      train: false
      test: true
      continue: true

.. Note::
  You need to change the filename from :code:`SAVER_iterationXXXX.*` to :code:`SAVER.*` to use the option :code:`continue: true`
  and modify the checkpoints file (remove '_iterationXXXX' in the text). 
  If you use the option :code:`continue: weights`, 
  change the filename from :code:`potential_saved_iterationXXXX` to :code:`potential_saved`.

After running SIMPLE-NN with the setting above, 
new output file named :code:`test_result` is generated. 
The file is pickle format and you can open this file with python code of below::

    from six.moves import cPickle as pickle

    with open('test_result') as fil:
        res = pickle.load(fil) # For Python 2

    with open('test_result', 'rb') as fil:
        res = pickle.load(fil, encoding='latin1') # For Python 3

In the file, DFT energies/forces, NNP energies/forces are included.

Molecular dynamics
==================
Please check in :doc:`/tutorials/tutorial` section for detailed LAMMPS script writing.


Principal component analysis
============================

SIMPLE-NN provides principal component analysis (PCA) as a method for preprocessing input descriptor vector.
Input descriptor vector, including Behler-type symmetry functions, often has high correlation between components.
In that case, decorrelating input descriptor vector using PCA before feeding it to a machine-learning model can give much faster convergence.

In order to use PCA, add following lines in :code:`input.yaml` when you do preprocess and when you do training and testing.
For detailed descriptions of input parameters, see :ref:`here <models/hdnn/hdnn:PCA-related parameters>`.

.. code:: yaml

   neural_network:
      pca: true
      pca_whiten: true
      pca_min_whiten_level: 1.0e-8

A pickle file named :code:`pca` will be generated during the preprocessing. You need to copy :code:`pca` file to where you run SIMPLE-NN with trained model, just like :code:`scale_factor` file.


Parameter tuning
================

GDF
---
GDF [#f1]_ is used to reduce the force errors of the sparsely sampled atoms. 
To use GDF, you need to calculate the :math:`\rho(\mathbf{G})` 
by adding the following lines to the :code:`symmetry_function` section in :code:`input.yaml`.
SIMPLE-NN supports automatic parameter generation scheme for :math:`\sigma` and :math:`c`.
Use the setting :code:`sigma: Auto` to get a robust :math:`\sigma` and :math:`c` (values are stored in LOG file).
Input files introduced in this section can be found in 
:code:`SIMPLE-NN/examples/SiO2/parameter_tuning_GDF`.

::

    #symmetry_function:
      #continue: true # if individual pickle file is not deleted
      atomic_weights:
        type: gdf
        params:
          sigma: Auto
          # for manual setting
          #  Si: 0.02 
          #  O: 0.02


:math:`\rho(\mathbf{G})` indicates the density of each training point.
After calculating :math:`\rho(\mathbf{G})`, histograms of :math:`\rho(\mathbf{G})^{-1}` 
are also saved as in the file of :code:`GDFinv_hist_XX.pdf`.

.. Note::
  If there is a peak in high :math:`\rho(\mathbf{G})^{-1}` region in the histogram, 
  increasing the Gaussian weight(:math:`\sigma`) is recommended until the peak is removed.
  On the contrary, if multiple peaks are shown in low :math:`\rho(\mathbf{G})^{-1}` region in the histogram,
  reduce :math:`\sigma` is recommended until the peaks are combined. 

In the default setting, the group of :math:`\rho(\mathbf{G})^{-1}` is scaled to have average value of 1. 
The interval-averaged force error with respect to the :math:`\rho(\mathbf{G})^{-1}` 
can be visualized with the following script.


::

    from simple_nn.utils import graph as grp

    grp.plot_error_vs_gdfinv(['Si','O'], 'test_result')

where :code:`test_result` is generated after :ref:`test_mode` as the output file. 
The graph of interval-averaged force errors with respect to the 
:math:`\rho(\mathbf{G})^{-1}` is generated as :code:`ferror_vs_GDFinv_XX.pdf`

.. .. image:: /images/ref_forceerror

If default GDF is not sufficient to reduce the force error of sparsely sampled training points, 
One can use scale function to increase the effect of GDF. In scale function, 
:math:`b` controls the decaying rate for low :math:`\rho(\mathbf{G})^{-1}` and 
:math:`c` separates highly concentrated and sparsely sampled training points.
To use the scale function, add following lines to the :code:`symmetry_function` section in :code:`input.yaml`.

::

    #symmetry_function:
      weight_modifier:
        type: modified sigmoid
        params:
          Si:
            b: 0.02
            c: 3500.
          O:
            b: 0.02
            c: 10000.

For our experience, :math:`b=1.0` and automatically selected :math:`c` shows reasonable results. 
To check the effect of scale function, use the following script for visualizing the 
force error distribution according to :math:`\rho(\mathbf{G})^{-1}`. 
In the script below, :code:`test_result_noscale` is the test result file from the training without scale function and 
:code:`test_result_wscale` is the test result file from the training with scale function.

::

    from simple_nn.utils import graph as grp

    grp.plot_error_vs_gdfinv(['Si','O'], 'test_result_noscale', 'test_result_wscale')




.. [#f1] `W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790`_

.. _W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790: https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.8b08063

Uncertainty Estimation
======================

Replica ensemble [#f2]_ is used to estimate the atomic-resolution uncertainty. 
Please read above paper for details.
We recommend you to make independent directories for each step

.. Note::
  Before following steps, you have prepared :code:`*.pickle` in :code:`path/data/`.
  If not, please run with below options first.

::

    #input.yaml
    generate_feature: true
    preprocess: false
    train_model: false

    symmetry_function:
      remain_pickle: true (default: false)


Step 1. Extract the atomic energy
---------------------------------
Extract the atomic energy that will be used for reference of replicas.
Make :code:`test_list` as described in `Potential test`_ and prepare the :code:`potential_saved`

::

    #input.yaml
    generate_feature: false
    preprocess: false
    train_model: true

    neural_network:
      NNP_to_pickle: true
      test: false
      train: false
      continue: true (or weights)

Step 2. Write the data into tfrecord
------------------------------------
Convert :code:`*.pickles` into :code:`tfrecord` to feed input data during training

::

    #input.yaml
    generate_feature: false
    preprocess: true
    train_model: false

    symmetry_function:
      add_NNP_ref: true
      continue: true

Step 3. Train with atomic energy
--------------------------------
Train model with atomic energy only to speed up (:code:`use_force` and :code:`use_stress` are :code:`false`). Choose a suitable the number of nodes and standard deviation of initial weight. Repeat this step several times by changing the number of nodes.

::

    #input.yaml
    generate_feature: false
    preprocess: false
    train_model: true

    neural_network:
      NNP_to_pickle: false
      use_force: false
      use_stress: false
      nodes: (user's choice)
      test: false
      train: true
      continue: false
      E_loss: 3
      weight_initializer:
        params:
          stddev: (user's choice)

    symmetry_function:
      add_NNP_ref: true
      continue: true

Step 4. Molecular dynamics
--------------------------

.. Note::
  Before this step, you have to compile your LAMMPS with :code:`pair_nn_replica.cpp` and :code:`pair_nn_replica.h`.

LAMMPS can calculate the atomic uncertainty through standard deviation of atomic energies.
Because our NNP do not deal with charged system, atomic uncertainty can be written as atomic charge.
Prepare your data file as charge format and please modify your LAMMPS input as below example.

::

    atom_style  charge
    pair_style  nn/r (# of replica potentials)
    pair_coeff  * * (reference potential) (element1) (element2) ... &
                (replica potential_#1) &
                (replica_potential_#2) &
                ...
    compute     (ID) (group-ID) property/atom q

.. [#f2] `W. Jeong, D. Yoo, K. Lee, J. Jung and S. Han, J. Phys. Chem. Lett. 2020, 11, 6090-6096`_

.. _W. Jeong, D. Yoo, K. Lee, J. Jung and S. Han, J. Phys. Chem. Lett. 2020, 11, 6090-6096: https://pubs.acs.org/doi/10.1021/acs.jpclett.0c01614

