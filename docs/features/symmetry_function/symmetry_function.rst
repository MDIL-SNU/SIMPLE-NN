.. include:: /share.rst

=================
Symmetry function
=================

Introduction
============
simple-nn use atom-centered symmetry function[ref] as a default descriptor vector.
Radial symmetry function G2 and angular symmetry function G4 and G5 are used.


Parameters
==========

feature vector related parameter
--------------------------------
    - :gray:`params`\:
      Define the name of text file which contains coefficients list of symmetry function 
      for each atom types::

        params:
          - Si: params_Si
          - O: params_O 

    - :gray:`compress_outcar`\: (boolean, default: true) 
      If :gray:`true`, VASP OUTCAR file is automatically compressed before handling it.
      This flag do not change original file. VASP OUTCAR only.

Atomic weight related parameter
-------------------------------
    - :gray:`atomic_weights`\: (dictionary) 
      Dictionary for atomic weights. To use GDF, set this parameter like below::
    
          atomic_weights:
            - type: gdf
            - params: 

    - :gray:`weight_modifier`\: (dictionary) 
      Dictionary for weight modifier. Detailed setting is like below::

          weight_modifier:
            - type:
            - params: 

    Detailed information to use GDF can be found in ~ [GDF paper]

preprocessing related parameter
-------------------------------
    - :gray:`valid_rate`\: (float, default: 0.1)
      The ratio of validation set from entire dataset.

    - :gray:`remain_pickle`\: (boolean, default: false)
      If :gray:`true`, pickle files contain symmetry functions and its derivatives is not
      removed after generating tfrecord files. Currently, we do not support any methods 
      to read tfrecord file externally. Thus, set this parameter :gray:`true` to check 
      the symmetry function of each structures.

tfrecord related parameter
--------------------------
    - :gray:`data_per_tfrecord`\: (int, default: 100)
      The number of structures that packed into one tfrecord file.

    - :gray:`num_parallel_calls`\: (int, default: 5) 
      Representing the number elements to process in parallel.
      If not specified, elements will be processed sequentially.

inputs
======
For using symmetry function, one need additional input file 'params_XX' and 'str_list'

**params_XX** indicates the coefficients for symmetry functions. XX is atom type which 
included in the target system. Detailed format of 'param_X' is described below::

    2 1 0 6.0 0.003214 0.0 0.0
    2 1 0 6.0 0.035711 0.0 0.0
    4 1 1 6.0 0.000357 1.0 -1.0
    4 1 1 6.0 0.028569 1.0 -1.0
    4 1 1 6.0 0.089277 1.0 -1.0

Each parameter indicates (SF means symmetry function) ::

    [type of SF(1)] [atom type index(2)] [cutoff distance(1)] [coefficients for SF(4)]

The number inside the indicates the number of parameters.

First one indicates the type of symmetry function. Currently G2, G4 and G5 is available.

Second and third indicates the type index of neighbor atoms which starts from 1. 
For radial symmetry function, 1 neighbor atom is need to calculate the symmetry function value. 
Thus, third parameter is set to zero. For angular symmtery function, 2 neighbor atom is needed. 
The order of second and third do not affect to the calculation result.

Fourth one means the cutoff radius for cutoff function.

The remaining parameters are the coefficients applied to each symmetry function.

**str_list** contains the location of reference calculation data. The format is described below::

    /location/of/calculation/data/oneshot_output_file :
    /location/of/calculation/data/MDtrajectory_output_file 100:2000:20
    /location/of/calculation/data/same_folder_format{1..10}/oneshot_output_file :

One can use the format of braceexpand[link] to set a path to reference file (like third line)

Methods
=======
.. py:function::
    __init__(self, \
            inputs, \
            descriptor=None, \
            model=None)

    Args:
        - :gray:`inputs`\: (str) Name of input file.
        - :gray:`descriptor`\: (object) Object of feature class
        - :gray:`model`\: (object) Object of model class

    Initiator of Simple-nn class. It takes feature and model object 
    and set the default parameters of simple-nn.

.. py:function::
    generate(self)

    Initiator of Simple-nn class. It takes feature and model object 
    and set the default parameters of simple-nn.

.. py:function::
    preprocess(self, \
               calc_scale=True, \
               use_force=False, \
               get_atomic_weights=None, \
               **kwargs)

    Args:
        - :gray:`calc_scale`\: (boolean) 
        - :gray:`use_force`\: (boolean) 
        - :gray:`get_atomic_weights`\: (object) Object of model class

    Initiator of Simple-nn class. It takes feature and model object 
    and set the default parameters of simple-nn.