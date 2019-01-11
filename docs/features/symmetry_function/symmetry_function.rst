.. include:: /share.rst

=================
Symmetry function
=================

Introduction
============
simple-nn use atom-centered symmetry function [#f1]_ as a default descriptor vector.
Radial symmetry function G2 and angular symmetry function G4 and G5 are used.


Parameters
==========

feature vector related parameter
--------------------------------
    - :gray:`params`\:
      Defines the name of a text file which contains coefficients list of symmetry function 
      for each atom types::

        params:
          Si: params_Si
          O: params_O 

    - :gray:`compress_outcar`\: (boolean, default: true) 
      If :gray:`true`, VASP OUTCAR file is automatically compressed before handling it.
      This flag does not change the original file. VASP OUTCAR only.

Atomic weight related parameter
-------------------------------
    - :gray:`atomic_weights`\: (dictionary) 
      Dictionary for atomic weights. To use GDF, set this parameter as below::
    
          atomic_weights:
            type: gdf
            params: 
              sigma:
                Si: 0.02

    - :gray:`weight_modifier`\: (dictionary) 
      Dictionary for weight modifier. Detailed setting is like below::

          weight_modifier:
            type: modified sigmoid
            params: 
              Si:
                b: 0.02
                c: 1000.

    Detailed information of GDF usage can be found in this paper [#f2]_,

preprocessing related parameter
-------------------------------
    - :gray:`valid_rate`\: (float, default: 0.1)
      The ratio of validation set relative to entire dataset.

    - :gray:`remain_pickle`\: (boolean, default: false)
      If :gray:`true`, pickle files containing symmetry functions and its derivatives are not
      removed after generating tfrecord files. Currently, we do not support any methods 
      to read tfrecord file externally. Thus, set this parameter :gray:`true` to check 
      the symmetry function of each structure.

tfrecord related parameter
--------------------------
    - :gray:`data_per_tfrecord`\: (int, default: 100)
      The number of structures that is packed into one tfrecord file.

    - :gray:`num_parallel_calls`\: (int, default: 5) 
      The number elements processed in parallel.
      If not specified, elements will be processed sequentially.

inputs
======
To use symmetry function as input vector, you need additional input file 'params_XX' and 'str_list'

**params_XX** contains the coefficients for symmetry functions. XX is an atom type which 
included in the target system. The detailed format of 'param_X' is described in below::

    2 1 0 6.0 0.003214 0.0 0.0
    2 1 0 6.0 0.035711 0.0 0.0
    4 1 1 6.0 0.000357 1.0 -1.0
    4 1 1 6.0 0.028569 1.0 -1.0
    4 1 1 6.0 0.089277 1.0 -1.0

Each parameter indicates (SF means symmetry function) ::

    [type of SF(1)] [atom type index(2)] [cutoff distance(1)] [coefficients for SF(4)]

The number inside the indicates the number of parameters.

First column indicates the type of symmetry function. Currently, G2, G4, and G5 are available.

Second and third column indicates the type index of neighbor atoms which starts from 1. 
For radial symmetry function, only one neighbor atom needed to calculate the symmetry function value, 
thus third parameter is set to zero. For angular symmetry function, two neighbor atoms are needed. 
The order of second and third column do not affect the calculation result.

The fourth column means the cutoff radius for cutoff function.

The remaining parameters are the coefficients applied to each symmetry function.
For radial symmetry function, the fifth and sixth column indicates :math:`\eta` and :math:`\mathrm{R_s}`.
The value in last column is dummy value.
For angular symmetry function, :math:`\eta`, :math:`\zeta`, and :math:`\lambda` are listed in order.

**str_list** contains the location of reference calculation data. The format is described below::

    [ structure_type_1 ]
    /location/of/calculation/data/oneshot_output_file :
    /location/of/calculation/data/MDtrajectory_output_file 100:2000:20

    [ structure_type_2 : 3.0 ]
    /location/of/calculation/data/same_folder_format{1..10}/oneshot_output_file :

You can use the format of `braceexpand`_ to set a path to reference file (like last line).
In addition, you can set the structure type of each data to set the structure weight for each structure type.

.. _braceexpand: https://pypi.org/project/braceexpand/

Methods
=======
.. py:function::
    __init__(self, \
            inputs, \
            descriptor=None, \
            model=None)

    Args:
        - :gray:`inputs`\: (str) Name of the input file.
        - :gray:`descriptor`\: (object) Object of the feature class
        - :gray:`model`\: (object) Object of the model class

    Initiator of Simple-nn class. It takes feature and model object 
    and set the default parameters of simple-nn.

.. py:function::
    generate(self)

    Method for generating symmetry functions and its derivatives.

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

.. rubric:: Reference

.. [#f1] `J. Behler, J. Chem. Phys. 134 (2011) 074106`_

.. _J. Behler, J. Chem. Phys. 134 (2011) 074106: https://aip.scitation.org/doi/10.1063/1.3553717

.. [#f2] `W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790`_

.. _W. Jeong, K. Lee, D. Yoo, D. Lee and S. Han, J. Phys. Chem. C 122 (2018) 22790: https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.8b08063