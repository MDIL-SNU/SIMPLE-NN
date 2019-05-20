# SIMPLE-NN
SIMPLE-NN(SNU Interatomic Machine-learning PotentiaL packagE – version Neural Network)

If you use SIMPLE-NN, please cite this article: 

K. Lee, D. Yoo, W. Jeong *et al*., SIMPLE-NN: An efficient package for training and executing neural-network interatomic potentials, *Computer Physics Communications* (2019), https://doi.org/10.1016/j.cpc.2019.04.014.

## Installation
SIMPLE-NN use Tensorflow and mpi4py(optional).
You need to install Tensorflow and mpi4py to use SIMPLE-NN

install Tensorflow: https://www.tensorflow.org/install/

install mpi4py:
```
pip install mpi4py
```

### From pip
```
pip install simple-nn
```

### Install LAMMPS' module
Currently, we support the module for symmetry_function - Neural_network model.
Copy the source code to LAMMPS src directory.
```
cp /directory/of/simple-nn/features/symmetry_function/pair_nn.* /directory/of/lammps/src/
cp /directory/of/simple-nn/features/symmetry_function/symmetry_function.* /directory/of/lammps/src/
```
And compile LAMMPS code.

## Usage
To use SIMPLE-NN, 3 types of files (input.yaml, params_XX, str_list) are required.

### input.yaml
Parameter list to control SIMPLE-NN code is listed in input.yaml. Full parameter list can be found at our online manual(http://mtcg.snu.ac.kr/doc/index.html).
The simplest form of input.yaml is described below:
```YAML
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
  # GDF setting
  #atomic_weights:
  #  type: gdf
  
neural_network:
  method: Adam
  nodes: 30-30
```

### params_XX
params_XX (XX means atom type that is included your target system) indicates the coefficients of symmetry functions.
Each line contains coefficients for one symmetry function. detailed format is described below:

```
2 1 0 6.0 0.003214 0.0 0.0
2 1 0 6.0 0.035711 0.0 0.0
4 1 1 6.0 0.000357 1.0 -1.0
4 1 1 6.0 0.028569 1.0 -1.0
4 1 1 6.0 0.089277 1.0 -1.0
```

First one indicates the type of symmetry function. Currently G2, G4 and G5 is available.

Second and third indicates the type index of neighbor atoms which starts from 1. For radial symmetry function, 1 neighbor atom is need to calculate the symmetry function value. Thus, third parameter is set to zero. For angular symmtery function, 2 neighbor atom is needed. The order of second and third do not affect to the calculation result.

Fourth one means the cutoff radius for cutoff function.

The remaining parameters are the coefficients applied to each symmetry function.

### str_list
str_list contains the location of reference calculation data. The format is described below:

```
/location/of/calculation/data/oneshot_output_file :
/location/of/calculation/data/MDtrajectory_output_file 100:2000:20
/location/of/calculation/data/same_folder_format{1..10}/oneshot_output_file :
``` 

### Script for running SIMPLE-NN
After preparing input.yaml, params_XX and str_list, one can run SIMPLE-NN using the script below:

```python
"""
Run the code below:
    python run.py

run.py:
"""

from simple_nn import Simple_nn
from simple_nn.features.symmetry_function import Symmetry_function
from simple_nn.models.neural_network import Neural_network

model = Simple_nn('input.yaml', 
                   descriptor=Symmetry_function(), 
                   model=Neural_network())
model.run()
```

## Example
In examples folder, one can find MD trajectories of bulk SiO<sub>2</sub>, corresponding input files (input.yaml, params_Si, params_O and str_list) and python script run.py. To use this example, one simply change the location in the 'str_list' file and run 'Python run.py' command.
