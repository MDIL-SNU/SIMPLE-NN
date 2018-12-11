# SPINN
SPINN(SNU Package for Interatomic Neural Network potential)

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
Parameter list to control SIMPLE-NN code is listed in input.yaml. Full parameter list can be found at our online manual().
The simplest form of input.yaml is described below:
```
# input.yaml

```

### params_XX
params_XX (XX means atom type that is included your target system)

### str_list
str_list contains the reference 

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
In examples folder, one can find MD trajectories of bulk SiO<sub>2</sub>, corresponding input files (input.yaml, params_Si, params_O and str_list) and python script run.py.
One can easily test SIMPLE-NN code with this example.
