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

## Usage

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

input.yaml is the input parameter file for run SIMPLE-NN
