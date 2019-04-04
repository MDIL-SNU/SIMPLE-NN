from simple_nn import Simple_nn
from simple_nn.features.symmetry_function import Symmetry_function
from simple_nn.models.neural_network import Neural_network

model = Simple_nn('input.yaml', descriptor=Symmetry_function(), model=Neural_network())
model.run()
