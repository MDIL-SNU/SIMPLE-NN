import sys
#import tensorflow as tf
sys.path.insert(0, '/data/khlee1992/SPINN')

from simple_nn import Simple_nn
from simple_nn.features.symmetry_function import Symmetry_function
from simple_nn.models.neural_network import Neural_network

#user_optim = tf.train.AdagradOptimizer

model = Simple_nn('input.yaml', descriptor=Symmetry_function(), model=Neural_network())
#model.run(user_optimizer=user_optim)
model.run()
