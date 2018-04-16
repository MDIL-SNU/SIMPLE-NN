import tensorflow as tf
import numpy as np

"""
Neural network model with symmetry function as a descriptor
"""

def _make_model(inputs, atom_types, inp_sizes, hlayers, hnodes, dtype, calc_deriv=False):
    models = dict()
    ys = dict()
    dys = dict()

    for item in atom_types:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(hnodes[item][0], activation='sigmoid', \
                                        input_dim=inp_sizes[item], dtype=dtype))
        for i in range(1, hlayers[item]):
           model.add(tf.keras.layers.Dense(hnodes[item][i], activation='sigmoid', dtype=dtype))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid', dtype=dtype))

        models[item] = model
        ys[item] = models[item](inputs[item])

        if calc_deriv:
            dys[item] = tf.gradients(ys[item], inputs[item])
        else:
            dys[item] = None

    return model, ys, dys

def _calc_output(atom_types, ys, segsum_ids, \
                 calc_force=False, dys=None, dsyms=None, partition_id=None):
    energy = 0
    force = 0
    return energy, force

#def _get_loss():

