import tensorflow as tf
import numpy as np

"""
Neural network model with symmetry function as a descriptor
"""

# TODO: complete the code
# TODO: add the part for selecting the memory device(CPU or GPU)
def _make_model(atom_types, inputs, nodes, dtype, calc_deriv=False):
    # FIXME: simplify the input parameters
    # FIXME: add the part for regularization (use kwargs?)
    models = dict()
    ys = dict()
    dys = dict()

    for item in atom_types:
        nlayers = len(nodes[item])
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nodes[item][1], activation='sigmoid', \
                                        input_dim=nodes[item][0], dtype=dtype))
        for i in range(2, nlayers):
           model.add(tf.keras.layers.Dense(nodes[item][i], activation='sigmoid', dtype=dtype))
        model.add(tf.keras.layers.Dense(1, activation='linear', dtype=dtype))

        models[item] = model
        ys[item] = models[item](inputs[item])

        if calc_deriv:
            dys[item] = tf.gradients(ys[item], inputs[item])
        else:
            dys[item] = None

    return model, ys, dys

def _calc_output(atom_types, ys, segsum_ids, \
                 calc_force=False, dys=None, dsyms=None, partition_id=None):
    energy = force = 0
    
    for item in atom_types:
        energy += tf.segment_sum(ys[item], segsum_ids[item])

        if calc_force:
            tmp_force = dsyms[item] * \
                        tf.expand_dims(\
                            tf.expand_dims(dys[item], axis=2), 
                            axis=3)
            tmp_force = tf.reduce_sum(\
                            tf.segment_sum(tmp_force, segsum_ids[item]),
                            axis=1)
            force -= tf.dynamic_partition(tf.reshape(tmp_force, [-1, 3]), \
                                          partition_id, 2)[0]

    return energy, force

def _get_loss(ref_energy, calced_energy, atom_num, \
              calc_force=False, ref_force=None, calced_force=None, force_coeff=0.03, \
              use_gdf=False, gdf_values=None):

    e_loss = tf.reduce_mean(tf.square(calced_energy - ref_energy) / atom_num)

    if calc_force:
        f_loss = tf.square(calced_force - ref_force)
        if use_gdf:
            f_loss *= gdf_values
        f_loss = tf.reduce_mean(f_loss) * force_coeff

    return e_loss, f_loss

def _make_optimizer(loss, method='Adam', lossscale=1., **kwargs):
    if method == 'L-BFGS-B':
        optim = tf.contrib.opt.ScipyOptimizerInterface(lossscale*loss, method=method, options=kwargs)
    elif method == 'Adam':
        learning_rate = tf.train.exponential_decay(**kwargs)
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(loss*lossscale)

    return optim


def run(sess, inputs):
    inputs = dict()
    for item in atom_types:
        inputs = tf.placeholder(tf.float64, [None, ])

    models, ys, dys = _make_model(atom_types, inputs, )
    energy, force = _calc_output()
    e_loss, f_loss = _get_loss()
    optim = _make_optimizer()



    return 0
    