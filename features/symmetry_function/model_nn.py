import tensorflow as tf
import numpy as np

"""
Neural network model with symmetry function as a descriptor
"""

# TODO: complete the code
# TODO: add the part for selecting the memory device(CPU or GPU)
def _make_model(inputs, atom_types, inp_sizes, hlayers, hnodes, dtype, calc_deriv=False):
    # FIXME: simplify the input parameters
    # FIXME: add the part for regularization (use kwargs?)
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
