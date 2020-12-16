import tensorflow as tf
import numpy as np
import random
import six
import os
from six.moves import cPickle as pickle
import collections
import functools
import timeit
import copy
from ..utils import _make_data_list, pickle_load, _generate_gdf_file, modified_sigmoid, memory, repeat, read_lammps_potential
from ..utils.lbfgs import L_BFGS
from tqdm import tqdm
from ase import units
#from tensorflow.python.client import timeline

"""
Neural network model with symmetry function as a descriptor
"""

# TODO: add the part for selecting the memory device(CPU or GPU)
# TODO: BFGS support
# TODO: add regularization
class Neural_network(object):
    """
    Class for training neural network potential.
    """
    def __init__(self):
        self.parent = None
        self.key = 'neural_network'
        self.default_inputs = {'neural_network':
                                  {
                                      # Function related
                                      'train': True,
                                      'test': False,
                                      'continue': False,

                                      # Network related
                                      'nodes': '30-30',
                                      'regularization': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'use_force': True,
                                      'use_stress': False,
                                      'double_precision': True,
                                      'weight_initializer': {
                                          'type': 'truncated normal',
                                          'params': {
                                              'stddev': 0.3,
                                          },
                                      },
                                      'acti_func': 'sigmoid',
                                      'dropout': None,

                                      # Optimization related
                                      'method': 'Adam',
                                      'batch_size': 64,
                                      'full_batch': False,
                                      'total_iteration': 10000,
                                      'learning_rate': 0.0001,
                                      'stress_coeff': 0.000001,
                                      'force_coeff': 0.1, 
                                      'energy_coeff': 1.,
                                      'loss_scale': 1.,
                                      'optimizer': dict(),

                                      # Loss function related
                                      'E_loss': 0,
                                      'F_loss': 1,

                                      # Logging & saving related
                                      'save_interval': 1000,
                                      'show_interval': 100,
                                      'save_criteria': [],
                                      'break_max': 10,
                                      'print_structure_rmse': False,

                                      # Performace related
                                      'inter_op_parallelism_threads': 0,
                                      'intra_op_parallelism_threads': 0,
                                      'cache': False,
                                      'pca': False,
                                      'pca_whiten': True,
                                      'pca_min_whiten_level': 1e-8,

                                      # Write atomic energies to pickle
                                      'NNP_to_pickle': False,
                                  }
                              }
        self.inputs = dict()
        self.global_step = tf.Variable(0, trainable=False)
        self.increment_global_step = tf.assign(self.global_step, self.global_step+1)
        self.train_data_list = './train_list'
        self.valid_data_list = './valid_list'
        self.test_data_list = './test_list'


    def _set_params(self, feature_name):
        self.inp_size = dict()
        self.params = dict()

        for item in self.parent.inputs['atom_types']:
            self.params[item] = list()

            with open(self.parent.inputs[feature_name]['params'][item]) as fil:
                for line in fil:
                    tmp = line.split()
                    self.params[item] += [list(map(float, tmp))]

            # Since it is only used to write lammps potential, it does not need degree to radian conversion.
            self.params[item] = np.array(self.params[item])
            self.inp_size[item] = self.params[item].shape[0]


    def _set_scale_parameter(self, scale_file):
        self.scale = pickle_load(scale_file)
        if self.inputs['pca']:
            if not os.path.exists("./pca"):
                err = "File not found: './pca'. PCA components must be calculated in the preprocess part."
                self.parent.logfile.write("Error: {:}\n".format(err))
                raise OSError(err)
            self.pca = pickle_load("./pca")
        # TODO: add the check code for valid scale file

    def _set_gdf_parameters(self, atomic_weights_file, modifier=None):
        atomic_weights = pickle_load(atomic_weights_file)
        self.gdf_scale = dict()
        for item in self.parent.inputs['atom_types']:
            if modifier != None and callable(modifier[item]):
                atomic_weights[item][:,0] = modifier[item](atomic_weights[item][:,0])
            self.gdf_scale[item] = np.mean(atomic_weights[item][:,0])


    def _make_model(self):
        self.models = dict()
        self.ys = dict()
        self.dys = dict()

        if self.inputs['double_precision']:
            dtype = tf.float64
        else:
            dtype = tf.float32

        if self.inputs['weight_initializer']['type'] == 'truncated normal':
            initializer = tf.initializers.truncated_normal(
                    stddev=self.inputs['weight_initializer']['params']['stddev'], dtype=dtype)
        elif self.inputs['weight_initializer']['type'] == 'xavier normal':
            initializer = tf.initializers.variance_scaling(scale=1.0, mode='fan_avg',
                    distribution="normal", dtype=dtype)
        elif self.inputs['weight_initializer']['type'] == 'he normal':
            initializer = tf.initializers.variance_scaling(scale=2.0, mode='fan_in',
                    distribution="normal", dtype=dtype)
        else:
            raise NotImplementedError("Not implemented weight initializer type!")

        # TODO: input validation for stddev.
        dense_basic_setting = {
            'dtype': dtype,
            'kernel_initializer': initializer,
            'bias_initializer': initializer,
        }
        dense_last_setting = copy.deepcopy(dense_basic_setting)

        if self.inputs['regularization']['type'] is not None:
            if self.inputs['regularization']['type'] == 'l2':
                coeff = self.inputs['regularization']['params'].get('coeff', 1e-6)
                dense_basic_setting['kernel_regularizer'] = tf.keras.regularizers.l2(l=coeff)
                dense_basic_setting['bias_regularizer'] = tf.keras.regularizers.l2(l=coeff)
                dense_last_setting['kernel_regularizer'] = tf.keras.regularizers.l2(l=coeff)
            elif self.inputs['regularization']['type'] == 'l1':
                coeff = self.inputs['regularization']['params'].get('coeff', 1e-6)
                dense_basic_setting['kernel_regularizer'] = tf.keras.regularizers.l1(l=coeff)
                dense_basic_setting['bias_regularizer'] = tf.keras.regularizers.l1(l=coeff)
                dense_last_setting['kernel_regularizer'] = tf.keras.regularizers.l1(l=coeff)
            else:
                raise NotImplementedError("Not implemented regularizer type!")

        if self.inputs['continue'] == 'weights':
            saved_weights = read_lammps_potential('potential_saved')

        #acti_func = 'elu'
        #acti_func = 'sigmoid'
        #acti_func = 'tanh'
        #acti_func = 'relu'

        self.nodes = dict()
        for item in self.parent.inputs['atom_types']:
            if isinstance(self.inputs['nodes'], collections.Mapping):
                nodes = list(map(int, self.inputs['nodes'][item].split('-')))
            else:
                nodes = list(map(int, self.inputs['nodes'].split('-')))
            nlayers = len(nodes)

            # Check if network size is the same as that of potential read.
            if self.inputs['continue'] == 'weights':
                mismatch = False
                if (self.inp_size[item], nodes[0]) != saved_weights[item][0].shape:
                    mismatch = True
                if not mismatch:
                    for i in range(1, nlayers):
                        if (nodes[i-1], nodes[i]) != saved_weights[item][2*i+0].shape:
                            mismatch = True
                            break
                if mismatch:
                    err = "The network size given as input does not match that of potential read."
                    self.parent.logfile.write("Error: {:}\n".format(err))
                    raise ValueError(err)

            model = tf.keras.models.Sequential()

            if self.inputs['continue'] == 'weights':
                dense_basic_setting['kernel_initializer'] = tf.constant_initializer(saved_weights[item][0])
                dense_basic_setting['bias_initializer'] = tf.constant_initializer(saved_weights[item][1])

            # Input dimension to the first layer can be changed due to truncation of principal components.
            if self.inputs['pca']:
                input_dim = self.pca[item][0].shape[1]
            else:
                input_dim = self.inp_size[item]
            model.add(tf.keras.layers.Dense(nodes[0], activation=self.inputs['acti_func'], \
                                            input_dim=input_dim,
                                            #kernel_initializer=tf.initializers.random_normal(stddev=1./self.inp_size[item], dtype=dtype),
                                            #use_bias=False,
                                            **dense_basic_setting))

            for i in range(1, nlayers):
                if self.inputs['continue'] == 'weights':
                    dense_basic_setting['kernel_initializer'] = tf.constant_initializer(saved_weights[item][2*i+0])
                    dense_basic_setting['bias_initializer'] = tf.constant_initializer(saved_weights[item][2*i+1])

                model.add(tf.keras.layers.Dense(nodes[i], activation=self.inputs['acti_func'],
                                                #kernel_initializer=tf.initializers.random_normal(stddev=1./nodes[i-1], dtype=dtype),
                                                #use_bias=False,
                                                **dense_basic_setting))
                if self.inputs['dropout'] is not None:
                    model.add(tf.keras.layers.Dropout(self.inputs['dropout']))

            if self.inputs['continue'] == 'weights':
                dense_last_setting['kernel_initializer'] = tf.constant_initializer(saved_weights[item][-2])
                dense_last_setting['bias_initializer'] = tf.constant_initializer(saved_weights[item][-1])

            model.add(tf.keras.layers.Dense(1, activation='linear',
                                            #kernel_initializer=tf.initializers.random_normal(stddev=1./nodes[-1], dtype=dtype),
                                            #bias_initializer=tf.initializers.random_normal(stddev=0.1, dtype=dtype),
                                            **dense_last_setting))

            nodes.append(1)
            self.nodes[item] = nodes

            self.models[item] = model
            self.ys[item] = self.models[item](self.next_elem['x_'+item])

            # Regularization losses of tf.keras.layers.Dense is not automatically added
            # to tf.GraphKeys.REGULARIZATION_LOSSES, so we add them manually.
            # To prevent potential bugs when the behaviour of tf.keras.layers.Dense get changed,
            # we check if losses are already in the graph before adding them.
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            for loss in model.losses:
                if loss not in reg_losses:
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)

            if self.inputs['use_force'] or self.inputs['use_stress']:
                self.dys[item] = tf.gradients(self.ys[item], self.next_elem['x_'+item])[0]
            else:
                self.dys[item] = None


    def _calc_output(self):
        self.E = self.F = self.S = 0
        self.atom_E = dict()

        for item in self.parent.inputs['atom_types']:
            self.atom_E[item] = tf.cast(0., tf.float64)

        for i, item in enumerate(self.parent.inputs['atom_types']):
            zero_cond = tf.equal(tf.reduce_sum(self.next_elem['N_'+item]), 0)

            self.E += tf.cond(zero_cond,
                              lambda: tf.cast(0., tf.float64),
                              lambda: tf.sparse_segment_sum(self.ys[item], self.next_elem['sparse_indices_'+item], self.next_elem['seg_id_'+item],
                                            num_segments=self.next_elem['num_seg'])[1:])

            if self.inputs['E_loss'] == 3 or self.inputs['NNP_to_pickle']:
                self.atom_E[item] += tf.cond(zero_cond,
                                  lambda: tf.cast(0., tf.float64),
                                  lambda: self.ys[item])

            if self.inputs['use_force']:
                tmp_force = self.next_elem['dx_'+item] * \
                            tf.expand_dims(\
                                tf.expand_dims(self.dys[item], axis=2),
                                axis=3)
                tmp_force = tf.reduce_sum(\
                                tf.sparse_segment_sum(tmp_force, self.next_elem['sparse_indices_'+item], self.next_elem['seg_id_'+item],
                                                      num_segments=self.next_elem['num_seg'])[1:],
                                axis=1)
                self.F -= tf.cond(zero_cond,
                                  lambda: tf.cast(0., tf.float64),
                                  lambda: tf.dynamic_partition(tf.reshape(tmp_force, [-1,3]),
                                                               self.next_elem['partition'], 2)[1])

            if self.inputs['use_stress']:
                tmp_stress = self.next_elem['da_'+item] * \
                             tf.expand_dims(\
                                 tf.expand_dims(self.dys[item], axis=2),
                                     axis=3)
                tmp_stress = tf.cond(zero_cond,
                                     lambda: tf.cast(0., tf.float64) * tmp_stress,
                                     lambda: tf.sparse_segment_sum(tmp_stress, self.next_elem['sparse_indices_'+item], self.next_elem['seg_id_'+item],
                                                                num_segments=self.next_elem['num_seg'])[1:])
                self.S -= tf.reduce_sum(tmp_stress, axis=[1,2])/units.GPa*10

    def _get_loss(self, use_gdf=False, atomic_weights=None):
        if self.inputs['E_loss'] == 1:
            self.e_loss = tf.square((self.next_elem['E'] - self.E) / self.next_elem['tot_num']) * self.next_elem['tot_num']
        elif self.inputs['E_loss'] == 2:
            self.e_loss = tf.square(self.next_elem['E'] - self.E)
        elif self.inputs['E_loss'] == 3:
            each_atom_idx = tf.dynamic_partition(self.next_elem['sparse_indices_'],\
                                                 tf.reshape(self.next_elem['atom_idx'],[-1]),\
                                                 len(self.parent.inputs['atom_types']) + 1)[1:]
            self.e_loss = [tf.square(self.atom_E[item] - self.next_elem['NNP_E_'+item]) for item in self.parent.inputs['atom_types']]
            self.e_loss = tf.dynamic_stitch(each_atom_idx, self.e_loss)
            self.e_loss = tf.sparse_segment_mean(self.e_loss, self.next_elem['sparse_indices_'], self.next_elem['seg_id_'],
                                                num_segments=self.next_elem['num_seg'])[1:]
        else:
            self.e_loss = tf.square((self.next_elem['E'] - self.E) / self.next_elem['tot_num'])

        self.sw_e_loss = self.e_loss * self.next_elem['struct_weight']
        self.e_loss = tf.reshape(self.e_loss, [-1])
        self.str_e_loss = tf.unsorted_segment_mean(self.e_loss, self.next_elem['struct_ind'], tf.size(self.next_elem['struct_type_set']))
        self.str_e_loss = tf.reshape(self.str_e_loss, [-1])
        self.e_loss = tf.reduce_mean(self.e_loss)
        self.sw_e_loss = tf.reduce_mean(self.sw_e_loss)
        self.total_loss = self.sw_e_loss * self.energy_coeff

        self.str_num_batch_atom = tf.reshape(tf.unsorted_segment_sum(self.next_elem['tot_num'], self.next_elem['struct_ind'], tf.size(self.next_elem['struct_type_set'])), [-1])
        if self.inputs['use_force']:
            self.f_loss = tf.reshape(tf.square(self.next_elem['F'] - self.F), [-1, 3])
            ind = tf.reshape(repeat(self.next_elem['struct_ind'],
                         tf.cast(tf.reshape(self.next_elem['tot_num'], shape=[-1]), tf.int32)), [-1])
            self.str_f_loss = tf.unsorted_segment_mean(self.f_loss, ind, tf.size(self.next_elem['struct_type_set']))
            self.str_f_loss = tf.reduce_mean(self.str_f_loss, axis=1)
            if self.parent.descriptor.inputs['atomic_weights']['type'] is not None:
                self.aw_f_loss = self.f_loss * self.next_elem['atomic_weights']

                if self.inputs['F_loss'] == 1:
                    self.f_loss = tf.reduce_mean(self.f_loss)
                    self.aw_f_loss = tf.reduce_mean(self.aw_f_loss)
                else:
                    self.f_loss = tf.reduce_mean(tf.sparse_segment_sum(self.f_loss, self.next_elem['sparse_indices_'], self.next_elem['seg_id_'],
                                                 num_segments=self.next_elem['num_seg'])[1:])
                    self.aw_f_loss = tf.reduce_mean(tf.sparse_segment_sum(self.aw_f_loss, self.next_elem['sparse_indices_'], self.next_elem['seg_id_'],
                                                    num_segments=self.next_elem['num_seg'])[1:])

                self.total_loss += self.aw_f_loss * self.force_coeff
            else:
                if self.inputs['F_loss'] == 1:
                    self.f_loss = tf.reduce_mean(self.f_loss)
                else:
                    self.f_loss = tf.reduce_mean(tf.sparse_segment_sum(self.f_loss, self.next_elem['sparse_indices_'], self.next_elem['seg_id_'],
                                                 num_segments=self.next_elem['num_seg'])[1:])
                self.total_loss += self.f_loss * self.force_coeff

        if self.inputs['use_stress']:
            self.ax_s_loss = tf.square(self.next_elem['S'] - self.S)
            self.s_loss = tf.reduce_mean(self.ax_s_loss, axis=1, keepdims=True)
            self.sw_s_loss = self.s_loss * self.next_elem['struct_weight']
            self.s_loss = tf.reshape(self.s_loss,[-1])
            self.str_s_loss = tf.unsorted_segment_mean(self.s_loss, self.next_elem['struct_ind'], tf.size(self.next_elem['struct_type_set']))
            self.str_s_loss = tf.reshape(self.str_s_loss, [-1])
            self.s_loss = tf.reduce_mean(self.s_loss)
            self.sw_s_loss = tf.reduce_mean(self.sw_s_loss)
            self.total_loss += self.sw_s_loss * self.stress_coeff

        if self.inputs['regularization']['type'] is not None:
            # FIXME: regularization_loss, which is float32, is casted into float64.
            self.total_loss += tf.cast(tf.losses.get_regularization_loss(), tf.float64)

    def _make_optimizer(self, user_optimizer=None):
        final_loss = self.inputs['loss_scale']*self.total_loss

        if isinstance(self.inputs['learning_rate'], collections.Mapping):
            exponential_decay_inputs = copy.deepcopy(self.inputs['learning_rate'])
            exponential_decay_inputs['learning_rate'] = tf.constant(exponential_decay_inputs['learning_rate'], tf.float64)
            self.learning_rate = tf.train.exponential_decay(global_step=self.global_step, **exponential_decay_inputs)
        else:
            self.learning_rate = tf.constant(self.inputs['learning_rate'], tf.float64)

        if self.inputs['method'] == 'L-BFGS':
            self.optim = tf.train.GradientDescentOptimizer(learning_rate=1.)
            self.compute_grad = self.optim.compute_gradients(final_loss)
            self.grad_and_vars = [[None, item[1]] for item in self.compute_grad]
            self.flat_grad = tf.reshape(tf.concat([tf.reshape(item[0], [-1]) for item in self.compute_grad], axis=0), [-1, 1])
            self.minim = self.optim.minimize(final_loss, global_step=self.global_step)

        elif self.inputs['method'] == 'Adam':
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                name='Adam', **self.inputs['optimizer'])
            self.compute_grad = self.optim.compute_gradients(final_loss)
            self.grad_and_vars = [[None, item[1]] for item in self.compute_grad]
            self.flat_grad = tf.reshape(tf.concat([tf.reshape(item[0], [-1]) for item in self.compute_grad], axis=0), [-1, 1])
            self.minim = self.optim.minimize(final_loss, global_step=self.global_step)
        else:
            if user_optimizer != None:
                self.optim = user_optimizer(learning_rate=self.learning_rate,
                                            name='user_optim', **self.inputs['optimizer'])
                self.compute_grad = self.optim.compute_gradients(final_loss)
                self.grad_and_vars = [[None, item[1]] for item in self.compute_grad]
                self.flat_grad = tf.reshape(tf.concat([tf.reshape(item[0], [-1]) for item in self.compute_grad], axis=0), [-1, 1])
                self.minim = self.optim.minimize(final_loss, global_step=self.global_step)
                #self.optim = user_optimizer
            else:
                raise ValueError


    def _get_decay_param(self, param):
        """
        get tf.exponential_decay or simple float
        """
        if isinstance(param, collections.Mapping):
            tmp_param = param.copy()
            tmp_param['learning_rate'] = tf.constant(tmp_param['learning_rate'], dtype=tf.float64)
            return tf.train.exponential_decay(global_step=self.global_step, **tmp_param)
        else:
            return tf.constant(param, dtype=tf.float64)

    def _generate_lammps_potential(self, sess):
        # TODO: get the parameter info from initial batch generting processs
        atom_type_str = ' '.join(self.parent.inputs['atom_types'])

        filename = './potential_saved_iteration{}'.format(sess.run(self.global_step)+1)
        FIL = open(filename, 'w')
        FIL.write('ELEM_LIST ' + atom_type_str + '\n\n')

        for item in self.parent.inputs['atom_types']:
            FIL.write('POT {} {}\n'.format(item, np.max(self.params[item][:,3])))
            FIL.write('SYM {}\n'.format(len(self.params[item])))

            for ctem in self.params[item]:
                tmp_types = self.parent.inputs['atom_types'][int(ctem[1])-1]
                if int(ctem[0]) > 3:
                    tmp_types += ' {}'.format(self.parent.inputs['atom_types'][int(ctem[2])-1])
                if len(ctem) != 7:
                    raise ValueError("params file must have lines with 7 columns.")

                FIL.write('{} {} {} {} {} {}\n'.\
                    format(int(ctem[0]), ctem[3], ctem[4], ctem[5], ctem[6], tmp_types))

            FIL.write('scale1 {}\n'.format(' '.join(self.scale[item][0,:].astype(np.str))))
            FIL.write('scale2 {}\n'.format(' '.join(self.scale[item][1,:].astype(np.str))))

            weights = sess.run(self.models[item].weights)
            nlayers = len(self.nodes[item])
            # An extra linear layer is used for PCA transformation.
            if self.inputs['pca']:
                nodes = [self.pca[item][0].shape[1]] + self.nodes[item]
                joffset = 1
            else:
                nodes = self.nodes[item]
                joffset = 0
            FIL.write('NET {} {}\n'.format(len(nodes)-1, ' '.join(map(str, nodes))))

            # PCA transformation layer.
            if self.inputs['pca']:
                FIL.write('LAYER 0 linear PCA\n')
                pca_mat = np.copy(self.pca[item][0])
                pca_mean = np.copy(self.pca[item][2])
                if self.inputs['pca_whiten']:
                    pca_mat /= self.pca[item][1].reshape([1, -1])
                    pca_mean /= self.pca[item][1]

                for k in range(nodes[0]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(pca_mat[:,k].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, -pca_mean[k]))

            for j in range(nlayers):
                # FIXME: add activation function type if new activation is added
                if j == nlayers-1:
                    acti = 'linear'
                else:
                    acti = self.inputs['acti_func']

                FIL.write('LAYER {} {}\n'.format(j+joffset, acti))

                for k in range(self.nodes[item][j]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(weights[j*2][:,k].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, weights[j*2 + 1][k]))

            FIL.write('\n')

        FIL.close()

    def _save(self, sess, saver):
        if not self.inputs['continue']:
            self.inputs['continue'] = True
            self.parent.write_inputs()

        cutline = '----------------------------------------------'
        if self.inputs['use_force']:
            cutline += '------------------------'
        if self.inputs['use_stress']:
            cutline += '------------------------'

        if not self.inputs['print_structure_rmse']:
            self.parent.logfile.write(cutline + "\n")
        self.parent.logfile.write("Save the weights and write the LAMMPS potential..\n")
        self.parent.logfile.write(cutline + "\n")
        saver.save(sess, './SAVER_iteration{}'.format(sess.run(self.global_step)+1))
        self._generate_lammps_potential(sess)


    def _make_iterator_from_handle(self, training_dataset, atomic_weights=False, modifier=None):
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, training_dataset.output_types, training_dataset.output_shapes)
        self.next_elem = self.iterator.get_next()

        max_totnum = tf.cast(tf.reduce_max(self.next_elem['tot_num']), tf.int32)
        self.next_elem['num_seg'] = tf.shape(self.next_elem['tot_num'])[0] + 1
        self.next_elem['sparse_indices_'] = tf.cast(tf.range(tf.reduce_sum(self.next_elem['tot_num'])
        ), tf.int32)

        if self.inputs['use_force']:
            self.next_elem['partition'] = tf.reshape(self.next_elem['partition'], [-1])
            self.next_elem['seg_id_'] = tf.dynamic_partition(tf.reshape(tf.map_fn(lambda x: tf.tile([x+1], [max_totnum]), # dx_shape[1]
                                                                                  tf.range(tf.shape(self.next_elem['tot_num'])[0])), [-1]),
                                                                        self.next_elem['partition'], 2)[1]
            self.next_elem['F'] = \
                tf.dynamic_partition(
                    tf.reshape(self.next_elem['F'], [-1, 3]),
                    self.next_elem['partition'], 2
                )[1]
            self.next_elem['atom_idx'] = tf.dynamic_partition(\
                                             tf.reshape(self.next_elem['atom_idx'], [-1, 1]),
                                             self.next_elem['partition'], 2
                                         )[1]

        if self.inputs['E_loss'] == 3:
            self.next_elem['partition'] = tf.reshape(self.next_elem['partition'], [-1])
            self.next_elem['seg_id_'] = tf.dynamic_partition(tf.reshape(tf.map_fn(lambda x: tf.tile([x+1], [max_totnum]), # dx_shape[1]
                                                                                  tf.range(tf.shape(self.next_elem['tot_num'])[0])), [-1]),
                                                                        self.next_elem['partition'], 2)[1]
            self.next_elem['atom_idx'] = tf.dynamic_partition(\
                                             tf.reshape(self.next_elem['atom_idx'], [-1, 1]),
                                             self.next_elem['partition'], 2
                                         )[1]
            # which place?
        if atomic_weights:
            self.next_elem['atomic_weights_org'] = list()
            self.next_elem['atomic_weights'] = list()
            self.next_elem['dense_out'] = list()
            self.next_elem['partition_peratom'] = list()

        for item in self.parent.inputs['atom_types']:
            zero_cond = tf.equal(tf.reduce_sum(self.next_elem['N_'+item]), 0)

            if atomic_weights:
                self.next_elem['partition_peratom'].append(self.next_elem['partition_'+item])
            self.next_elem['partition_'+item] = tf.cond(zero_cond,
                                                        lambda: tf.zeros([1], tf.int32),
                                                        lambda: tf.reshape(self.next_elem['partition_'+item], [-1]))

            x_shape = tf.shape(self.next_elem['x_'+item])
            self.next_elem['x_'+item] = tf.cond(zero_cond,
                                                lambda: tf.zeros([1, self.inp_size[item]], dtype=tf.float64),
                                                lambda: tf.dynamic_partition(
                                                            tf.reshape(self.next_elem['x_'+item], [-1, self.inp_size[item]]),
                                                            self.next_elem['partition_'+item], 2)[1])

            self.next_elem['x_'+item] -= self.scale[item][0:1,:]
            self.next_elem['x_'+item] /= self.scale[item][1:2,:]
            if self.inputs['pca']:
                self.next_elem['x_'+item] = tf.matmul(self.next_elem['x_'+item], self.pca[item][0]) - self.pca[item][2]
                if self.inputs['pca_whiten']:
                    self.next_elem['x_'+item] /= self.pca[item][1].reshape([1, -1])

            if self.parent.inputs['symmetry_function']['add_NNP_ref']:
                self.next_elem['NNP_E_'+item] = tf.cond(zero_cond, 
                                                    lambda: tf.zeros([1, 1], dtype=tf.float64),
                                                    lambda: tf.dynamic_partition(
                                                                tf.reshape(self.next_elem['NNP_E_'+item], [-1, 1]),
                                                                self.next_elem['partition_'+item], 2)[1])
                
            if self.inputs['use_force']:
                dx_shape = tf.shape(self.next_elem['dx_'+item])
                self.next_elem['dx_'+item] = tf.cond(zero_cond,
                                                     lambda: tf.zeros([1, dx_shape[2], 1, dx_shape[4]], dtype=tf.float64),
                                                     lambda: tf.dynamic_partition(tf.reshape(self.next_elem['dx_'+item], [-1, dx_shape[2], dx_shape[3], dx_shape[4]]),
                                                                                  self.next_elem['partition_'+item], 2)[1])
            if self.inputs['use_stress']:
                da_shape = tf.shape(self.next_elem['da_'+item])
                self.next_elem['da_'+item] = tf.cond(zero_cond,
                                                     lambda: tf.zeros([1, da_shape[2], da_shape[3], da_shape[4]], dtype=tf.float64),
                                                     lambda: tf.dynamic_partition(tf.reshape(self.next_elem['da_'+item], [-1, da_shape[2], da_shape[3], da_shape[4]]),
                                                                                  self.next_elem['partition_'+item], 2)[1])

            self.next_elem['struct_type_set'], self.next_elem['struct_ind'], self.next_elem['struct_N'] = \
                    tf.unique_with_counts(tf.reshape(self.next_elem['struct_type'], [-1]))

            if self.inputs['use_force']:
                self.next_elem['dx_'+item] = tf.cond(tf.equal(tf.shape(self.next_elem['dx_'+item])[2], max_totnum),
                                                     lambda: self.next_elem['dx_'+item],
                                                     lambda: tf.pad(self.next_elem['dx_'+item], 
                                                                    [[0, 0], [0, 0], [0, max_totnum-tf.shape(self.next_elem['dx_'+item])[2]], [0, 0]]))
             
                self.next_elem['dx_'+item] /= self.scale[item][1:2,:].reshape([1, self.inp_size[item], 1, 1])
                if self.inputs['pca']:
                    self.next_elem['dx_'+item] = tf.einsum('ijkl,jm->imkl', self.next_elem['dx_'+item], tf.constant(self.pca[item][0]))
                    if self.inputs['pca_whiten']:
                        self.next_elem['dx_'+item] /= self.pca[item][1].reshape([1, -1, 1, 1])

            if self.inputs['use_stress']:
                self.next_elem['da_'+item] /= self.scale[item][1:2,:].reshape([1, self.inp_size[item], 1, 1])
                if self.inputs['pca']:
                    self.next_elem['da_'+item] = tf.einsum('ijkl,jm->imkl', self.next_elem['da_'+item], tf.constant(self.pca[item][0]))
                    if self.inputs['pca_whiten']:
                        self.next_elem['da_'+item] /= self.pca[item][1].reshape([1, -1, 1, 1])

            self.next_elem['seg_id_'+item] = tf.cond(zero_cond,
                                                     lambda: tf.zeros([1], tf.int32),
                                                     lambda: tf.dynamic_partition(tf.reshape(tf.map_fn(lambda x: tf.tile([x+1], [x_shape[1]]), # dx_shape[1]
                                                                                             tf.range(tf.shape(self.next_elem['N_'+item])[0])), [-1]),
                                                                                  self.next_elem['partition_'+item], 2)[1])

            self.next_elem['sparse_indices_'+item] = tf.cast(tf.range(tf.reduce_sum(
                tf.cond(zero_cond,
                        lambda: tf.constant([1], dtype=tf.int64),
                        lambda: self.next_elem['N_'+item])
                )), tf.int32)


            if atomic_weights:
                #self.next_elem['atomic_weights_'+item] = tf.cond(
                #                                            zero_cond,
                #                                            lambda: tf.ones([0], tf.float64),#tf.ones([tf.reduce_sum(self.next_elem['N_'+item])], tf.float64),
                #                                            lambda: tf.dynamic_partition(
                #                                                tf.reshape(self.next_elem['atomic_weights_'+item], [-1]),
                #                                                self.next_elem['partition_'+item], 2)[1]
                #                                         )

                self.next_elem['atomic_weights_org'].append(self.next_elem['atomic_weights_'+item])
                if modifier != None and callable(modifier[item]):
                    self.next_elem['dense_out_'+item] = tf.greater_equal(
                                                            self.next_elem['atomic_weights_'+item], 
                                                            self.parent.inputs['symmetry_function']['weight_modifier']['params'][item]['c']
                                                            )
                    self.next_elem['atomic_weights_'+item] = modifier[item](self.next_elem['atomic_weights_'+item], module_type=tf)
                else:
                    self.next_elem['dense_out_'+item] = tf.cast(tf.ones(tf.shape(self.next_elem['atomic_weights_'+item]), tf.float64), tf.bool)

                self.next_elem['atomic_weights_'+item] /= self.gdf_scale[item]
                self.next_elem['atomic_weights'].append(self.next_elem['atomic_weights_'+item])
                self.next_elem['dense_out'].append(self.next_elem['dense_out_'+item])

        if atomic_weights:
            self.next_elem['partition_peratom'] = tf.reshape(tf.concat(self.next_elem['partition_peratom'], axis=1), [-1,1])
            self.next_elem['atomic_weights_org'] = tf.dynamic_partition(
                                                    tf.reshape(tf.concat(self.next_elem['atomic_weights_org'], axis=1), [-1,1]),
                                                    self.next_elem['partition_peratom'], 2)[1]
            self.next_elem['atomic_weights'] = tf.reshape(
                                                    tf.dynamic_partition(
                                                        tf.reshape(tf.concat(self.next_elem['atomic_weights'], axis=1), [-1,1]),
                                                        self.next_elem['partition_peratom'], 2)[1], [-1,1]
                                                    )
            self.next_elem['dense_out'] = tf.reshape(
                                            tf.dynamic_partition(
                                                tf.reshape(tf.concat(self.next_elem['dense_out'], axis=1), [-1,1]),
                                                self.next_elem['partition_peratom'], 2)[1], [-1,1]
                                            )


    def train(self, user_optimizer=None, aw_modifier=None):
        self.inputs = self.parent.inputs['neural_network']
        # read data?

        self._set_params('symmetry_function')
        self._set_scale_parameter('./scale_factor')

        modifier_tag = dict()
        modifier_total = False
        for item in self.parent.inputs['atom_types']:
            modifier_tag[item] = True
            if aw_modifier == None:
                modifier_tag[item] = False
            elif (item in aw_modifier.keys()) and callable(aw_modifier[item]):
                modifier_tag[item] = True
            else:
                modifier_tag[item] = False
            modifier_total = modifier_total or modifier_tag[item]
        # Turn off modifier breakdown if atomic weights are not calculated (e.g. read from file or user-provided function)
        modifier_tag['total'] = modifier_total and self.parent.descriptor.inputs['atomic_weights']['type'] == 'gdf'

        if self.inputs['train']:
            train_filequeue = _make_data_list(self.train_data_list)
            valid_filequeue = _make_data_list(self.valid_data_list)

            if self.parent.descriptor.inputs['atomic_weights']['type'] == None:
                aw_tag = False
            else:
                aw_tag = True
                self._set_gdf_parameters('./atomic_weights', aw_modifier)

            train_iter = self.parent.descriptor._tfrecord_input_fn(train_filequeue, self.inp_size, cache=self.inputs['cache'],
                                                                   batch_size=self.inputs['batch_size'], use_force=self.inputs['use_force'], 
                                                                   use_stress=self.inputs['use_stress'], full_batch=self.inputs['full_batch'], 
                                                                   atomic_weights=aw_tag)
            valid_iter = self.parent.descriptor._tfrecord_input_fn(valid_filequeue, self.inp_size, cache=self.inputs['cache'],
                                                                   batch_size=self.inputs['batch_size'], use_force=self.inputs['use_force'], 
                                                                   use_stress=self.inputs['use_stress'],
                                                                   valid=True, atomic_weights=aw_tag)
            self._make_iterator_from_handle(train_iter, aw_tag, modifier=aw_modifier)

        if self.inputs['test'] or self.inputs['NNP_to_pickle']:
            test_filequeue = _make_data_list(self.test_data_list)

            if self.parent.descriptor.inputs['atomic_weights']['type'] == None:
                aw_tag = False
            else:
                aw_tag = True
                self._set_gdf_parameters('./atomic_weights', aw_modifier)

            if self.inputs['NNP_to_pickle']:
                batch_size=1            

            test_iter = self.parent.descriptor._tfrecord_input_fn(test_filequeue, self.inp_size, cache=self.inputs['cache'], 
                                                                  batch_size=(1 if self.inputs['NNP_to_pickle'] else self.inputs['batch_size']), 
                                                                  use_force=self.inputs['use_force'], use_stress=self.inputs['use_stress'],
                                                                  valid=True, atomic_weights=aw_tag)
            if not self.inputs['train']:
                self._make_iterator_from_handle(test_iter, aw_tag)

        self.stress_coeff = self._get_decay_param(self.inputs['stress_coeff'])
        self.force_coeff = self._get_decay_param(self.inputs['force_coeff'])
        self.energy_coeff = self._get_decay_param(self.inputs['energy_coeff'])

        self._make_model()
        self._calc_output()
        self._get_loss()
        self._make_optimizer(user_optimizer=user_optimizer)

        config = tf.ConfigProto()
        config.inter_op_parallelism_threads = self.inputs['inter_op_parallelism_threads']
        config.intra_op_parallelism_threads = self.inputs['intra_op_parallelism_threads']
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.45
        with tf.Session(config=config) as sess:
            # Load or initialize the variables
            saver = tf.train.Saver(max_to_keep=None)
            if self.inputs['continue'] == True:
                saver.restore(sess, './SAVER')
            else:
                sess.run(tf.global_variables_initializer())

            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()

            # Save criteria
            prev_criteria = list()
            if 'v_E' in self.inputs['save_criteria']:
                prev_criteria.append(float('inf')) # E_loss (valid)

            if self.inputs['use_force']:
                if 'v_F' in self.inputs['save_criteria']:
                    prev_criteria.append(float('inf')) # F_loss (valid)

                if modifier_tag['total']:
                    for item in self.parent.inputs['atom_types']:
                        if modifier_tag[item] and 'v_F_{}_sparse'.format(item) in self.inputs['save_criteria']:
                            prev_criteria.append(float('inf'))

            if self.inputs['use_stress']:
                if 'v_S' in self.inputs['save_criteria']:
                    prev_criteria.append(float('inf'))

            prev_criteria = np.array(prev_criteria)
            #prev_eloss = float('inf')
            #prev_floss = float('inf')
            save_stack = 1

            if self.inputs['train']:
                train_handle = sess.run(train_iter.string_handle())
                train_fdict = {self.handle: train_handle}
                if self.inputs['full_batch']:
                    self.grad_ph = list()
                    #full_batch_dict = dict()
                    self.grad_shape = list()
                    for i,item in enumerate(self.compute_grad):
                        self.grad_shape.append(sess.run(tf.shape(item[1])))
                        self.grad_ph.append(tf.placeholder(tf.float64, self.grad_shape[-1]))
                        #full_batch_dict[self.grad_ph[i]] = None

                    self.apply_grad = self.optim.apply_gradients([(self.grad_ph[i], item[1]) for i,item in enumerate(self.compute_grad)],
                                                                 global_step=self.global_step)
                    self.tmp_apply_grad = self.optim.apply_gradients([(self.grad_ph[i], item[1]) for i,item in enumerate(self.compute_grad)])
                else:
                    sess.run(train_iter.initializer)

                valid_handle = sess.run(valid_iter.string_handle())
                valid_fdict = {self.handle: valid_handle}

                # Log validation set statistics.
                _, _, _, _, _, _, _, str_tot_struc, str_tot_atom, str_weight, str_set = self._get_loss_for_print(
                    sess, valid_fdict, full_batch=True, iter_for_initialize=valid_iter, modifier_tag=modifier_tag)

                self._log_statistics(str_tot_struc, str_tot_atom, str_weight)
                """
                if self.inputs['method'] == 'L-BFGS':
                    # TODO: complete this part
                    raise ValueError

                elif self.inputs['method'] == 'Adam':
                """
                if self.inputs['method'] == 'L-BFGS':
                    lbfgs = L_BFGS()

                time1 = timeit.default_timer()
                save_time = 0

                if self.inputs['total_iteration'] < 0:
                    total_iteration = -self.inputs['total_iteration']
                    break_tag = True
                    break_count = 1
                    break_max = self.inputs['break_max']
                else:
                    total_iteration = self.inputs['total_iteration']
                    break_tag = False

                time_begin = timeit.default_timer()
                for iteration in tqdm(range(total_iteration)):
                    if self.inputs['full_batch']:
                        if self.inputs['method'] == 'Adam':
                            [flat_grad] = self._get_full_batch_values(sess, train_iter, train_fdict, need_loss=False)
                            sess.run(self.apply_grad, feed_dict=self._get_grad_dict(flat_grad))
                        elif self.inputs['method'] == 'L-BFGS':
                            # calculate the direction
                            #zero_vals = _get_full_batch_values(sess, train_iter, train_fdict, need_loss=True)
                            if iteration == 0:
                                zero_vals = self._get_full_batch_values(sess, train_iter, train_fdict, need_loss=True)
                                z = zero_vals[0]
                            else:
                                zero_vals = [np.copy(alpha_vals[0]), np.copy(alpha_vals[1])]
                                z = lbfgs.find_direction(zero_vals[0])

                            lbfgs.initialize_line_search()
                            old_step = lbfgs.step
                            sess.run(self.tmp_apply_grad, feed_dict=self._get_grad_dict(z*lbfgs.step))
                            alpha_vals = self._get_full_batch_values(sess, train_iter, train_fdict, need_loss=True)
                            while lbfgs.wolfe_line_search_iter(zero_vals, alpha_vals, z):
                                sess.run(self.tmp_apply_grad, feed_dict=self._get_grad_dict(-z*old_step))
                                sess.run(self.tmp_apply_grad, feed_dict=self._get_grad_dict(z*lbfgs.step))
                                alpha_vals = self._get_full_batch_values(sess, train_iter, train_fdict, need_loss=True)
                                old_step = lbfgs.step

                            sess.run(self.increment_global_step)
                            lbfgs.update_lists(np.copy(alpha_vals[0] - zero_vals[0]), np.copy(z))

                    else:
                        self.minim.run(feed_dict=train_fdict)
                    #sess.run(self.optim, feed_dict=train_fdict, options=options, run_metadata=run_metadata)

                    # Logging
                    if (iteration+1) % self.inputs['show_interval'] == 0:
                        # Profiling
                        #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        #with open('timeline_test.json', 'w') as fil:
                        #    fil.write(chrome_trace)

                        time2 = timeit.default_timer()

                        # TODO: need to fix the calculation part for training loss
                        save_stack += self.inputs['show_interval']

                        result = "iteration {:7d} ".format(sess.run(self.global_step)+1)

                        t_eloss, t_floss, t_sloss, t_aw_floss, t_str_eloss, t_str_floss, t_str_sloss, _, _, _, t_str_set = \
                            self._get_loss_for_print(sess, train_fdict, full_batch=self.inputs['full_batch'], 
                                                    iter_for_initialize=train_iter, modifier_tag=modifier_tag)

                        eloss, floss, sloss, aw_floss, str_eloss, str_floss, str_sloss, _, _, _, _ = self._get_loss_for_print(
                            sess, valid_fdict, full_batch=True, iter_for_initialize=valid_iter, modifier_tag=modifier_tag)

                        full_str_set = list(set(t_str_set + str_set))

                        result += 'E RMSE(T V) = {:6.4e} {:6.4e}'.format(t_eloss, eloss)
                        if self.inputs['use_force']:
                            result += ' F RMSE(T V) = {:6.4e} {:6.4e}'.format(t_floss, floss)
                        if self.inputs['use_stress']:
                            result += ' S RMSE(T V) = {:6.4e} {:6.4e}'.format(t_sloss, sloss)

                        if self.inputs['method'] != 'L-BFGS':
                            lr = sess.run(self.learning_rate)
                            result += ' learning_rate: {:6.4e}'.format(lr)
                        result += ' elapsed: {:4.2e} s speed: {:4.2e} it/s\n'.format(
                                timeit.default_timer() - time_begin,
                                self.inputs['show_interval']/(time2-time1-save_time))

                        # Print splitted RMSE according to atomic weights
                        if modifier_tag['total']:
                            cutline = '----------------------------------------------------------------------\n'
                            result += cutline
                            result += 'modifier breakdown:\n'
                            result += '     |  rho(G)^-1 < c (dense)  | rho(G)^-1 >= c (sparse) |\n'
                            result += '     |    Train   |    Valid   |    Train   |    Valid   |\n'
                            for atem in self.parent.inputs['atom_types']:
                                if modifier_tag[atem]:
                                    result += '  {:2} | {:6.4e} | {:6.4e} | {:6.4e} | {:6.4e} |\n'.format(atem, 
                                                                                                          t_aw_floss[atem][1], aw_floss[atem][1], 
                                                                                                          t_aw_floss[atem][0], aw_floss[atem][0]) 

                        # Print structural breakdown of RMSE
                        if self.inputs['print_structure_rmse']:
                            cutline = '----------------------------------------------'
                            if self.inputs['use_force']:
                                cutline += '------------------------'
                            if self.inputs['use_stress']:
                                cutline += '------------------------'
                            result += cutline + '\n'
                            result += 'structural breakdown:\n'
                            result += '  label                  E_RMSE(T)   E_RMSE(V)'
                            if self.inputs['use_force']:
                                result += '   F_RMSE(T)   F_RMSE(V)'
                            if self.inputs['use_stress']:
                                result += '   S_RMSE(T)   S_RMSE(V)'
                            result += '\n'
                            for struct in sorted(full_str_set):
                                label = str(struct.decode()).replace(' ', '_')
                                if struct not in t_str_eloss:
                                    teloss = '          -'
                                    tfloss = '          -'
                                    tsloss = '          -'
                                else:
                                    teloss = '{:>11.4e}'.format(t_str_eloss[struct])
                                    if self.inputs['use_force']:
                                        tfloss = '{:>11.4e}'.format(t_str_floss[struct])
                                    if self.inputs['use_stress']:
                                        tsloss = '{:>11.4e}'.format(t_str_sloss[struct])
                                if struct not in str_eloss:
                                    veloss = '          -'
                                    vfloss = '          -'
                                    vsloss = '          -'
                                else:
                                    veloss = '{:>11.4e}'.format(str_eloss[struct])
                                    if self.inputs['use_force']:
                                        vfloss = '{:>11.4e}'.format(str_floss[struct])
                                    if self.inputs['use_stress']:
                                        vsloss = '{:>11.4e}'.format(str_sloss[struct])
                                result += '  {:<20.20} {:} {:}'.format(label, teloss, veloss)
                                if self.inputs['use_force']:
                                    result += ' {:} {:}'.format(tfloss, vfloss)
                                if self.inputs['use_stress']:
                                    result += ' {:} {:}'.format(tsloss, vsloss)
                                result += '\n'
                            result += cutline + '\n'

                        self.parent.logfile.write(result)
                        time1 = timeit.default_timer()

                        # TODO: modify save criteria
                        cur_criteria = list()
                        if 'v_E' in self.inputs['save_criteria']:
                            cur_criteria.append(eloss) # E_loss (valid)

                        if self.inputs['use_force']:
                            if 'v_F' in self.inputs['save_criteria']:
                                cur_criteria.append(floss) # F_loss (valid)

                            if modifier_tag['total']:
                                for atem in self.parent.inputs['atom_types']:
                                    if modifier_tag[atem] and 'v_F_{}_sparse'.format(atem) in self.inputs['save_criteria']:
                                        cur_criteria.append(aw_floss[atem][0])
                            
                        if self.inputs['use_stress']:
                            if 'v_S' in self.inputs['save_criteria']:
                                cur_criteria.append(sloss)
                
                        cur_criteria = np.array(cur_criteria)

                        save_criteria = np.prod(cur_criteria < prev_criteria)

                    # Temp saving
                    #if (iteration+1) % self.inputs['save_interval'] == 0:
                        if save_stack > self.inputs['save_interval'] and save_criteria:
                            #(prev_eloss > eloss or not self.inputs['echeck']) and \
                            #(prev_floss > floss or not self.inputs['fcheck'] or floss == 0.):

                            temp_time = timeit.default_timer()
                            self._save(sess, saver)
                            #prev_eloss = eloss
                            #prev_floss = floss
                            prev_criteria = np.copy(cur_criteria)
                            save_stack = 1
                            save_time = timeit.default_timer() - temp_time
                            if break_tag:
                                break_count = 1
                        else:
                            save_time = 0
                            if break_tag:
                                break_count += 1

                    # Stop the training if overfitting is occur
                    if break_tag:
                        if (break_count > self.inputs['break_max']):
                            break
                #self._save(sess, saver)


            if self.inputs['test']:
                test_handle = sess.run(test_iter.string_handle())
                test_fdict = {self.handle: test_handle}
                sess.run(test_iter.initializer)

                test_save = dict()
                test_save['DFT_E'] = list()
                test_save['NN_E'] = list()
                test_save['N'] = list()
                test_save['NN_atom_E'] = list()

                if self.inputs['use_force']:
                    test_save['DFT_F'] = list()
                    test_save['NN_F'] = list()
                    if self.parent.inputs['symmetry_function']['add_atom_idx']:
                        test_save['atom_idx'] = list()
                    if aw_tag:
                        test_save['atomic_weights'] = list()

                if self.inputs['use_stress']:
                    test_save['DFT_S'] = list()
                    test_save['NN_S'] = list()

                eloss = floss = sloss = 0.
                test_tot_struc = test_tot_atom = 0
                result = ' Test'
                while True:
                    try:
                        if self.inputs['use_force']:
                            if self.inputs['use_stress']:
                                test_elem, tmp_nne, tmp_nnf, tmp_nns, tmp_eloss, tmp_floss, tmp_sloss, tmp_atom_E = sess.run(
                                        [self.next_elem, self.E, self.F, self.S, self.e_loss, self.f_loss, self.s_loss, self.atom_E],
                                        feed_dict=test_fdict)
                                num_batch_struc = test_elem['num_seg'] - 1
                                sloss += tmp_sloss * num_batch_struc
                                test_save['DFT_S'].append(test_elem['S'])
                                test_save['NN_S'].append(tmp_nns)
                            else:
                                test_elem, tmp_nne, tmp_nnf, tmp_eloss, tmp_floss, tmp_atom_E = sess.run(
                                        [self.next_elem, self.E, self.F, self.e_loss, self.f_loss, self.atom_E], feed_dict=test_fdict)
                            num_batch_atom = np.sum(test_elem['tot_num'])
                            floss += tmp_floss * num_batch_atom

                            test_save['DFT_F'].append(test_elem['F'])
                            test_save['NN_F'].append(tmp_nnf)

                            if self.parent.inputs['symmetry_function']['add_atom_idx']:
                                temp_idx = test_elem['atom_idx'].reshape([-1])
                                temp_idx = temp_idx[temp_idx != 0]
                                test_save['atom_idx'].append(temp_idx)
                            if aw_tag:
                                test_save['atomic_weights'].append(test_elem['atomic_weights_org'])

                            test_tot_atom += num_batch_atom
                        elif self.inputs['use_stress']:
                            test_elem, tmp_nne, tmp_nns, tmp_eloss, tmp_sloss, tmp_atom_E = sess.run(
                                    [self.next_elem, self.E, self.S, self.e_loss, self.f_loss, self.s_loss, self.atom_E],
                                    feed_dict=test_fdict)
                            num_batch_struc = test_elem['num_seg'] - 1
                            sloss += tmp_sloss * num_batch_struc
                            test_save['DFT_S'].append(test_elem['S'])
                            test_save['NN_S'].append(tmp_nns)
                        else:
                            test_elem, tmp_nne, tmp_eloss, tmp_atom_E = \
                                sess.run([self.next_elem, self.E, self.e_loss, self.atom_E], feed_dict=test_fdict)
                        num_batch_struc = test_elem['num_seg'] - 1
                        eloss += tmp_eloss * num_batch_struc

                        test_save['DFT_E'].append(test_elem['E'])
                        test_save['NN_E'].append(tmp_nne)
                        test_save['N'].append(test_elem['tot_num'])
                        test_save['NN_atom_E'].append(tmp_atom_E)
                        
                        test_tot_struc += num_batch_struc
                    except tf.errors.OutOfRangeError:
                        eloss = np.sqrt(eloss/test_tot_struc)
                        result += 'E RMSE = {:6.4e}'.format(eloss)

                        test_save['DFT_E'] = np.concatenate(test_save['DFT_E'], axis=0)
                        test_save['NN_E'] = np.concatenate(test_save['NN_E'], axis=0)
                        test_save['N'] = np.concatenate(test_save['N'], axis=0)
                        if self.inputs['use_force']:
                            floss = np.sqrt(floss*3/test_tot_atom)
                            result += ', F RMSE = {:6.4e}'.format(floss)

                            test_save['DFT_F'] = np.concatenate(test_save['DFT_F'], axis=0)
                            test_save['NN_F'] = np.concatenate(test_save['NN_F'], axis=0)
                            test_save['atom_idx'] = np.concatenate(test_save['atom_idx'], axis=0)

                            if aw_tag:
                                test_save['atomic_weights'] = np.concatenate(test_save['atomic_weights'], axis=0)

                        if self.inputs['use_stress']:
                            sloss = np.sqrt(sloss*6/test_tot_struc)
                            result += ', S RMSE = {:6.4e}'.format(sloss)

                            test_save['DFT_S'] = np.concatenate(test_save['DFT_S'], axis=0)
                            test_save['NN_S'] = np.concatenate(test_save['NN_S'], axis=0)

                        break

                with open('./test_result', 'wb') as fil:
                    pickle.dump(test_save, fil, protocol=2)

                self.parent.logfile.write('Test result saved..\n')
                self.parent.logfile.write(result + '\n')

            elif self.inputs['NNP_to_pickle']:
                test_handle = sess.run(test_iter.string_handle())
                test_fdict = {self.handle: test_handle}
                sess.run(test_iter.initializer)

                while True:
                    try:
                        test_elem, tmp_nnate = sess.run([self.next_elem, self.atom_E], feed_dict=test_fdict)
                        tmp_pickledata = pickle_load(test_elem['pickle_name'][0][0])

                        tmp_pickledata['NNP_E'] = dict()
                        for item in self.parent.inputs['atom_types']:
                            tmp_pickledata['NNP_E'][item] = tmp_nnate[item]

                        with open(test_elem['pickle_name'][0][0], 'wb') as fil:
                            pickle.dump(tmp_pickledata, fil, protocol=2)
                        self.parent.logfile.write('NNP energy for {} are saved\n'.format(test_elem['pickle_name'][0][0]))
                    except tf.errors.OutOfRangeError:
                        break 


    def _log_statistics(self, str_tot_struc, str_tot_atom, str_weight):
        result = ''
        result += 'validation set statistics:\n'
        result += '  label                 struct_count percentage atom_count percentage      weight\n'
        total_count_struc = sum(str_tot_struc.values())
        total_count_atom = sum(str_tot_atom.values())
        for struct in sorted(str_tot_struc.keys()):
            label = str(struct.decode()).replace(' ', '_')
            count_struc = str_tot_struc[struct]
            count_atom = str_tot_atom[struct]
            result += '  {:<20.20} {:>13} {:>10.2f} {:>10} {:>10.2f} {:>11.4e}\n'.format(
                    label, count_struc, float(count_struc) / total_count_struc * 100,
                    int(count_atom), float(count_atom) / total_count_atom * 100, str_weight[struct])
        result += '  {:<20.20} {:>13} {:>10.2f} {:>10} {:>10.2f} {:>11}\n\n'.format(
                'TOTAL', total_count_struc, 100.0, int(total_count_atom), 100.0, '-')
        self.parent.logfile.write(result)


    # TODO: check memory leak!
    def _get_loss_for_print(self, sess, fdict, full_batch=False, iter_for_initialize=None, modifier_tag=None):
        eloss = floss = sloss = 0
        num_tot_struc = num_tot_atom = 0
        aw_floss = {}
        str_eloss = {}
        str_floss = {}
        str_sloss = {}
        str_tot_struc = {}
        str_tot_atom = {}
        str_weight = {}

        """
        for item in self.str_set:
            str_eloss[item] = 0.0
            str_floss[item] = 0.0
            str_tot_struc[item] = 0
            str_tot_atom[item] = 0
        """

        if full_batch:
            # TODO: check the error
            sess.run(iter_for_initialize.initializer)
            while True:
                try:
                    if self.inputs['use_force']:
                        if self.inputs['use_stress']:
                            next_elem, tmp_eloss, tmp_f, tmp_floss, tmp_sloss, \
                            tmp_str_eloss, tmp_str_floss, tmp_str_sloss, tmp_str_atom = sess.run(
                                    [self.next_elem, self.e_loss, self.F, self.f_loss, self.s_loss, \
                                    self.str_e_loss, self.str_f_loss, self.str_s_loss, self.str_num_batch_atom],
                                    feed_dict = fdict)
                            num_batch_struc = next_elem['num_seg'] - 1
                            sloss += tmp_sloss * num_batch_struc
                        else:
                            next_elem, tmp_eloss, tmp_f, tmp_floss, tmp_str_eloss, tmp_str_floss, tmp_str_atom = sess.run(
                                    [self.next_elem, self.e_loss, self.F, self.f_loss, self.str_e_loss,
                                    self.str_f_loss, self.str_num_batch_atom], feed_dict=fdict)
                        num_batch_atom = np.sum(next_elem['tot_num'])
                        if self.inputs['F_loss'] == 1:
                            floss += tmp_floss * num_batch_atom
                        else:
                            floss += tmp_floss

                        if modifier_tag['total']:
                            tmp_floss_not_packed = np.sum(np.square(next_elem['F'] - tmp_f), axis=1, keepdims=True)
                            tmp_sparse_idx = next_elem['dense_out']
                            tmp_dense_idx = np.logical_not(next_elem['dense_out'])

                            for i,item in enumerate(self.parent.inputs['atom_types']):
                                if item not in aw_floss:
                                    aw_floss[item] = [list(), list()]

                                aw_floss[item][0].append(tmp_floss_not_packed[np.logical_and(tmp_sparse_idx, next_elem['atom_idx'] == (i+1))])
                                aw_floss[item][1].append(tmp_floss_not_packed[np.logical_and(tmp_dense_idx, next_elem['atom_idx'] == (i+1))])
                        num_tot_atom += num_batch_atom
                    elif self.inputs['use_stress']:
                        next_elem, tmp_eloss, tmp_sloss, tmp_str_eloss, tmp_str_sloss, tmp_str_atom = sess.run(
                                [self.next_elem, self.e_loss, self.s_loss, self.str_e_loss, self.str_s_loss, self.str_num_batch_atom],
                                feed_dict=fdict)
                        num_batch_struc = next_elem['num_seg'] - 1
                        sloss += tmp_sloss * num_batch_struc
                    else:
                        next_elem, tmp_eloss, tmp_str_eloss, tmp_str_atom = sess.run(
                            [self.next_elem, self.e_loss, self.str_e_loss, self.str_num_batch_atom], feed_dict=fdict)
                    num_batch_struc = next_elem['num_seg'] - 1
                    eloss += tmp_eloss * num_batch_struc
                    num_tot_struc += num_batch_struc

                    for i,struct in enumerate(next_elem['struct_type_set']):
                        if struct not in str_eloss:
                            str_eloss[struct] = 0.
                            str_floss[struct] = 0.
                            str_sloss[struct] = 0.
                            str_tot_struc[struct] = 0
                            str_tot_atom[struct] = 0

                        str_eloss[struct] += tmp_str_eloss[i] * next_elem['struct_N'][i]
                        str_tot_struc[struct] += next_elem['struct_N'][i]
                        str_tot_atom[struct] += tmp_str_atom[i]
                        if self.inputs['use_force']:
                            str_floss[struct] += tmp_str_floss[i] * tmp_str_atom[i]
                        if self.inputs['use_stress']:
                            str_sloss[struct] += tmp_str_sloss[i] * next_elem['struct_N'][i]

                    for struct, weight in zip(next_elem['struct_type'].reshape([-1]),
                                              next_elem['struct_weight'].reshape([-1])):
                        str_weight[struct] = weight

                except tf.errors.OutOfRangeError:
                    eloss = np.sqrt(eloss/num_tot_struc)
                    for struct in str_eloss.keys():
                        str_eloss[struct] = np.sqrt(str_eloss[struct]/str_tot_struc[struct])

                    if self.inputs['use_force']:
                        if self.inputs['F_loss'] == 1:
                            floss = np.sqrt(floss*3/num_tot_atom)
                        else:
                            floss = np.sqrt(floss*3/num_tot_struc)

                        for struct in str_floss.keys():
                            str_floss[struct] = np.sqrt(str_floss[struct]*3/str_tot_atom[struct])

                        if modifier_tag['total']:
                            for item in self.parent.inputs['atom_types']:
                                aw_floss[item][0] = np.sqrt(np.mean(np.concatenate(aw_floss[item][0], axis=0)))
                                aw_floss[item][1] = np.sqrt(np.mean(np.concatenate(aw_floss[item][1], axis=0)))
                        
                    if self.inputs['use_stress']:
                        sloss = np.sqrt(sloss*6/num_tot_struc)

                        for struct in str_sloss.keys():
                            str_sloss[struct] = np.sqrt(str_sloss[struct]*6/str_tot_struc[struct])
                    break
        else:
            if self.inputs['use_force']:
                if self.inputs['use_stress']:
                    next_elem, eloss, tmp_f, floss, sloss, tmp_str_eloss, tmp_str_floss, tmp_str_sloss = sess.run(
                            [self.next_elem, self.e_loss, self.F, self.f_loss, self.s_loss, 
                            self.str_e_loss, self.str_f_loss, self.str_s_loss], feed_dict = fdict)
                    sloss = np.sqrt(sloss*6)
                    tmp_str_sloss = np.sqrt(tmp_str_sloss*6)
                else:
                    next_elem, eloss, tmp_f, floss, tmp_str_eloss, tmp_str_floss = sess.run(
                        [self.next_elem, self.e_loss, self.F, self.f_loss, self.str_e_loss, self.str_f_loss], feed_dict=fdict)
                floss = np.sqrt(floss*3)
                tmp_str_floss = np.sqrt(tmp_str_floss*3)

                if modifier_tag['total']:
                    tmp_floss_not_packed = np.sum(np.square(next_elem['F'] - tmp_f), axis=1, keepdims=True)
                    tmp_sparse_idx = next_elem['dense_out']
                    tmp_dense_idx = np.logical_not(next_elem['dense_out'])

                    for i,item in enumerate(self.parent.inputs['atom_types']):
                        if item not in aw_floss:
                            aw_floss[item] = [list(), list()]
                        aw_floss[item][0] = np.sqrt(np.mean(tmp_floss_not_packed[np.logical_and(tmp_sparse_idx, next_elem['atom_idx'] == (i+1))]))
                        aw_floss[item][1] = np.sqrt(np.mean(tmp_floss_not_packed[np.logical_and(tmp_dense_idx, next_elem['atom_idx'] == (i+1))]))
            elif self.inputs['use_stress']:
                next_elem, eloss, sloss, tmp_str_eloss, tmp_str_sloss = sess.run(
                        [self.next_elem, self.e_loss, self.s_loss, self.str_e_loss, self.str_s_loss],
                        feed_dict = fdict)
                sloss = np.sqrt(sloss*6)
                tmp_str_sloss = np.sqrt(tmp_str_sloss*6)
            else:
                next_elem, eloss, tmp_str_eloss = sess.run(
                    [self.next_elem, self.e_loss, self.str_e_loss], feed_dict=fdict)
            eloss = np.sqrt(eloss)
            tmp_str_eloss = np.sqrt(tmp_str_eloss)

            for i,struct in enumerate(next_elem['struct_type_set']):
                if struct not in str_eloss:
                    str_eloss[struct] = 0.
                    str_floss[struct] = 0.
                    str_sloss[struct] = 0.
                    str_tot_struc[struct] = 0
                    str_tot_atom[struct] = 0

                str_eloss[struct] = tmp_str_eloss[i]
                if self.inputs['use_force']:
                    str_floss[struct] = tmp_str_floss[i]
                if self.inputs['use_stress']:
                    str_sloss[struct] = tmp_str_sloss[i]

            for struct, weight in zip(next_elem['struct_type'].reshape([-1]),
                                      next_elem['struct_weight'].reshape([-1])):
                str_weight[struct] = weight

        return eloss, floss, sloss, aw_floss, str_eloss, str_floss, str_sloss, str_tot_struc, str_tot_atom, str_weight, list(str_eloss.keys())


    def _get_grad_dict(self, flat_grad):
        grad_dict = dict()
        idx = 0
        for item, ishape in zip(self.grad_ph, self.grad_shape):
            size_1d = np.prod(ishape)
            grad_dict[item] = flat_grad[idx:idx+size_1d].reshape(ishape)
            idx += size_1d

        return grad_dict

    def _get_full_batch_values(self, sess, target_iter, target_fdict, need_loss=False):
        sess.run(target_iter.initializer)
        res = sess.run(([self.flat_grad, self.total_loss]
                        if need_loss
                        else [self.flat_grad]), feed_dict=target_fdict)
        while True:
            try:
                for i,item in enumerate(sess.run(([self.flat_grad, self.total_loss]
                                                  if need_loss
                                                  else [self.flat_grad]), feed_dict=target_fdict)):
                    res[i] += item
            except tf.errors.OutOfRangeError:
                break
        return res

    def set_inputs(self):
        self.inputs = self.parent.inputs['neural_network']
