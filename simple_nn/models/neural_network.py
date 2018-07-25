import tensorflow as tf
import numpy as np
import random
import six
from six.moves import cPickle as pickle
import collections
import functools
import timeit
import copy
from ..utils import _make_data_list, pickle_load, _generate_gdf_file, modified_sigmoid, memory
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
                                      'method': 'Adam',
                                      'continue': False,
                                      'use_force': False,
                                      'force_coeff': 0.3,
                                      'energy_coeff': 1.,
                                      'total_epoch': 10000,
                                      'save_interval': 1000,
                                      'show_interval': 100,
                                      'max_iteration': 1000,
                                      'batch_size': 64,
                                      'full_batch': False,
                                      'loss_scale': 1.,
                                      'double_precision': True,
                                      'learning_rate': 0.01,
                                      'optimizer': dict(),
                                      'nodes': '30-30',
                                      'test': False,
                                      'train': True,
                                      'regularization': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'inter_op_parallelism_threads': 0,
                                      'intra_op_parallelism_threads': 0,
                                  }
                              }
        self.inputs = dict()
        self.global_step = tf.Variable(0, trainable=False)
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

            self.params[item] = np.array(self.params[item])
            self.inp_size[item] = self.params[item].shape[0]


    def _set_scale_parameter(self, scale_file):
        self.scale = pickle_load(scale_file)
        # TODO: add the check code for valid scale file

    def _make_model(self):
        self.models = dict()
        self.ys = dict()
        self.dys = dict()

        if self.inputs['double_precision']:
            dtype = tf.float64
        else:
            dtype = tf.float32

        dense_basic_setting = {
            'dtype': dtype,
            'kernel_initializer': tf.initializers.truncated_normal(stddev=0.3, dtype=dtype),
            'bias_initializer': tf.initializers.truncated_normal(stddev=0.3, dtype=dtype)
        }
        dense_last_setting = copy.deepcopy(dense_basic_setting)

        if self.inputs['regularization']['type'] is not None:
            if self.inputs['regularization']['type'] == 'l2':
                coeff = self.inputs['regularization']['params'].get('coeff', 1e-6)
                dense_basic_setting['kernel_regularizer'] = tf.keras.regularizers.l2(l=coeff)
                dense_basic_setting['bias_regularizer'] = tf.keras.regularizers.l2(l=coeff)
                dense_last_setting['kernel_regularizer'] = tf.keras.regularizers.l2(l=coeff)

        #acti_func = 'selu'
        #acti_func = 'elu'
        acti_func = 'sigmoid'
        #acti_func = 'tanh'

        self.nodes = dict()
        for item in self.parent.inputs['atom_types']:
            if isinstance(self.inputs['nodes'], collections.Mapping):
                nodes = list(map(int, self.inputs['nodes'][item].split('-')))
            else:
                nodes = list(map(int, self.inputs['nodes'].split('-')))
            nlayers = len(nodes)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(nodes[0], activation=acti_func, \
                                            input_dim=self.inp_size[item],
                                            #kernel_initializer=tf.initializers.random_normal(stddev=1./self.inp_size[item], dtype=dtype),
                                            #use_bias=False,
                                            **dense_basic_setting))

            for i in range(1, nlayers):
                model.add(tf.keras.layers.Dense(nodes[i], activation=acti_func,
                                                #kernel_initializer=tf.initializers.random_normal(stddev=1./nodes[i-1], dtype=dtype),
                                                #use_bias=False,
                                                **dense_basic_setting))
            model.add(tf.keras.layers.Dense(1, activation='linear', 
                                            #kernel_initializer=tf.initializers.random_normal(stddev=1./nodes[-1], dtype=dtype),
                                            #bias_initializer=tf.initializers.random_normal(stddev=0.1, dtype=dtype),
                                            **dense_last_setting))

            nodes.append(1)
            self.nodes[item] = nodes

            self.models[item] = model
            self.ys[item] = self.models[item](self.next_elem['x_'+item])

            if self.inputs['use_force']:
                self.dys[item] = tf.gradients(self.ys[item], self.next_elem['x_'+item])[0]
            else:
                self.dys[item] = None


    def _calc_output(self):
        self.E = self.F = 0

        for item in self.parent.inputs['atom_types']:
            self.E += tf.sparse_segment_sum(self.ys[item], self.next_elem['sparse_indices_'+item], self.next_elem['seg_id_'+item], 
                                            num_segments=self.next_elem['num_seg'])[1:]

            if self.inputs['use_force']:
                tmp_force = self.next_elem['dx_'+item] * \
                            tf.expand_dims(\
                                tf.expand_dims(self.dys[item], axis=2),
                                axis=3)
                tmp_force = tf.reduce_sum(\
                                tf.sparse_segment_sum(tmp_force, self.next_elem['sparse_indices_'+item], self.next_elem['seg_id_'+item], 
                                                      num_segments=self.next_elem['num_seg'])[1:],
                                axis=1)
                self.F -= tf.cond(tf.equal(tf.reduce_sum(self.next_elem['N_'+item]), 0),
                                  lambda: tf.cast(0., tf.float64),
                                  lambda: tf.dynamic_partition(tf.reshape(tmp_force, [-1,3]),
                                                               self.next_elem['partition'], 2)[1])

    def _get_loss(self, use_gdf=False, atomic_weights=None):
        self.e_loss = tf.reduce_mean(tf.square((self.next_elem['E'] - self.E) / self.next_elem['tot_num']))
        self.total_loss = self.e_loss * self.energy_coeff

        if self.inputs['use_force']:
            self.f_loss = tf.square(self.next_elem['F'] - self.F)
            if self.parent.descriptor.inputs['atomic_weights']['type'] is not None:
                self.aw_f_loss = self.f_loss * self.next_elem['atomic_weights']
                self.f_loss = tf.reduce_mean(self.f_loss)
                self.aw_f_loss = tf.reduce_mean(self.aw_f_loss)
                self.total_loss += self.aw_f_loss * self.force_coeff
            else:
                self.f_loss = tf.reduce_mean(self.f_loss)
                self.total_loss += self.f_loss * self.force_coeff

        if self.inputs['regularization']['type'] is not None:
            # FIXME: regularization_loss, which is float32, is casted into float64.
            self.total_loss += tf.cast(tf.losses.get_regularization_loss(), tf.float64)

    def _make_optimizer(self, user_optimizer=None):
        final_loss = self.inputs['loss_scale']*self.total_loss
        if self.inputs['method'] == 'L-BFGS-B':
            self.optim = tf.contrib.opt.ScipyOptimizerInterface(final_loss, 
                                                                method=self.inputs['method'], 
                                                                options=self.inputs['optimizer'])
        elif self.inputs['method'] == 'Adam':
            if isinstance(self.inputs['learning_rate'], collections.Mapping):
                exponential_decay_inputs = copy.deepcopy(self.inputs['learning_rate'])
                exponential_decay_inputs['learning_rate'] = tf.constant(exponential_decay_inputs['learning_rate'], tf.float64)
                self.learning_rate = tf.train.exponential_decay(global_step=self.global_step, **exponential_decay_inputs)
            else:
                self.learning_rate = tf.constant(self.inputs['learning_rate'], tf.float64)

            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                name='Adam', **self.inputs['optimizer'])
            self.compute_grad = self.optim.compute_gradients(final_loss)
            self.grad_and_vars = [[None, item[1]] for item in self.compute_grad]
            self.minim = self.optim.minimize(final_loss, global_step=self.global_step)
        else:
            if user_optimizer != None:
                self.optim = user_optimizer.minimize(final_loss, global_step=self.global_step)
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

        filename = './potential_saved'
        FIL = open(filename, 'w')
        FIL.write('ELEM_LIST ' + atom_type_str + '\n\n')

        for item in self.parent.inputs['atom_types']:
            FIL.write('POT {} {}\n'.format(item, np.max(self.params[item][:,3])))
            FIL.write('SYM {}\n'.format(len(self.params[item])))

            for ctem in self.params[item]:
                tmp_types = self.parent.inputs['atom_types'][int(ctem[1])-1]
                if int(ctem[0]) > 3:
                    tmp_types += ' {}'.format(self.parent.inputs['atom_types'][int(ctem[2])-1])

                FIL.write('{} {} {} {} {} {}\n'.\
                    format(int(ctem[0]), ctem[3], ctem[4], ctem[5], ctem[6], tmp_types))

            FIL.write('scale1 {}\n'.format(' '.join(self.scale[item][0,:].astype(np.str))))
            FIL.write('scale2 {}\n'.format(' '.join(self.scale[item][1,:].astype(np.str))))
            
            weights = sess.run(self.models[item].weights)
            nlayers = len(self.nodes[item])
            FIL.write('NET {} {}\n'.format(nlayers-1, ' '.join(map(str, self.nodes[item]))))

            for j in range(nlayers):
                # FIXME: add activation function type if new activation is added
                if j == nlayers-1:
                    acti = 'linear'
                else:
                    acti = 'sigmoid'

                FIL.write('LAYER {} {}\n'.format(j, acti))

                for k in range(self.nodes[item][j]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(weights[j*2][:,k].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, weights[j*2 + 1][k]))
            
            FIL.write('\n')

        FIL.close()

    def _save(self, sess, saver):
        if not self.inputs['continue']:
            self.inputs['continue'] = True
            self.parent.write_inputs()

        self.parent.logfile.write("Save the weights and write the LAMMPS potential..\n")              
        saver.save(sess, './SAVER')
        self._generate_lammps_potential(sess)
        

    def _make_iterator_from_handle(self, training_dataset, atomic_weights=False):
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, training_dataset.output_types, training_dataset.output_shapes)
        self.next_elem = self.iterator.get_next()

        # which place?
        self.next_elem['partition'] = tf.reshape(self.next_elem['partition'], [-1])
        self.next_elem['F'] = \
            tf.dynamic_partition(
                tf.reshape(self.next_elem['F'], [-1, 3]),
                self.next_elem['partition'], 2
            )[1]
        self.next_elem['num_seg'] = tf.shape(self.next_elem['tot_num'])[0] + 1
        
        if atomic_weights:
            self.next_elem['atomic_weights'] = \
                tf.dynamic_partition(
                    tf.reshape(self.next_elem['atomic_weights'], [-1, 1]),
                    self.next_elem['partition'], 2
                )[1]

        for item in self.parent.inputs['atom_types']:
            zero_cond = tf.equal(tf.reduce_sum(self.next_elem['N_'+item]), 0)

            self.next_elem['partition_'+item] = tf.cond(zero_cond, 
                                                        lambda: tf.zeros([0], tf.int32),
                                                        lambda: tf.reshape(self.next_elem['partition_'+item], [-1]))

            self.next_elem['x_'+item] = tf.cond(zero_cond, 
                                                lambda: tf.zeros([0, self.inp_size[item]], dtype=tf.float64),
                                                lambda: tf.dynamic_partition(
                                                            tf.reshape(self.next_elem['x_'+item], [-1, self.inp_size[item]]),
                                                            self.next_elem['partition_'+item], 2)[1])

            self.next_elem['x_'+item] -= self.scale[item][0:1,:]
            self.next_elem['x_'+item] /= self.scale[item][1:2,:]

            dx_shape = tf.shape(self.next_elem['dx_'+item])

            self.next_elem['dx_'+item] = tf.cond(zero_cond, 
                                                 lambda: tf.zeros([0, dx_shape[2], 0, dx_shape[4]], dtype=tf.float64), 
                                                 lambda: tf.dynamic_partition(tf.reshape(self.next_elem['dx_'+item], [-1, dx_shape[2], dx_shape[3], dx_shape[4]]),
                                                                              self.next_elem['partition_'+item], 2
                                                                              )[1])

            max_totnum = tf.cast(tf.reduce_max(self.next_elem['tot_num']), tf.int32)
            self.next_elem['dx_'+item] = tf.cond(tf.equal(tf.shape(self.next_elem['dx_'+item])[2], max_totnum),
                                                 lambda: self.next_elem['dx_'+item],
                                                 lambda: tf.pad(self.next_elem['dx_'+item], 
                                                                [[0, 0], [0, 0], [0, max_totnum-tf.shape(self.next_elem['dx_'+item])[2]], [0,0]]))

            self.next_elem['dx_'+item] /= self.scale[item][1:2,:].reshape([1, self.inp_size[item], 1, 1])

            self.next_elem['seg_id_'+item] = tf.cond(zero_cond,
                                                     lambda: tf.zeros([0], tf.int32), 
                                                     lambda: tf.dynamic_partition(tf.reshape(tf.map_fn(lambda x: tf.tile([x+1], [dx_shape[1]]), 
                                                                                             tf.range(tf.shape(self.next_elem['N_'+item])[0])), [-1]),
                                                                                  self.next_elem['partition_'+item], 2)[1])

            self.next_elem['sparse_indices_'+item] = tf.cast(tf.range(tf.reduce_sum(self.next_elem['N_'+item])), tf.int32)


    def train(self, user_optimizer=None, user_atomic_weights_function=None):
        self.inputs = self.parent.inputs['neural_network']
        # read data?

        self._set_params('symmetry_function')
        self._set_scale_parameter('./scale_factor')

        if self.inputs['train']:
            train_filequeue = _make_data_list(self.train_data_list)
            valid_filequeue = _make_data_list(self.valid_data_list)
        
            if self.parent.descriptor.inputs['atomic_weights']['type'] == None:
                aw_tag = False
            else:
                aw_tag = True

            train_iter = self.parent.descriptor._tfrecord_input_fn(train_filequeue, self.inp_size, 
                                                                   batch_size=self.inputs['batch_size'], full_batch=self.inputs['full_batch'], atomic_weights=aw_tag)
            valid_iter = self.parent.descriptor._tfrecord_input_fn(valid_filequeue, self.inp_size, 
                                                                   batch_size=self.inputs['batch_size'], valid=True, atomic_weights=aw_tag)
#                                                                   valid=True, atomic_weights=aw_tag)
            self._make_iterator_from_handle(train_iter, aw_tag)

        if self.inputs['test']:
            test_filequeue = _make_data_list(self.test_data_list)
            test_iter = self.parent.descriptor._tfrecord_input_fn(test_filequeue, self.inp_size, batch_size=self.inputs['batch_size'], valid=True, atomic_weights=False)
            if not self.inputs['train']:
                self._make_iterator_from_handle(test_iter)

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
            saver = tf.train.Saver()
            if self.inputs['continue']:
                saver.restore(sess, './SAVER')
            else:
                sess.run(tf.global_variables_initializer())

            #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            #run_metadata = tf.RunMetadata()

            if self.inputs['train']:
                if self.inputs['method'] == 'L-BFGS-B':
                    # TODO: complete this part
                    raise ValueError

                elif self.inputs['method'] == 'Adam':
                    train_handle = sess.run(train_iter.string_handle())
                    train_fdict = {self.handle: train_handle}
                    if self.inputs['full_batch']:
                        self.grad_ph = list()
                        full_batch_dict = dict()
                        for i,item in enumerate(self.compute_grad):
                            self.grad_ph.append(tf.placeholder(tf.float64, sess.run(tf.shape(item[1]))))
                            full_batch_dict[self.grad_ph[i]] = None

                        self.apply_grad = self.optim.apply_gradients([(self.grad_ph[i], item[1]) for i,item in enumerate(self.compute_grad)],
                                                                     global_step=self.global_step)
                    else:
                        sess.run(train_iter.initializer)

                    valid_handle = sess.run(valid_iter.string_handle())
                    valid_fdict = {self.handle: valid_handle}
                    #sess.run(valid_iter.initializer)

                    for epoch in range(self.inputs['total_epoch']):
                        time1 = timeit.default_timer()
                        if self.inputs['full_batch']:
                            sess.run(train_iter.initializer)
                            #grad_and_vars = sess.run(self.compute_grad, feed_dict=train_fdict)
                            for i,item in enumerate(sess.run(self.compute_grad, feed_dict=train_fdict)):
                                full_batch_dict[self.grad_ph[i]] = item[0]
                            while True:
                                try:
                                    #grad_and_vars = [(item[0] + jtem[0], item[1]) for item, jtem in zip(grad_and_vars, sess.run(self.compute_grad, feed_dict=train_fdict))]
                                    #grad_and_vars = sess.run(self.compute_grad, feed_dict=train_fdict)
                                    for i,item in enumerate(sess.run(self.compute_grad, feed_dict=train_fdict)):
                                        full_batch_dict[self.grad_ph[i]] += item[0]
                                except tf.errors.OutOfRangeError:
                                    # FIXME: memory leak?
                                    #grad_and_vars = [(item[0], jtem[1]) for item, jtem in zip(grad_and_vars, self.compute_grad)]
                                    #for v,gandv in enumerate(self.grad_and_vars):
                                    #    gandv[0] = grad_and_vars[v][0]

                                    # FIXME: in this part, new model is builded and not removed
                                    #sess.run(self.optim.apply_gradients(self.grad_and_vars, global_step=self.global_step))
                                    sess.run(self.apply_grad, feed_dict=full_batch_dict)
                                    break
                        else:
                            self.minim.run(feed_dict=train_fdict)
                        #sess.run(self.optim, feed_dict=train_fdict, options=options, run_metadata=run_metadata)
                        time2 = timeit.default_timer()

                        # Logging
                        if (epoch+1) % self.inputs['show_interval'] == 0:
                            # Profiling
                            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            #with open('timeline_test.json', 'w') as fil:
                            #    fil.write(chrome_trace)

                            # TODO: need to fix the calculation part for training loss
                            result = "epoch {:7d}: ".format(sess.run(self.global_step)+1)

                            if self.inputs['full_batch']:
                                sess.run(train_iter.initializer)
                                t_eloss = t_floss = 0
                                train_total = 0
                                while True:
                                    try:
                                        if self.inputs['use_force']:
                                            valid_elem, tmp_eloss, tmp_floss = sess.run([self.next_elem, self.e_loss, self.f_loss], feed_dict=train_fdict)
                                            num_batch = valid_elem['num_seg'] - 1
                                            t_eloss += tmp_eloss * num_batch
                                            t_floss += tmp_floss * num_batch
                                        else:
                                            valid_elem, tmp_eloss = sess.run([self.next_elem, self.e_loss], feed_dict=train_fdict)
                                            num_batch = valid_elem['num_seg'] - 1
                                            t_eloss += tmp_eloss * num_batch

                                        train_total += num_batch
                                    except tf.errors.OutOfRangeError:
                                        t_eloss = np.sqrt(t_eloss/train_total)
                                        if self.inputs['use_force']:
                                            t_floss = np.sqrt(t_floss*3/train_total)
                                        break
                            else:
                                t_eloss = sess.run(self.e_loss, feed_dict=train_fdict)
                                t_eloss = np.sqrt(t_eloss)
                                if self.inputs['use_force']:
                                    t_floss = sess.run(self.f_loss, feed_dict=train_fdict)
                                    t_floss = np.sqrt(t_floss*3)

                            sess.run(valid_iter.initializer)
                            eloss = floss = 0
                            valid_total = 0
                            while True:
                                try:
                                    if self.inputs['use_force']:
                                        valid_elem, tmp_eloss, tmp_floss = sess.run([self.next_elem, self.e_loss, self.f_loss], feed_dict=valid_fdict)
                                        num_batch = valid_elem['num_seg'] - 1
                                        eloss += tmp_eloss * num_batch
                                        floss += tmp_floss * num_batch
                                    else:
                                        valid_elem, tmp_eloss = sess.run([self.next_elem, self.e_loss], feed_dict=valid_fdict)
                                        num_batch = valid_elem['num_seg'] - 1
                                        eloss += tmp_eloss * num_batch

                                    valid_total += num_batch
                                except tf.errors.OutOfRangeError:
                                    eloss = np.sqrt(eloss/valid_total)
                                    result += 'E RMSE(T V) = {:6.4e} {:6.4e}'.format(t_eloss,eloss)
                                    if self.inputs['use_force']:
                                        floss = np.sqrt(floss*3/valid_total)
                                        result += ', F RMSE(T V) = {:6.4e} {:6.4e}'.format(t_floss,floss)
                                    break

                            lr = sess.run(self.learning_rate)
                            result += ', learning_rate: {:6.4e}'.format(lr)
                            result += ', elapsed: {:4.2e}\n'.format(time2-time1)
                            self.parent.logfile.write(result)

                        # Temp saving
                        if (epoch+1) % self.inputs['save_interval'] == 0:
                            self._save(sess, saver)

                self._save(sess, saver)

            if self.inputs['test']:
                test_handle = sess.run(test_iter.string_handle())
                test_fdict = {self.handle: test_handle}
                sess.run(test_iter.initializer)

                test_save = dict()
                test_save['DFT_E'] = list()
                test_save['NN_E'] = list()
                test_save['N'] = list()

                if self.inputs['use_force']:
                    test_save['DFT_F'] = list()
                    test_save['NN_F'] = list()

                eloss = floss = 0.
                test_total = 0
                result = ' Test'
                while True:
                    try:
                        if self.inputs['use_force']:
                            test_elem, tmp_nne, tmp_nnf, tmp_eloss, tmp_floss = \
                                sess.run([self.next_elem, self.E, self.F, self.e_loss, self.f_loss], feed_dict=test_fdict)
                            num_batch = test_elem['num_seg'] - 1
                            eloss += tmp_eloss * num_batch
                            floss += tmp_floss * num_batch

                            test_save['DFT_E'].append(test_elem['E'])
                            test_save['NN_E'].append(tmp_nne)
                            test_save['N'].append(test_elem['tot_num'])
                            test_save['DFT_F'].append(test_elem['F'])
                            test_save['NN_F'].append(tmp_nnf)
                        else:
                            test_elem, tmp_nne, tmp_eloss = \
                                sess.run([self.next_elem, self.E, self.e_loss], feed_dict=test_fdict)
                            num_batch = valid_elem['num_seg'] - 1
                            eloss += tmp_eloss * num_batch

                            test_save['DFT_E'].append(test_elem['E'])
                            test_save['NN_E'].append(tmp_nne)
                            test_save['N'].append(test_elem['tot_num'])

                        test_total += num_batch
                    except tf.errors.OutOfRangeError:
                        eloss = np.sqrt(eloss/test_total)
                        result += 'E RMSE = {:6.4e}'.format(eloss)

                        test_save['DFT_E'] = np.concatenate(test_save['DFT_E'], axis=0)
                        test_save['NN_E'] = np.concatenate(test_save['NN_E'], axis=0)
                        test_save['N'] = np.concatenate(test_save['N'], axis=0)
                        if self.inputs['use_force']:
                            floss = np.sqrt(floss*3/test_total)
                            result += ', F RMSE = {:6.4e}'.format(floss)

                            test_save['DFT_F'] = np.concatenate(test_save['DFT_F'], axis=0)
                            test_save['NN_F'] = np.concatenate(test_save['NN_F'], axis=0)
                        break
                
                with open('./test_result', 'wb') as fil:
                    pickle.dump(test_save, fil, pickle.HIGHEST_PROTOCOL)

                self.parent.logfile.write('Test result saved..\n')
                self.parent.logfile.write(result + '\n')


