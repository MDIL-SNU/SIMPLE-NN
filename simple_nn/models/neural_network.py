import tensorflow as tf
import numpy as np
import random
import six
from six.moves import cPickle as pickle
import collections
import functools
import timeit
from ..utils import _make_data_list, pickle_load, preprocessing, _generate_gdf_file, modified_sigmoid

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
                                      'loss_scale': 1.,
                                      'double_precision': True,
                                      'valid_rate': 0.1,
                                      'generate_valid': True,
                                      'scale': True,
                                      'learning_rate': 0.01,
                                      'optimizer': dict(),
                                      'nodes': '30-30',
                                      'test': False,
                                      'train': True,
                                      'atomic_weights': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'weight_modifier': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'regularization': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                  }
                              }
        self.inputs = dict()
        self.global_step = tf.Variable(0, trainable=False)
        self.train_data_list = './train_list'
        self.valid_data_list = './valid_list'
        self.test_data_list = './test_list'

    def _make_fileiter(self):
        """
        function to generate iterator for input data file list.
        for training set, infinite random iterator is generated and
        for validation/test set, normal iterator is generated.
        """
        # TODO: check tf.data
        if self.inputs['generate_valid']:
            with open(self.train_data_list, 'r') as fil:
                full_data = fil.readlines()

            random.shuffle(full_data)
            valid_idx = int(len(full_data) * self.inputs['valid_rate'])

            with open(self.valid_data_list, 'w') as fil:
                for item in full_data[:valid_idx]:
                    fil.write(item)

            with open(self.train_data_list, 'w') as fil:
                for item in full_data[valid_idx:]:
                    fil.write(item)

            self.inputs['generate_valid'] = False
            self.parent.write_inputs()

        class iterfile(object):
            def __init__(self, filelist, maxiter=None):
                self.items = filelist
                self.index = 0

                if maxiter is None:
                    self.maxiter = len(filelist)
                else:
                    self.maxiter = maxiter
                
                self.curiter = 0

            def __iter__(self):
                self.curiter = 0
                return self
            
            def set_maxiter(self, maxiter):
                self.maxiter = maxiter

            def _next(self):
                if self.index == 0:
                    random.shuffle(self.items)
        
                n = self.items[self.index]

                if self.curiter == self.maxiter:
                    raise StopIteration
                else:
                    self.curiter += 1

                self.index += 1
                self.index = self.index % len(self.items)
                return n

            if six.PY2:
                def next(self):
                    return self._next()
            elif six.PY3:
                def __next__(self):
                    return self._next()

        if self.inputs['train']:
            train_data = _make_data_list(self.train_data_list)
            train_data = list(zip(train_data, list(range(len(train_data)))))
            train_data = iterfile(train_data)
            valid_data = _make_data_list(self.valid_data_list)
            valid_data = list(zip(valid_data, list(range(len(valid_data)))))
        else:
            train_data = valid_data = None

        if self.inputs['test']:
            test_data = _make_data_list(self.test_data_list)
            test_data = list(zip(test_data, list(range(len(test_data)))))
        else:
            test_data = None
 
        return train_data, valid_data, test_data

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

    def _get_batch(self, fileiter, valid=False):
        batch = {
            'x': dict(),
            'dx': dict(),
            '_E': list(),
            '_F': list(),
            'N': dict(),
            'seg_id': dict()
        }
        
        tag_atomic_weights = self.inputs['atomic_weights']['type']
        if tag_atomic_weights != None:
            batch['atomic_weights'] = list()

        for item in self.parent.inputs['atom_types']:
            batch['x'][item] = list()
            batch['dx'][item] = list()
            batch['N'][item] = list()
        
        for item in fileiter:
            loaded_fil = pickle_load(item[0])
            tmp_atom_types = loaded_fil['x'].keys()

            # TODO: add parameter check part
            batch['_E'].append(loaded_fil['E'])
            batch['_F'].append(loaded_fil['F'])
            for jtem in self.parent.inputs['atom_types']:
                if jtem in tmp_atom_types:
                    batch['x'][jtem].append(loaded_fil['x'][jtem])
                    batch['dx'][jtem].append(loaded_fil['dx'][jtem])
                    batch['N'][jtem].append(loaded_fil['N'][jtem])
                    if tag_atomic_weights != None:
                        if valid:
                            batch['atomic_weights'].append([1.]*loaded_fil['N'][jtem])
                        else:
                            batch['atomic_weights'].\
                                append(self.atomic_weights_full[jtem][self.atomic_weights_full[jtem][:,1] == item[1],1])
                else:
                    batch['x'][jtem].append(np.zeros([0, self.inp_size[jtem]], dtype=np.float64))
                    batch['dx'][jtem].append(np.zeros([0, self.inp_size[jtem], int(np.sum(list(loaded_fil['N'].values()))), 3], dtype=np.float64))
                    batch['N'][jtem].append(0)

        batch['_E'] = np.array(batch['_E'], dtype=np.float64).reshape([-1,1])
        batch['_F'] = np.concatenate(batch['_F']).astype(np.float64)
        if tag_atomic_weights != None:
            batch['atomic_weights'] = np.concatenate(batch['atomic_weights']).astype(np.float64).reshape([-1,1])

        batch['tot_num'] = np.sum(list(batch['N'].values()), axis=0)
        max_atom_num = np.max(batch['tot_num'])
        batch['max_atom_num'] = max_atom_num

        for item in self.parent.inputs['atom_types']:
            batch['N'][item] = np.array(batch['N'][item], dtype=np.int)
            batch['x'][item] = np.concatenate(batch['x'][item], axis=0).astype(np.float64)
            batch['x'][item] -= self.scale[item][0:1,:]
            batch['x'][item] /= self.scale[item][1:2,:]

            tmp_dx = np.zeros([np.sum(batch['N'][item]), self.inp_size[item],\
                               max_atom_num, 3], dtype=np.float64)

            tmp_idx = 0
            for jtem in batch['dx'][item]:
                tmp_dx[tmp_idx:tmp_idx+jtem.shape[0],:,:jtem.shape[2],:] = jtem
                tmp_idx += jtem.shape[0]
            batch['dx'][item] = tmp_dx / self.scale[item][1:2,:].reshape([1,self.inp_size[item],1,1])

            batch['seg_id'][item] = \
                np.concatenate([[j]*jtem for j,jtem in enumerate(batch['N'][item])]) + 1.

        batch['partition'] = \
            np.concatenate([[0]*item + [1]*(max_atom_num - item) for item in batch['tot_num']])
        
        return batch

    def _set_scale_parameter(self, scale_file, gdf_file=None):
        self.scale = pickle_load(scale_file)
        # TODO: add the check code for valid scale file
        self.gdf = pickle_load(gdf_file)

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

        if self.inputs['regularization']['type'] is not None:
            if self.inputs['regularization']['type'] == 'l2':
                coeff = self.inputs['regularization']['params'].get('coeff', 1e-6)
                dense_basic_setting['kernel_regularizer'] = tf.keras.regularizers.l2(l=coeff)
                dense_basic_setting['bias_regularizer'] = tf.keras.regularizers.l2(l=coeff)

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
                                            **dense_basic_setting))

            nodes.append(1)
            self.nodes[item] = nodes

            self.models[item] = model
            self.ys[item] = self.models[item](self.x[item])

            if self.inputs['use_force']:
                self.dys[item] = tf.gradients(self.ys[item], self.x[item])[0]
            else:
                self.dys[item] = None


    def _calc_output(self):
        self.E = self.F = 0

        for item in self.parent.inputs['atom_types']:
            #self.E += tf.segment_sum(self.ys[item], self.seg_id[item])
            self.E += tf.sparse_segment_sum(self.ys[item], self.sparse_indices[item], self.seg_id[item], 
                                            num_segments=self.num_seg)[1:]

            if self.inputs['use_force']:
                tmp_force = self.dx[item] * \
                            tf.expand_dims(\
                                tf.expand_dims(self.dys[item], axis=2),
                                axis=3)
                tmp_force = tf.reduce_sum(\
                                tf.sparse_segment_sum(tmp_force, self.sparse_indices[item], self.seg_id[item], 
                                                      num_segments=self.num_seg)[1:],
                                axis=1)
                self.F -= tf.dynamic_partition(tf.reshape(tmp_force, [-1,3]),
                                                   self.partition, 2)[0]

    def _get_loss(self, use_gdf=False, atomic_weights=None):
        self.e_loss = tf.reduce_mean(tf.square((self._E - self.E) / self.tot_num))
        self.total_loss = self.e_loss * self.energy_coeff

        if self.inputs['use_force']:
            self.f_loss = tf.square(self._F - self.F)
            if self.inputs['atomic_weights']['type'] is not None:
                self.f_loss *= self.atomic_weights
            self.f_loss = tf.reduce_mean(self.f_loss)
        
            self.total_loss += self.f_loss * self.force_coeff

        if self.inputs['regularization']['type'] is not None:
            self.total_loss += tf.losses.get_regularization_loss()

    def _make_optimizer(self, user_optimizer=None):
        final_loss = self.inputs['loss_scale']*self.total_loss
        if self.inputs['method'] == 'L-BFGS-B':
            self.optim = tf.contrib.opt.ScipyOptimizerInterface(final_loss, 
                                                                method=self.inputs['method'], 
                                                                options=self.inputs['optimizer'])
        elif self.inputs['method'] == 'Adam':
            if isinstance(self.inputs['learning_rate'], collections.Mapping):
                self.learning_rate = tf.train.exponential_decay(global_step=self.global_step, **self.inputs['learning_rate'])
            else:
                self.learning_rate = tf.constant(self.inputs['learning_rate'])

            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                name='Adam', **self.inputs['optimizer'])
            self.optim = self.optim.minimize(final_loss, global_step=self.global_step)
        else:
            if user_optimizer != None:
                self.optim = user_optimizer.minimize(final_loss, global_step=self.global_step)
                #self.optim = user_optimizer
            else:
                raise ValueError

    def _make_feed_dict(self, batch):
        fdict = {
            self._E: batch['_E'],
            self._F: batch['_F'],
            self.tot_num: batch['tot_num'],
            self.partition: batch['partition'],
            self.max_atom_num: batch['max_atom_num']
        }

        if self.inputs['atomic_weights']['type'] != None:
            fdict[self.atomic_weights] = batch['atomic_weights']

        fdict[self.num_seg] = len(batch['_E']) + 1
        #fdict[self.num_seg] = self.inputs['batch_size'] + 1

        for item in self.parent.inputs['atom_types']:
            if batch['x'][item].shape[0] > 0:
                fdict[self.x[item]] = batch['x'][item]
                fdict[self.dx[item]] = batch['dx'][item]
                fdict[self.seg_id[item]] = batch['seg_id'][item]
                fdict[self.sparse_indices[item]] = list(range(len(batch['x'][item])))

        return fdict 

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

    def _parse_data(self, serialized, preprocess=False):
        features = {
            'E': tf.FixedLenFeature([], dtype=tf.string),
            'F': tf.FixedLenFeature([], dtype=tf.string),
        }

        for item in self.parent.inputs['atom_types']:
            features['x_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['N_'+item] = tf.VarLenFeature(tf.int64)
            features['params_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['dx_indices_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['dx_values_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['dx_dense_shape_'+item] = tf.FixedLenFeature([], dtype=tf.string)

        read_data = tf.parse_single_example(serialized=serialized, features=features)

        if preprocess:
            res = dict()

            for item in self.parent.inputs['atom_types']:
                res['x_'+item] = tf.reshape(tf.decode_raw(read_data['x'], tf.float64), [-1, self.inp_size[item]])

            return res
        else:
            padded_res = dict()
            sparse_res = dict()

            padded_res['E'] = tf.decode_raw(read_data['E'], tf.float64)
            padded_res['F'] = tf.reshape(tf.decode_raw(read_data['F'], tf.float64), [-1, 3])
            for item in self.parent.inputs['atom_types']:
                padded_res['x_'+item] = tf.reshape(tf.decode_raw(read_data['x_'+item], tf.float64), [-1, self.inp_size[item]])
                padded_res['N_'+item] = read_data['N_'+item]

                sparse_res['dx_'+item] = tf.SparseTensor(
                    indices=np.frombuffer(read_data['dx_indices_'+item], dtype=np.uint32).astype(np.int64),
                    values=tf.decode_raw(read_data['dx_values_'+item], tf.float64),
                    dense_shape=tf.decode_raw(read_data['dx_dense_shape_'+item], tf.int64)
                )

            # TODO: seg_id, dynamic_partition_id, 

            return padded_res, sparse_res

    def _tfrecord_input_fn(self, filename_queue, batch_size, preprocess=False):

        return 0#iterator
        


    def train(self, user_optimizer=None, user_atomic_weights_function=None):
        self.inputs = self.parent.inputs['neural_network']
        # read data?
        
        train_fileiter, valid_fileiter, test_fileiter = self._make_fileiter()

        # preprocessing: scale, GDF...
        modifier = None
        if self.inputs['weight_modifier']['type'] == 'modified sigmoid':
            modifier = functools.partial(modified_sigmoid, **self.inputs['weight_modifier']['params'])
        if self.inputs['atomic_weights']['type'] == 'gdf':
            get_atomic_weights = functools.partial(_generate_gdf_file, modifier=modifier)
        elif self.inputs['atomic_weights']['type'] == 'user':
            get_atomic_weights = user_atomic_weights_function
        elif self.inputs['atomic_weights']['type'] == 'file':
            get_atomic_weights = './atomic_weights'
        else:
            get_atomic_weights = None

        self._set_params('symmetry_function')

        # FIXME: Error occur: only test
        if self.inputs['train']:
            self.scale, self.atomic_weights_full = \
                preprocessing(self.train_data_list, self.parent.inputs['atom_types'], 'x', self.inp_size,\
                            calc_scale=self.inputs['scale'], \
                            get_atomic_weights=get_atomic_weights, \
                            **self.inputs['atomic_weights']['params'])

            #self._get_batch(train_fileiter, 1, initial=True)
            #self._set_params(train_fileiter)
            train_fileiter.set_maxiter(self.inputs['batch_size'])
        else:
            self.scale, self.atomic_weights_full = \
                preprocessing(self.test_data_list, self.parent.inputs['atom_types'], 'x', self.inp_size,\
                            calc_scale=False, get_atomic_weights=None)
            #self._get_batch(test_fileiter, 1, initial=True, valid=True)
            #self._set_params(test_fileiter)

        # Generate placeholder
        self._E = tf.placeholder(tf.float64, [None, 1])
        self._F = tf.placeholder(tf.float64, [None, 3])
        self.tot_num = tf.placeholder(tf.float64, [None])
        self.partition = tf.placeholder(tf.int32, [None])
        self.seg_id = dict()
        self.sparse_indices = dict()
        self.x = dict()
        self.dx = dict()
        self.num_seg = tf.placeholder(tf.int32, ())
        self.max_atom_num = tf.placeholder(tf.int32, ())
        self.atomic_weights = tf.placeholder(tf.float64, [None, 1]) \
                                if self.inputs['atomic_weights']['type'] != None else None
        for item in self.parent.inputs['atom_types']:
            self.x[item] = tf.placeholder_with_default(tf.zeros([1, self.inp_size[item]], dtype=tf.float64), 
                                                       [None, self.inp_size[item]])
            self.dx[item] = tf.placeholder_with_default(tf.zeros([1, self.inp_size[item], self.max_atom_num, 3], dtype=tf.float64), 
                                                        [None, self.inp_size[item], None, 3])
            self.seg_id[item] = tf.placeholder_with_default(tf.constant([0], dtype=tf.int32), [None])
            self.sparse_indices[item] = tf.placeholder_with_default(tf.constant([0], dtype=tf.int32), [None])

        self.force_coeff = self._get_decay_param(self.inputs['force_coeff'])
        self.energy_coeff = self._get_decay_param(self.inputs['energy_coeff'])

        self._make_model()
        self._calc_output()
        self._get_loss()
        self._make_optimizer(user_optimizer=user_optimizer)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.45
        with tf.Session(config=config) as sess:
            # Load or initialize the variables
            saver = tf.train.Saver()
            if self.inputs['continue']:
                saver.restore(sess, './SAVER')
            else:
                sess.run(tf.global_variables_initializer())

            if self.inputs['train']:
                if self.inputs['method'] == 'L-BFGS-B':
                    # TODO: complete this part
                    raise ValueError

                    train_set = self._get_batch(train_fileiter, valid=True)
                    train_fdict = self._make_feed_dict(train_set)
                    valid_set = self._get_batch(valid_fileiter, valid=True)
                    valid_fdict = self._make_feed_dict(valid_set)
                elif self.inputs['method'] == 'Adam':
                    valid_set = self._get_batch(valid_fileiter, valid=True)
                    valid_fdict = self._make_feed_dict(valid_set)

                    for epoch in range(self.inputs['total_epoch']):
                        time1 = timeit.default_timer()
                        train_batch = self._get_batch(train_fileiter)
                        train_fdict = self._make_feed_dict(train_batch)
                        self.optim.run(feed_dict=train_fdict)
                        time2 = timeit.default_timer()

                        # Logging
                        if (epoch+1) % self.inputs['show_interval'] == 0:
                            result = "epoch {:7d}: ".format(sess.run(self.global_step)+1)

                            eloss = sess.run(self.e_loss, feed_dict=valid_fdict)
                            eloss = np.sqrt(eloss)
                            t_eloss = sess.run(self.e_loss, feed_dict=train_fdict)
                            t_eloss = np.sqrt(t_eloss)
                            result += 'E loss(T V) = {:6.4e} {:6.4e}'.format(t_eloss,eloss)

                            if self.inputs['use_force']:
                                floss = sess.run(self.f_loss, feed_dict=valid_fdict)
                                floss = np.sqrt(floss*3)
                                t_floss = sess.run(self.f_loss, feed_dict=train_fdict)
                                t_floss = np.sqrt(t_floss*3)
                                result += ', F loss(T V) = {:6.4e} {:6.4e}'.format(t_floss,floss)

                            lr = sess.run(self.learning_rate)
                            result += ', learning_rate: {:6.4e}'.format(lr)
                            result += ', elapsed: {:4.2e}\n'.format(time2-time1)
                            self.parent.logfile.write(result)

                        # Temp saving
                        if (epoch+1) % self.inputs['save_interval'] == 0:
                            self._save(sess, saver)

                self._save(sess, saver)

            if self.inputs['test']:
                test_set = self._get_batch(test_fileiter, valid=True)
                test_fdict = self._make_feed_dict(test_set)

                test_save = dict()
                test_save['DFT_E'] = test_set['_E']
                test_save['NN_E'] = sess.run(self.E, feed_dict=test_fdict)

                eloss = sess.run(self.e_loss, feed_dict=test_fdict)
                eloss = np.sqrt(eloss)

                if self.inputs['use_force']:
                    test_save['DFT_F'] = test_set['_F']
                    test_save['NN_F'] = sess.run(self.F, feed_dict=test_fdict)

                    floss = sess.run(self.f_loss, feed_dict=test_fdict)
                    floss = np.sqrt(floss*3)

                with open('./test_result', 'wb') as fil:
                    pickle.dump(test_save, fil, pickle.HIGHEST_PROTOCOL)

                self.parent.logfile.write('Test result saved..\n')
                self.parent.logfile.write(' Test E RMSE: {}, F RMSE: {}\n'.format(eloss, floss))

                
    
